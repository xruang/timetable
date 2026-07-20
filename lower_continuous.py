"""Continuous-action lower controllers for eco-driving comparisons."""

from __future__ import annotations

import random
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

try:
    from .config import EPS
    from .lower_control import LowerSectionEnvironment
    from .train_model import (
        TRAIN_CONFIGS,
        TrainDynamics,
        create_train_config_from_mass,
    )
except ImportError:
    from config import EPS
    from lower_control import LowerSectionEnvironment
    from train_model import TRAIN_CONFIGS, TrainDynamics, create_train_config_from_mass


class ContinuousReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)

    def push(self, *transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int):
        state, action, reward, next_state, done = zip(
            *random.sample(self.buffer, batch_size)
        )
        return (
            torch.as_tensor(np.asarray(state), dtype=torch.float32),
            torch.as_tensor(np.asarray(action), dtype=torch.float32).unsqueeze(1),
            torch.as_tensor(reward, dtype=torch.float32).unsqueeze(1),
            torch.as_tensor(np.asarray(next_state), dtype=torch.float32),
            torch.as_tensor(done, dtype=torch.float32).unsqueeze(1),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DDPGActor(nn.Module):
    def __init__(self, state_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.layers(state)


class DDPGCritic(nn.Module):
    def __init__(self, state_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.layers(torch.cat((state, action), dim=1))


class SACActor(nn.Module):
    def __init__(self, state_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.mean = nn.Linear(128, 1)
        self.log_std = nn.Linear(128, 1)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.layers(state)
        mean = self.mean(features)
        log_std = torch.clamp(self.log_std(features), -5.0, 1.0)
        return mean, log_std

    def deterministic_action(self, state: torch.Tensor) -> torch.Tensor:
        mean, _ = self.forward(state)
        return torch.tanh(mean)

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        raw_action = normal.rsample()
        action = torch.tanh(raw_action)
        log_probability = normal.log_prob(raw_action) - torch.log(
            1.0 - action.pow(2) + 1e-6
        )
        return action, log_probability.sum(dim=1, keepdim=True)


class SACCritic(nn.Module):
    def __init__(self, state_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.layers(torch.cat((state, action), dim=1))


def force_from_ratio(environment: LowerSectionEnvironment, ratio: float) -> float:
    cfg = environment.dynamics.config
    ratio = environment.clip_force_ratio(float(np.clip(ratio, -1.0, 1.0)))
    if ratio >= 0.0:
        return ratio * environment.dynamics.max_traction_force_n(environment.speed_mps)
    return environment.clip_force_to_action_space_n(ratio * cfg.max_braking_force_n)


def _copy_target_if_present(controller: Any) -> None:
    if hasattr(controller, "target_actor"):
        controller.target_actor.load_state_dict(controller.actor.state_dict())
    if hasattr(controller, "target_critic"):
        controller.target_critic.load_state_dict(controller.critic.state_dict())
    if hasattr(controller, "target_critic1"):
        controller.target_critic1.load_state_dict(controller.critic1.state_dict())
    if hasattr(controller, "target_critic2"):
        controller.target_critic2.load_state_dict(controller.critic2.state_dict())


def _safe_stem(value: str) -> str:
    return "".join(
        char if char.isalnum() or char in {"-", "_"} else "_"
        for char in str(value)
    ).strip("_")


def plot_lower_training_speed_snapshot(
    result: Any,
    output_dir: Optional[Path],
    algorithm_name: str,
    train_name: str,
    episode: int,
) -> None:
    if output_dir is None or not result.trajectory:
        return
    snapshot_dir = Path(output_dir) / "lower_training_speed_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    mileage_km = [
        (point.position_m - result.trajectory[0].position_m) / 1000.0
        for point in result.trajectory
    ]
    speed_kmh = [point.speed_kmh for point in result.trajectory]
    limit_kmh = [point.speed_limit_kmh for point in result.trajectory]
    force_kn = [point.force_kn for point in result.trajectory]
    elevations_m = [0.0]
    for previous, current in zip(result.trajectory[:-1], result.trajectory[1:]):
        delta_m = max(0.0, current.position_m - previous.position_m)
        elevations_m.append(
            elevations_m[-1] + previous.gradient_per_mille / 1000.0 * delta_m
        )

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    speed_line = axes[0].plot(mileage_km, speed_kmh, label="Train speed", linewidth=1.8)
    limit_line = axes[0].step(
        mileage_km,
        limit_kmh,
        where="post",
        linestyle="--",
        color="tab:red",
        label="Speed limit",
        linewidth=1.2,
    )
    axes[0].set_ylabel("Speed (km/h)")
    axes[0].grid(alpha=0.25)
    elevation_axis = axes[0].twinx()
    elevation_line = elevation_axis.plot(
        mileage_km,
        elevations_m,
        color="#8A8A8A",
        alpha=0.65,
        linewidth=1.2,
        label="Line elevation",
    )
    elevation_axis.axhline(0.0, color="#BDBDBD", linewidth=0.5, alpha=0.65)
    elevation_axis.set_ylabel("Relative elevation (m)")
    lower = min(elevations_m)
    upper = max(elevations_m)
    if upper - lower < 1.0:
        lower -= 0.5
        upper += 0.5
    padding = max(1.0, 0.08 * (upper - lower))
    elevation_axis.set_ylim(lower - padding, upper + padding)
    axes[0].legend(speed_line + limit_line + elevation_line, [
        item.get_label() for item in speed_line + limit_line + elevation_line
    ], loc="best")

    axes[1].plot(mileage_km, force_kn, color="tab:green", label="Control force")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Section mileage (km)")
    axes[1].set_ylabel("Force (kN)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")

    fig.suptitle(
        f"{algorithm_name.upper()} episode {episode}: {train_name} "
        f"{result.section_name} target={result.target_time_s:.1f}s "
        f"actual={result.actual_time_s:.1f}s"
    )
    fig.tight_layout()
    stem = (
        f"{algorithm_name}_ep{episode:05d}_"
        f"{_safe_stem(train_name)}_{_safe_stem(result.section_name)}"
    )
    fig.savefig(snapshot_dir / f"{stem}.png", dpi=160)
    plt.close(fig)


class DDPGAgent:
    """Continuous force-ratio lower agent.

    The actor outputs a value in [-1, 1]. Negative values mean braking, positive
    values mean traction, and values near zero mean coasting.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 1,
        actor_learning_rate: float = 2e-4,
        critic_learning_rate: float = 8e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        replay_capacity: int = 100000,
        batch_size: int = 128,
        device_name: Optional[str] = None,
    ) -> None:
        if action_dim != 1:
            raise ValueError("DDPGAgent expects one continuous force-ratio action")
        if device_name is None or device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.batch_size = int(batch_size)
        self.actor = DDPGActor(state_dim).to(self.device)
        self.target_actor = DDPGActor(state_dim).to(self.device)
        self.critic = DDPGCritic(state_dim).to(self.device)
        self.target_critic = DDPGCritic(state_dim).to(self.device)
        _copy_target_if_present(self)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)
        self.replay = ContinuousReplayBuffer(replay_capacity)
        self.exploration_noise_std = 0.20
        self.last_action_ratio = 0.0
        self.last_loss: Optional[float] = None
        self.last_actor_loss: Optional[float] = None
        self.policy = self.actor

    def set_exploration_noise(self, std: float) -> None:
        self.exploration_noise_std = max(0.0, float(std))

    def observation_from_environment(
        self,
        environment: LowerSectionEnvironment,
    ) -> np.ndarray:
        return environment.compact_observation()

    @staticmethod
    def force_from_ratio(environment: LowerSectionEnvironment, ratio: float) -> float:
        return force_from_ratio(environment, ratio)

    def select_action_and_force(
        self,
        environment: LowerSectionEnvironment,
    ) -> Tuple[int, float]:
        state = self.observation_from_environment(environment)
        with torch.no_grad():
            tensor = torch.as_tensor(
                state,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            ratio = float(self.actor(tensor).squeeze().item())
        if self.exploration_noise_std > 0.0:
            ratio += random.gauss(0.0, self.exploration_noise_std)
        ratio = environment.clip_force_ratio(float(np.clip(ratio, -1.0, 1.0)))
        self.last_action_ratio = ratio
        force_n = self.force_from_ratio(environment, ratio)
        return environment.action_for_force_n(force_n), force_n

    def remember(self, state, action, reward, next_state, done) -> None:
        action_ratio = self.last_action_ratio if action is None else float(action)
        self.replay.push(state, action_ratio, reward, next_state, done)

    def _soft_update(self, target: nn.Module, source: nn.Module) -> None:
        with torch.no_grad():
            for target_param, source_param in zip(target.parameters(), source.parameters()):
                target_param.data.mul_(1.0 - self.tau)
                target_param.data.add_(self.tau * source_param.data)

    def behavior_clone_update(
        self,
        states: Sequence[np.ndarray],
        action_ratios: Sequence[float],
    ) -> float:
        state = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=self.device)
        action = torch.as_tensor(
            np.asarray(action_ratios),
            dtype=torch.float32,
            device=self.device,
        ).view(-1, 1)
        prediction = self.actor(state)
        loss = F.mse_loss(prediction, action)
        self.actor_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
        self.actor_optimizer.step()
        self.last_actor_loss = float(loss.item())
        return self.last_actor_loss

    def update(self) -> Optional[float]:
        if len(self.replay) < self.batch_size:
            return None
        state, action, reward, next_state, done = self.replay.sample(self.batch_size)
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)

        with torch.no_grad():
            next_action = self.target_actor(next_state)
            target_q = reward + self.gamma * (1.0 - done) * self.target_critic(
                next_state,
                next_action,
            )
        current_q = self.critic(state, action)
        critic_loss = F.smooth_l1_loss(current_q, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 5.0)
        self.critic_optimizer.step()

        actor_action = self.actor(state)
        actor_loss = -self.critic(state, actor_action).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
        self.actor_optimizer.step()

        self._soft_update(self.target_actor, self.actor)
        self._soft_update(self.target_critic, self.critic)
        self.last_loss = float(critic_loss.item())
        self.last_actor_loss = float(actor_loss.item())
        return self.last_loss


class SACAgent:
    """Soft Actor-Critic lower agent using the same force-ratio action."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 1,
        actor_learning_rate: float = 3e-4,
        critic_learning_rate: float = 8e-4,
        alpha_learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        replay_capacity: int = 100000,
        batch_size: int = 128,
        target_entropy: float = -1.0,
        device_name: Optional[str] = None,
    ) -> None:
        if action_dim != 1:
            raise ValueError("SACAgent expects one continuous force-ratio action")
        if device_name is None or device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.batch_size = int(batch_size)
        self.target_entropy = float(target_entropy)
        self.actor = SACActor(state_dim).to(self.device)
        self.critic1 = SACCritic(state_dim).to(self.device)
        self.critic2 = SACCritic(state_dim).to(self.device)
        self.target_critic1 = SACCritic(state_dim).to(self.device)
        self.target_critic2 = SACCritic(state_dim).to(self.device)
        _copy_target_if_present(self)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_learning_rate)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_learning_rate)
        self.replay = ContinuousReplayBuffer(replay_capacity)
        self.training_mode = False
        self.exploration_noise_std = 0.0
        self.last_action_ratio = 0.0
        self.last_loss: Optional[float] = None
        self.last_actor_loss: Optional[float] = None
        self.last_alpha_loss: Optional[float] = None
        self.policy = self.actor

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @property
    def alpha_value(self) -> float:
        return float(self.alpha.detach().cpu().item())

    def set_training_mode(self, training: bool) -> None:
        self.training_mode = bool(training)

    def set_exploration_noise(self, std: float) -> None:
        self.exploration_noise_std = max(0.0, float(std))

    def observation_from_environment(
        self,
        environment: LowerSectionEnvironment,
    ) -> np.ndarray:
        return environment.compact_observation()

    @staticmethod
    def force_from_ratio(environment: LowerSectionEnvironment, ratio: float) -> float:
        return force_from_ratio(environment, ratio)

    def select_action_and_force(
        self,
        environment: LowerSectionEnvironment,
    ) -> Tuple[int, float]:
        state = self.observation_from_environment(environment)
        with torch.no_grad():
            tensor = torch.as_tensor(
                state,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            if self.training_mode:
                ratio_tensor, _ = self.actor.sample(tensor)
            else:
                ratio_tensor = self.actor.deterministic_action(tensor)
            ratio = float(ratio_tensor.squeeze().item())
        ratio = environment.clip_force_ratio(float(np.clip(ratio, -1.0, 1.0)))
        self.last_action_ratio = ratio
        force_n = self.force_from_ratio(environment, ratio)
        return environment.action_for_force_n(force_n), force_n

    def remember(self, state, action, reward, next_state, done) -> None:
        action_ratio = self.last_action_ratio if action is None else float(action)
        self.replay.push(state, action_ratio, reward, next_state, done)

    def _soft_update(self, target: nn.Module, source: nn.Module) -> None:
        with torch.no_grad():
            for target_param, source_param in zip(target.parameters(), source.parameters()):
                target_param.data.mul_(1.0 - self.tau)
                target_param.data.add_(self.tau * source_param.data)

    def behavior_clone_update(
        self,
        states: Sequence[np.ndarray],
        action_ratios: Sequence[float],
    ) -> float:
        state = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=self.device)
        action = torch.as_tensor(
            np.asarray(action_ratios),
            dtype=torch.float32,
            device=self.device,
        ).view(-1, 1)
        prediction = self.actor.deterministic_action(state)
        loss = F.mse_loss(prediction, action)
        self.actor_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
        self.actor_optimizer.step()
        self.last_actor_loss = float(loss.item())
        return self.last_actor_loss

    def update(self) -> Optional[float]:
        if len(self.replay) < self.batch_size:
            return None
        state, action, reward, next_state, done = self.replay.sample(self.batch_size)
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)

        with torch.no_grad():
            next_action, next_log_probability = self.actor.sample(next_state)
            next_q = torch.min(
                self.target_critic1(next_state, next_action),
                self.target_critic2(next_state, next_action),
            )
            target_q = reward + self.gamma * (1.0 - done) * (
                next_q - self.alpha.detach() * next_log_probability
            )

        current_q1 = self.critic1(state, action)
        current_q2 = self.critic2(state, action)
        critic1_loss = F.smooth_l1_loss(current_q1, target_q)
        critic2_loss = F.smooth_l1_loss(current_q2, target_q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        nn.utils.clip_grad_norm_(self.critic1.parameters(), 5.0)
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        nn.utils.clip_grad_norm_(self.critic2.parameters(), 5.0)
        self.critic2_optimizer.step()

        sampled_action, log_probability = self.actor.sample(state)
        sampled_q = torch.min(
            self.critic1(state, sampled_action),
            self.critic2(state, sampled_action),
        )
        actor_loss = (self.alpha.detach() * log_probability - sampled_q).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
        self.actor_optimizer.step()

        alpha_loss = -(
            self.log_alpha * (log_probability + self.target_entropy).detach()
        ).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self._soft_update(self.target_critic1, self.critic1)
        self._soft_update(self.target_critic2, self.critic2)
        self.last_loss = float((critic1_loss + critic2_loss).item() * 0.5)
        self.last_actor_loss = float(actor_loss.item())
        self.last_alpha_loss = float(alpha_loss.item())
        return self.last_loss


def _sample_training_mass_t(
    mass_min_t: float,
    mass_max_t: float,
    anchor_mass_probability: float,
    boundary_mass_probability: float,
) -> float:
    anchor_masses_t = [config.mass_kg / 1000.0 for config in TRAIN_CONFIGS]
    boundary_probability = max(0.0, min(1.0, float(boundary_mass_probability)))
    anchor_probability = max(0.0, min(1.0, float(anchor_mass_probability)))
    draw = random.random()
    if draw < boundary_probability:
        return random.choice([mass_min_t, mass_max_t])
    if draw < boundary_probability + anchor_probability:
        anchors_in_range = [
            mass for mass in anchor_masses_t if mass_min_t <= mass <= mass_max_t
        ]
        return random.choice(anchors_in_range or anchor_masses_t)
    return random.uniform(mass_min_t, mass_max_t)


def _training_environment(
    environment: LowerSectionEnvironment,
    train_name: str,
    random_mass_training: bool,
    mass_min_t: float,
    mass_max_t: float,
    anchor_mass_probability: float,
    boundary_mass_probability: float,
) -> Tuple[LowerSectionEnvironment, float]:
    sampled_mass_t = environment.dynamics.config.mass_kg / 1000.0
    if not random_mass_training:
        return environment, sampled_mass_t
    sampled_mass_t = _sample_training_mass_t(
        mass_min_t,
        mass_max_t,
        anchor_mass_probability,
        boundary_mass_probability,
    )
    sampled_config = create_train_config_from_mass(
        name=f"{train_name} sampled {sampled_mass_t:.1f}t",
        total_mass_t=sampled_mass_t,
        max_braking_force_kn=environment.dynamics.config.max_braking_force_n / 1000.0,
    )
    return (
        LowerSectionEnvironment(
            environment.section,
            TrainDynamics(sampled_config),
            speed_limit_penalty_ratio=environment.speed_limit_penalty_ratio,
        ),
        sampled_mass_t,
    )


def pretrain_continuous_actor_with_heuristic(
    environments: Sequence[LowerSectionEnvironment],
    controller: Any,
    initial_timetable: np.ndarray,
    episodes: int,
    target_span_s: int,
    train_name: str,
    algorithm_name: str,
    random_mass_training: bool = False,
    mass_min_t: float = 235.0,
    mass_max_t: float = 275.0,
    anchor_mass_probability: float = 0.15,
    boundary_mass_probability: float = 0.05,
    log_interval: int = 10,
    bc_buffer_capacity: int = 50000,
    bc_updates_per_episode: int = 2,
    plot_interval: int = 0,
    output_dir: Optional[Path] = None,
    environment_tags: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    if episodes <= 0:
        return []
    print(
        f"Pretraining lower {algorithm_name.upper()} actor with heuristic demonstrations "
        f"for {train_name}: {episodes} episodes"
    )
    history: List[Dict[str, Any]] = []
    bc_buffer: Deque[Tuple[np.ndarray, float]] = deque(maxlen=bc_buffer_capacity)
    interval_start_time = time.perf_counter()
    for episode in range(episodes):
        idx = random.randrange(len(environments))
        tag = environment_tags[idx] if environment_tags is not None else {}
        environment, sampled_mass_t = _training_environment(
            environments[idx],
            train_name,
            random_mass_training,
            float(mass_min_t),
            float(mass_max_t),
            anchor_mass_probability,
            boundary_mass_probability,
        )
        target = float(initial_timetable[idx] + random.randint(-target_span_s, target_span_s))
        environment.reset(target)
        episode_loss: Optional[float] = None
        while not environment.done:
            state = controller.observation_from_environment(environment)
            action = environment.mask_invalid_discrete_action(
                environment.heuristic_action(aggressiveness=1.0)
            )
            _, reward, done, step_info = environment.step(action, None)
            next_state = controller.observation_from_environment(environment)
            action_ratio = float(step_info.get("force_ratio", 0.0))
            bc_buffer.append((state, action_ratio))
            controller.remember(state, action_ratio, reward, next_state, done)
            controller.update()
        batch_size = min(controller.batch_size, len(bc_buffer))
        if batch_size > 0:
            for _ in range(max(1, int(bc_updates_per_episode))):
                batch = random.sample(list(bc_buffer), batch_size)
                states, actions = zip(*batch)
                episode_loss = controller.behavior_clone_update(states, actions)
        result = environment.result()
        if plot_interval > 0 and (episode + 1) % plot_interval == 0:
            plot_lower_training_speed_snapshot(
                result,
                output_dir,
                f"{algorithm_name}_bc",
                train_name,
                episode + 1,
            )
        section_length_km = max(environment.section.length_m / 1000.0, EPS)
        history.append(
            {
                "episode": episode + 1,
                "phase": "behavior_clone_pretrain",
                "train_name": train_name,
                "lower_algorithm": f"{algorithm_name}_bc",
                "source_train_id": tag.get("train_id", ""),
                "source_train_name": tag.get("train_name", ""),
                "direction": tag.get("direction", ""),
                "speed_limit_penalty_ratio": environment.speed_limit_penalty_ratio,
                "section_name": result.section_name,
                "target_time_s": target,
                "actual_time_s": result.actual_time_s,
                "timetable_error_s": result.timetable_error_s,
                "traction_kwh": result.traction_kwh,
                "traction_kwh_per_km": result.traction_kwh / section_length_km,
                "episode_reward": result.episode_reward,
                "episode_reward_per_km": result.episode_reward / section_length_km,
                "bc_loss": episode_loss,
                "sampled_total_mass_t": sampled_mass_t,
                "feasible": result.feasible,
                "expert_interventions": result.expert_interventions,
            }
        )
        should_log = episode == 0 or (
            log_interval > 0 and (episode + 1) % log_interval == 0
        )
        if should_log:
            interval_rows = history[-max(1, min(log_interval, len(history))):]
            elapsed_s = time.perf_counter() - interval_start_time
            interval_start_time = time.perf_counter()
            mean_loss = float(
                np.nanmean(
                    [
                        row["bc_loss"]
                        for row in interval_rows
                        if row["bc_loss"] is not None
                    ]
                    or [np.nan]
                )
            )
            feasible_count = sum(1 for row in interval_rows if row["feasible"])
            print(
                f"  {algorithm_name.upper()} BC {episode + 1:5d}/{episodes}: "
                f"last {len(interval_rows):2d} eps {elapsed_s:7.2f}s "
                f"section={result.section_name:9s} target={target:6.1f}s "
                f"mass={sampled_mass_t:5.1f}t "
                f"bc_loss={mean_loss:8.5f} "
                f"actual={result.actual_time_s:7.1f}s "
                f"error={result.timetable_error_s:+6.1f}s "
                f"feasible={feasible_count}/{len(interval_rows)}"
            )
    _copy_target_if_present(controller)
    return history


def _train_lower_continuous_agents(
    environments: Sequence[LowerSectionEnvironment],
    controller: Any,
    initial_timetable: np.ndarray,
    episodes: int,
    target_span_s: int,
    train_name: str,
    algorithm_name: str,
    random_mass_training: bool = False,
    mass_min_t: float = 235.0,
    mass_max_t: float = 275.0,
    anchor_mass_probability: float = 0.15,
    boundary_mass_probability: float = 0.05,
    log_interval: int = 10,
    plot_interval: int = 0,
    output_dir: Optional[Path] = None,
    environment_tags: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    print(
        f"Fine-tuning lower {algorithm_name.upper()} continuous-force policy "
        f"for {train_name}: {episodes} episodes"
    )
    history: List[Dict[str, Any]] = []
    if hasattr(controller, "set_training_mode"):
        controller.set_training_mode(True)
    interval_start_time = time.perf_counter()
    for episode in range(episodes):
        idx = random.randrange(len(environments))
        tag = environment_tags[idx] if environment_tags is not None else {}
        environment, sampled_mass_t = _training_environment(
            environments[idx],
            train_name,
            random_mass_training,
            float(mass_min_t),
            float(mass_max_t),
            anchor_mass_probability,
            boundary_mass_probability,
        )
        target = float(initial_timetable[idx] + random.randint(-target_span_s, target_span_s))
        if algorithm_name == "ddpg":
            noise = 0.30 + (0.03 - 0.30) * min(1.0, episode / max(episodes - 1, 1))
        else:
            noise = 0.0
        controller.set_exploration_noise(noise)
        result = environment.rollout(
            controller,
            target,
            training=True,
            heuristic_probability=0.0,
        )
        if plot_interval > 0 and (episode + 1) % plot_interval == 0:
            plot_lower_training_speed_snapshot(
                result,
                output_dir,
                algorithm_name,
                train_name,
                episode + 1,
            )
        sampled_config = environment.dynamics.config
        section_length_km = max(environment.section.length_m / 1000.0, EPS)
        entropy_alpha = (
            controller.alpha_value
            if hasattr(controller, "alpha_value")
            else None
        )
        history.append(
            {
                "episode": episode + 1,
                "phase": "rl_finetune",
                "train_name": train_name,
                "lower_algorithm": algorithm_name,
                "source_train_id": tag.get("train_id", ""),
                "source_train_name": tag.get("train_name", ""),
                "direction": tag.get("direction", ""),
                "random_mass_training": random_mass_training,
                "speed_limit_penalty_ratio": environment.speed_limit_penalty_ratio,
                "sampled_total_mass_t": sampled_config.mass_kg / 1000.0,
                "sampled_motor_car_mass_t": sampled_config.effective_motor_car_mass_kg
                / 1000.0,
                "sampled_trailer_car_mass_t": sampled_config.effective_trailer_car_mass_kg
                / 1000.0,
                "sampled_max_traction_force_kn": sampled_config.max_traction_force_n
                / 1000.0,
                "sampled_max_traction_power_kw": sampled_config.max_traction_power_w
                / 1000.0,
                "section_name": result.section_name,
                "target_time_s": target,
                "actual_time_s": result.actual_time_s,
                "timetable_error_s": result.timetable_error_s,
                "net_energy_kwh": result.net_energy_kwh,
                "net_energy_kwh_per_km": result.net_energy_kwh / section_length_km,
                "traction_kwh": result.traction_kwh,
                "traction_kwh_per_km": result.traction_kwh / section_length_km,
                "feasible": result.feasible,
                "regime_switches": result.regime_switches,
                "traction_coast_transitions": result.traction_coast_transitions,
                "short_dwell_transitions": result.short_dwell_transitions,
                "force_jump_index": result.force_jump_index,
                "smoothness_penalty": result.smoothness_penalty,
                "jerk_integral_mps3_s": result.jerk_integral_mps3_s,
                "jerk_integral_mps3_s_per_km": result.jerk_integral_mps3_s / section_length_km,
                "comfort_penalty": result.comfort_penalty,
                "traction_reward_component": result.traction_reward_component,
                "punctuality_reward_component": result.punctuality_reward_component,
                "comfort_reward_component": result.comfort_reward_component,
                "max_speed_limit_violation_kmh": result.max_speed_limit_violation_kmh,
                "speed_regulation_steps": result.speed_regulation_steps,
                "expert_interventions": result.expert_interventions,
                "episode_reward": result.episode_reward,
                "episode_reward_per_km": result.episode_reward / section_length_km,
                "epsilon": noise,
                "entropy_alpha": entropy_alpha,
                "loss": controller.last_loss,
                "actor_loss": controller.last_actor_loss,
                "alpha_loss": getattr(controller, "last_alpha_loss", None),
            }
        )
        should_log = episode == 0 or (
            log_interval > 0 and (episode + 1) % log_interval == 0
        )
        if should_log:
            interval_rows = history[-max(1, min(log_interval, len(history))):]
            interval_elapsed_s = time.perf_counter() - interval_start_time
            interval_start_time = time.perf_counter()
            mean_reward_per_km = float(
                np.mean([row["episode_reward_per_km"] for row in interval_rows])
            )
            mean_actual_time_s = float(
                np.mean([row["actual_time_s"] for row in interval_rows])
            )
            mean_abs_error_s = float(
                np.mean([abs(row["timetable_error_s"]) for row in interval_rows])
            )
            feasible_count = sum(1 for row in interval_rows if row["feasible"])
            mean_expert_interventions = float(
                np.mean([row["expert_interventions"] for row in interval_rows])
            )
            alpha_text = (
                f" alpha={entropy_alpha:5.3f}"
                if entropy_alpha is not None
                else f" noise={noise:4.2f}"
            )
            print(
                f"  {algorithm_name.upper()} episode {episode + 1:5d}/{episodes}: "
                f"last {len(interval_rows):2d} eps {interval_elapsed_s:7.2f}s "
                f"section={result.section_name:9s} target={target:6.1f}s "
                f"mass={sampled_mass_t:5.1f}t"
                f"{alpha_text} "
                f"actual={result.actual_time_s:7.1f}s traction={result.traction_kwh:7.2f}kWh "
                f"error={result.timetable_error_s:+6.1f}s "
                f"jerk={result.jerk_integral_mps3_s:7.2f} "
                f"reward/km={result.episode_reward / section_length_km:7.2f} "
                f"| avg_reward/km={mean_reward_per_km:7.2f} "
                f"avg_actual={mean_actual_time_s:7.1f}s "
                f"avg_abs_err={mean_abs_error_s:6.1f}s "
                f"feasible={feasible_count}/{len(interval_rows)} "
                f"expert={mean_expert_interventions:5.1f}/ep"
            )
    controller.set_exploration_noise(0.0)
    if hasattr(controller, "set_training_mode"):
        controller.set_training_mode(False)
    return history


def train_lower_ddpg_agents(
    environments: Sequence[LowerSectionEnvironment],
    controller: DDPGAgent,
    initial_timetable: np.ndarray,
    episodes: int,
    target_span_s: int,
    train_name: str = "Train 1",
    random_mass_training: bool = False,
    mass_min_t: float = 235.0,
    mass_max_t: float = 275.0,
    anchor_mass_probability: float = 0.15,
    boundary_mass_probability: float = 0.05,
    log_interval: int = 10,
    pretrain_episodes: int = 0,
    plot_interval: int = 0,
    output_dir: Optional[Path] = None,
    environment_tags: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    pretrain_history = pretrain_continuous_actor_with_heuristic(
        environments,
        controller,
        initial_timetable,
        episodes=int(pretrain_episodes),
        target_span_s=target_span_s,
        train_name=train_name,
        algorithm_name="ddpg",
        random_mass_training=random_mass_training,
        mass_min_t=mass_min_t,
        mass_max_t=mass_max_t,
        anchor_mass_probability=anchor_mass_probability,
        boundary_mass_probability=boundary_mass_probability,
        log_interval=log_interval,
        plot_interval=plot_interval,
        output_dir=output_dir,
        environment_tags=environment_tags,
    )
    finetune_history = _train_lower_continuous_agents(
        environments,
        controller,
        initial_timetable,
        episodes=episodes,
        target_span_s=target_span_s,
        train_name=train_name,
        algorithm_name="ddpg",
        random_mass_training=random_mass_training,
        mass_min_t=mass_min_t,
        mass_max_t=mass_max_t,
        anchor_mass_probability=anchor_mass_probability,
        boundary_mass_probability=boundary_mass_probability,
        log_interval=log_interval,
        plot_interval=plot_interval,
        output_dir=output_dir,
        environment_tags=environment_tags,
    )
    return pretrain_history + finetune_history


def train_lower_sac_agents(
    environments: Sequence[LowerSectionEnvironment],
    controller: SACAgent,
    initial_timetable: np.ndarray,
    episodes: int,
    target_span_s: int,
    train_name: str = "Train 1",
    random_mass_training: bool = False,
    mass_min_t: float = 235.0,
    mass_max_t: float = 275.0,
    anchor_mass_probability: float = 0.15,
    boundary_mass_probability: float = 0.05,
    log_interval: int = 10,
    pretrain_episodes: int = 0,
    plot_interval: int = 0,
    output_dir: Optional[Path] = None,
    environment_tags: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    pretrain_history = pretrain_continuous_actor_with_heuristic(
        environments,
        controller,
        initial_timetable,
        episodes=int(pretrain_episodes),
        target_span_s=target_span_s,
        train_name=train_name,
        algorithm_name="sac",
        random_mass_training=random_mass_training,
        mass_min_t=mass_min_t,
        mass_max_t=mass_max_t,
        anchor_mass_probability=anchor_mass_probability,
        boundary_mass_probability=boundary_mass_probability,
        log_interval=log_interval,
        plot_interval=plot_interval,
        output_dir=output_dir,
        environment_tags=environment_tags,
    )
    finetune_history = _train_lower_continuous_agents(
        environments,
        controller,
        initial_timetable,
        episodes=episodes,
        target_span_s=target_span_s,
        train_name=train_name,
        algorithm_name="sac",
        random_mass_training=random_mass_training,
        mass_min_t=mass_min_t,
        mass_max_t=mass_max_t,
        anchor_mass_probability=anchor_mass_probability,
        boundary_mass_probability=boundary_mass_probability,
        log_interval=log_interval,
        plot_interval=plot_interval,
        output_dir=output_dir,
        environment_tags=environment_tags,
    )
    return pretrain_history + finetune_history
