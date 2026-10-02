
import math
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# Matplotlib font fallback for Chinese labels when available.
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class VisualTrainEnvironment:
    def __init__(self, section_split_k=100):
        self.mass = 260000.0
        self.max_traction = 300000.0
        self.max_braking = -250000.0
        self.gravity = 9.81

        self.eff_traction = 0.9
        self.eff_braking = 0.6

        self.energy_unit = 0.25 * 3.6e6
        self.section_split_k = max(1, int(section_split_k))

        self.res_a = 0.6566
        self.res_b = 45.746302
        self.res_c = 3511.5066

        self.sections = [
            {"len": 302.189, "grad": -2.0, "v_limit": 10.556, "type": "MA"},
            {"len": 81.811, "grad": -2.0, "v_limit": 25.278, "type": "DIST"},
            {"len": 320.0, "grad": -29.0, "v_limit": 25.278, "type": "DIST"},
            {"len": 426.0, "grad": -8.0, "v_limit": 25.278, "type": "DIST"},
            {"len": 109.512, "grad": -3.635, "v_limit": 25.278, "type": "DIST"},
            {"len": 314.488, "grad": -3.635, "v_limit": 34.722, "type": "DIST"},
            {"len": 390.0, "grad": -3.659, "v_limit": 34.722, "type": "DIST"},
            {"len": 410.0, "grad": 12.0, "v_limit": 34.722, "type": "DIST"},
            {"len": 510.0, "grad": 14.0, "v_limit": 34.722, "type": "DIST"},
            {"len": 260.0, "grad": 25.0, "v_limit": 34.722, "type": "DIST"},
            {"len": 209.0, "grad": 0.0, "v_limit": 34.722, "type": "MB"},
        ]
        self.num_sections = len(self.sections)
        self.traction_section_types = {"MA", "DIST"}
        self.control_section_count = sum(
            1 for sec in self.sections if sec["type"] in self.traction_section_types
        )
        self.start_pos = 346.0
        self.end_pos = self.start_pos + sum(sec["len"] for sec in self.sections)

        self.history_log = []
        self.current_pos = self.start_pos
        self.current_time = 0.0
        self.total_regen_energy = 0.0
        self.trip_stalled = False
        self.stall_time_penalty = 1e6
        self.stall_distance_penalty = 100.0

    def calc_resistance(self, v):
        return self.res_a * (v ** 2 * 12.96) + self.res_b * v * 3.6 + self.res_c

    def reset_log(self):
        self.history_log = []
        self.current_pos = self.start_pos
        self.current_time = 0.0
        self.total_regen_energy = 0.0
        self.trip_stalled = False

    def simulate_segment(self, v_in, section_data, allocated_energy):
        v = max(v_in, 0.0)
        energy_used = 0.0

        sec_length = section_data["len"]
        gradient = section_data["grad"]
        v_limit = section_data["v_limit"]
        regime = section_data["type"]

        grad_resistance = self.mass * self.gravity * (gradient / 1000.0)
        step_distance = sec_length / self.section_split_k

        for _ in range(self.section_split_k):
            run_resistance = self.calc_resistance(v)
            total_resistance = run_resistance + grad_resistance

            if regime == 'MB':
                force = self.max_braking
            else:
                if v < v_limit:
                    if regime in self.traction_section_types:
                        if energy_used < allocated_energy:
                            max_step_energy = (self.max_traction * step_distance) / self.eff_traction
                            remaining_energy = allocated_energy - energy_used
                            if remaining_energy >= max_step_energy:
                                force = self.max_traction
                                energy_used += max_step_energy
                            else:
                                force = (remaining_energy * self.eff_traction) / step_distance
                                energy_used = allocated_energy
                        else:
                            force = 0.0
                    else:
                        force = 0.0
                else:
                    force = 0.0

            if force == 0.0 and v >= v_limit and total_resistance < 0:
                force = total_resistance

            self.history_log.append((self.current_time, self.current_pos, v, force))

            if v <= 1e-9 and force <= total_resistance:
                self.trip_stalled = True
                return 0.0

            if force < 0:
                self.total_regen_energy += abs(force) * step_distance * self.eff_braking

            accel = (force - total_resistance) / self.mass
            v_new_sq = v ** 2 + 2 * accel * step_distance
            v_new = math.sqrt(max(0.0, v_new_sq))

            dt = 2 * step_distance / max(v + v_new, 1e-6)
            self.current_time += dt
            self.current_pos += step_distance
            v = v_new

        return v

    def run_and_log_trip(self, state):
        if len(state) != self.control_section_count:
            raise ValueError(
                f"state length {len(state)} does not match traction section count {self.control_section_count}"
            )

        self.reset_log()
        v = 0.0
        section_idx = 0

        for sec in self.sections:
            if sec["type"] in self.traction_section_types:
                allocated_joules = state[section_idx] * self.energy_unit
                v = self.simulate_segment(v, sec, allocated_joules)
                section_idx += 1
            else:
                v = self.simulate_segment(v, sec, 0.0)

            if self.trip_stalled:
                self.history_log.append((self.current_time, self.current_pos, 0.0, 0.0))
                remaining_distance = self.end_pos - self.current_pos
                penalty_time = self.stall_time_penalty + remaining_distance * self.stall_distance_penalty
                return penalty_time, self.history_log

        self.history_log.append((self.current_time, self.current_pos, 0.0, 0.0))
        return self.current_time, self.history_log


class ReservoirReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.c_s = 0

    def push(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.c_s += 1
            p = random.randint(0, self.c_s)
            if p < self.capacity:
                self.buffer[p] = transition

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions).unsqueeze(1),
            torch.FloatTensor(rewards).unsqueeze(1),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)


class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class SACAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3):
        self.gamma = 0.99
        self.tau = 0.01
        self.alpha = 0.1

        self.actor = ActorNetwork(state_dim, action_dim)
        self.q1 = QNetwork(state_dim, action_dim)
        self.q2 = QNetwork(state_dim, action_dim)
        self.target_q1 = QNetwork(state_dim, action_dim)
        self.target_q2 = QNetwork(state_dim, action_dim)

        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.q1_opt = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_opt = optim.Adam(self.q2.parameters(), lr=lr)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            probs = self.actor(state)
        dist = Categorical(probs)
        return dist.sample().item()

    def select_greedy_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            probs = self.actor(state)
        return torch.argmax(probs, dim=-1).item()

    def update(self, buffer, batch_size):
        if len(buffer) < batch_size:
            return

        s, a, r, ns, d = buffer.sample(batch_size)

        with torch.no_grad():
            next_probs = self.actor(ns)
            next_log_probs = torch.log(next_probs + 1e-8)
            next_q = torch.min(self.target_q1(ns), self.target_q2(ns))
            target_v = (next_probs * (next_q - self.alpha * next_log_probs)).sum(dim=1, keepdim=True)
            target_q_val = r + (1 - d) * self.gamma * target_v

        curr_q1 = self.q1(s).gather(1, a)
        curr_q2 = self.q2(s).gather(1, a)

        self.q1_opt.zero_grad()
        F.mse_loss(curr_q1, target_q_val).backward()
        self.q1_opt.step()

        self.q2_opt.zero_grad()
        F.mse_loss(curr_q2, target_q_val).backward()
        self.q2_opt.step()

        probs = self.actor(s)
        log_probs = torch.log(probs + 1e-8)
        min_q = torch.min(self.q1(s), self.q2(s))
        actor_loss = (probs * (self.alpha * log_probs - min_q)).sum(dim=1).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        for p, tp in zip(self.q1.parameters(), self.target_q1.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        for p, tp in zip(self.q2.parameters(), self.target_q2.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)


def normalize_state(state, max_energy_units):
    return np.asarray(state, dtype=np.float32) / max(max_energy_units, 1)


def decode_transfer_action(action, section_count):
    return divmod(action, section_count)


def apply_transfer_action(state, action, section_count):
    from_idx, to_idx = decode_transfer_action(action, section_count)
    next_state = np.copy(state)
    if from_idx == to_idx or next_state[from_idx] <= 0:
        return next_state, False

    next_state[from_idx] -= 1
    next_state[to_idx] += 1
    return next_state, True


def build_feasible_initial_state(env, scheduled_trip_time, total_energy_units, max_attempts=2000):
    uniform_units = total_energy_units // env.control_section_count
    remainder = total_energy_units % env.control_section_count
    state = np.full(env.control_section_count, uniform_units, dtype=float)
    if remainder > 0:
        state[:remainder] += 1

    best_state = np.copy(state)
    best_time, _ = env.run_and_log_trip(best_state)
    has_feasible_state = not env.trip_stalled
    best_error = abs(best_time - scheduled_trip_time)

    for _ in range(max_attempts):
        candidate = np.zeros(env.control_section_count)
        for _ in range(total_energy_units):
            candidate[random.randrange(env.control_section_count)] += 1

        candidate_time, _ = env.run_and_log_trip(candidate)
        if env.trip_stalled:
            continue

        candidate_error = abs(candidate_time - scheduled_trip_time)
        if not has_feasible_state or candidate_error < best_error:
            best_state = candidate
            best_time = candidate_time
            best_error = candidate_error
            has_feasible_state = True

        if candidate_error <= 5.0:
            break

    if not has_feasible_state:
        raise RuntimeError("Unable to build a feasible initial energy allocation.")

    return best_state, best_time


def compute_schedule_reward(
    current_time,
    new_time,
    scheduled_trip_time,
    energy_units,
    max_energy_units,
    terminal=False,
):
    stall_penalty_error = scheduled_trip_time * 2.0

    def schedule_error(trip_time):
        if trip_time >= 1e5:
            return stall_penalty_error
        return abs(trip_time - scheduled_trip_time)

    current_error = schedule_error(current_time)
    new_error = schedule_error(new_time)

    progress_reward = current_error - new_error
    energy_penalty = 0.02
    reward = progress_reward - energy_penalty

    if terminal:
        normalized_energy = energy_units / max(max_energy_units, 1)
        terminal_reward = -new_error - normalized_energy
        if new_time >= 1e5:
            terminal_reward -= 5.0
        reward += terminal_reward

    return reward


def rollout_policy(
    env,
    agent,
    initial_state,
    scheduled_trip_time,
    total_energy_units,
    max_transfer_steps,
    target_tolerance,
    stochastic,
):
    state = np.copy(initial_state)
    current_time, history = env.run_and_log_trip(state)
    step_count = 0

    action_dim = env.control_section_count * env.control_section_count
    while step_count < max_transfer_steps:
        norm_state = normalize_state(state, total_energy_units)
        if stochastic:
            action = agent.select_action(norm_state)
        else:
            action = agent.select_greedy_action(norm_state)
        if action < 0 or action >= action_dim:
            break

        next_state, is_valid_action = apply_transfer_action(
            state, action, env.control_section_count
        )
        if not is_valid_action:
            break

        new_time, new_history = env.run_and_log_trip(next_state)
        current_error = abs(current_time - scheduled_trip_time)
        new_error = abs(new_time - scheduled_trip_time)

        if env.trip_stalled or new_error > current_error:
            break

        state = next_state
        current_time = new_time
        history = new_history
        step_count += 1

        if new_error <= target_tolerance:
            break

    return current_time, state, history, step_count


def evaluate_policy(
    env,
    agent,
    initial_state,
    scheduled_trip_time,
    total_energy_units,
    max_transfer_steps,
    target_tolerance,
    num_rollouts=5,
):
    rollout_results = [
        rollout_policy(
            env,
            agent,
            initial_state,
            scheduled_trip_time,
            total_energy_units,
            max_transfer_steps,
            target_tolerance,
            stochastic=True,
        )
        for _ in range(num_rollouts)
    ]

    mean_time = sum(result[0] for result in rollout_results) / len(rollout_results)
    best_result = min(rollout_results, key=lambda result: abs(result[0] - scheduled_trip_time))
    return mean_time, best_result[0], best_result[1], best_result[2], best_result[3]


def plot_convergence(ep_rewards, ep_times, ep_energy, scheduled_time):
    episodes = range(len(ep_rewards))
    plt.figure(figsize=(12, 10))

    plt.subplot(3, 1, 1)
    plt.plot(episodes, ep_rewards, 'b-', linewidth=1.5)
    plt.title('SAC Training Convergence')
    plt.ylabel('Episode Reward')
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(episodes, ep_times, 'g-', linewidth=1.5, label='Trip Time')
    plt.axhline(y=scheduled_time, color='r', linestyle='--', label='Target Time')
    plt.ylabel('Time (s)')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(episodes, ep_energy, 'm-', linewidth=1.5)
    plt.xlabel('Episode')
    plt.ylabel('Energy Units')
    plt.grid(True)

    plt.tight_layout()


def plot_results(history_log, state_distribution, scheduled_time, env):
    times = [log[0] for log in history_log]
    positions = [log[1] for log in history_log]
    velocities = [log[2] for log in history_log]
    forces = [log[3] / 1000.0 for log in history_log]

    total_time = times[-1]
    regen_kwh = env.total_regen_energy / 3.6e6

    _, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    ax1.plot(positions, velocities, 'b-', linewidth=2, label='Speed')
    current_pos = env.start_pos
    for sec in env.sections:
        ax1.axvline(x=current_pos, color='gray', linestyle=':', alpha=0.5)
        ax1.plot([current_pos, current_pos + sec["len"]], [sec["v_limit"], sec["v_limit"]], 'r--', alpha=0.8)
        current_pos += sec["len"]
    ax1.plot([], [], 'r--', label='Speed Limit')
    ax1.set_title(f'Best Control Curve (time: {total_time:.1f}s / target: {scheduled_time}s) | Regen: {regen_kwh:.2f} kWh')
    ax1.set_ylabel('Speed (m/s)')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(positions, forces, 'g-', linewidth=1.5, label='Force')
    ax2.axhline(0, color='black', linewidth=1)
    current_pos = env.start_pos
    for sec in env.sections:
        ax2.axvline(x=current_pos, color='gray', linestyle=':', alpha=0.5)
        current_pos += sec["len"]
    ax2.set_ylabel('Force (kN)')
    ax2.grid(True)

    current_pos = env.start_pos
    section_idx = 0
    max_height = max(state_distribution) if sum(state_distribution) > 0 else 1.0
    for sec in env.sections:
        sec_len = sec["len"]
        ax3.axvline(x=current_pos, color='gray', linestyle=':', alpha=0.5)
        if sec["type"] in env.traction_section_types:
            allocated_units = state_distribution[section_idx]
            ax3.bar(
                x=current_pos,
                height=allocated_units,
                width=sec_len,
                align='edge',
                color='orange',
                alpha=0.7,
                edgecolor='black',
                linewidth=1,
            )
            if allocated_units > 0:
                ax3.text(
                    current_pos + sec_len / 2.0,
                    allocated_units + max_height * 0.05,
                    f'{int(allocated_units)}',
                    ha='center',
                    va='bottom',
                    fontsize=12,
                    fontweight='bold',
                    color='black',
                )
            ax3.text(
                current_pos + sec_len / 2.0,
                -max_height * 0.15,
                f'{sec["grad"]} permille',
                ha='center',
                va='top',
                fontsize=10,
                color='brown',
            )
            section_idx += 1
        current_pos += sec_len

    ax3.axvline(x=current_pos, color='gray', linestyle=':', alpha=0.5)
    ax3.set_xlim([env.start_pos, current_pos])
    ax3.set_ylabel('Energy Units')
    ax3.set_xlabel('Position (m)')
    ax3.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    section_split_k = 100
    env = VisualTrainEnvironment(section_split_k=section_split_k)
    control_sections = env.control_section_count
    action_dim = control_sections * control_sections

    agent = SACAgent(state_dim=control_sections, action_dim=action_dim)
    buffer = ReservoirReplayBuffer(capacity=5000)

    scheduled_trip_time = 200.0
    max_episodes = 200
    total_energy_units = 120
    max_transfer_steps = 120
    target_tolerance = 2.0
    initial_state, initial_time = build_feasible_initial_state(
        env, scheduled_trip_time, total_energy_units
    )

    best_state = None
    best_time_error = float('inf')
    best_history = None

    ep_rewards = []
    ep_times = []
    ep_energy = []
    train_times = []

    print('SAC training started...')
    print(f'Initial feasible time: {initial_time:.1f}s | Total energy units: {int(np.sum(initial_state))}')
    for ep in range(max_episodes):
        state = np.copy(initial_state)
        current_time, history = env.run_and_log_trip(state)

        step_count = 0
        total_reward = 0.0
        exploration_rate = max(0.02, 0.12 * (1.0 - ep / max_episodes))

        while step_count < max_transfer_steps:
            if random.random() < exploration_rate:
                action = random.randrange(action_dim)
            else:
                action = agent.select_action(normalize_state(state, total_energy_units))
            next_state, is_valid_action = apply_transfer_action(state, action, control_sections)

            if not is_valid_action:
                reward = -5.0
                done = 1.0
                buffer.push(
                    normalize_state(state, total_energy_units),
                    action,
                    reward,
                    normalize_state(next_state, total_energy_units),
                    done,
                )
                agent.update(buffer, batch_size=32)
                total_reward += reward
                break

            new_time, new_history = env.run_and_log_trip(next_state)
            current_error = abs(current_time - scheduled_trip_time)
            new_error = abs(new_time - scheduled_trip_time)
            energy_units = int(np.sum(next_state))

            if env.trip_stalled or new_error > current_error:
                reward = compute_schedule_reward(
                    current_time,
                    new_time,
                    scheduled_trip_time,
                    energy_units,
                    total_energy_units,
                    terminal=True,
                ) - 5.0
                done = 1.0
                buffer.push(
                    normalize_state(state, total_energy_units),
                    action,
                    reward,
                    normalize_state(next_state, total_energy_units),
                    done,
                )
                agent.update(buffer, batch_size=32)
                total_reward += reward
                break
            elif new_error <= target_tolerance:
                reward = compute_schedule_reward(
                    current_time,
                    new_time,
                    scheduled_trip_time,
                    energy_units,
                    total_energy_units,
                    terminal=True,
                )
                done = 1.0
            elif step_count + 1 >= max_transfer_steps:
                reward = compute_schedule_reward(
                    current_time,
                    new_time,
                    scheduled_trip_time,
                    energy_units,
                    total_energy_units,
                    terminal=True,
                )
                done = 1.0
            else:
                reward = compute_schedule_reward(
                    current_time,
                    new_time,
                    scheduled_trip_time,
                    energy_units,
                    total_energy_units,
                )
                done = 0.0

            buffer.push(
                normalize_state(state, total_energy_units),
                action,
                reward,
                normalize_state(next_state, total_energy_units),
                done,
            )
            agent.update(buffer, batch_size=32)

            state = next_state
            current_time = new_time
            history = new_history
            total_reward += reward
            step_count += 1

        ep_rewards.append(total_reward)
        ep_times.append(current_time)
        ep_energy.append(np.sum(state))
        train_times.append(current_time)

        current_time_error = abs(current_time - scheduled_trip_time)
        if current_time_error < best_time_error:
            best_time_error = current_time_error
            best_state = np.copy(state)
            best_history = history.copy()

        time_diff = current_time - scheduled_trip_time
        print(
            f'Episode: {ep:03d} | Reward: {total_reward:8.3f} | Time: {current_time:9.1f}s '
            f'| TimeDiff: {time_diff:9.1f}s | EnergyUnits: {int(np.sum(state)):3d}'
        )

    print('Training finished. Plotting...')
    plot_convergence(ep_rewards, ep_times, ep_energy, scheduled_trip_time)

    if best_state is not None:
        print('Found a feasible policy.')
        env.run_and_log_trip(best_state)
        plot_results(best_history, best_state, scheduled_trip_time, env)
    else:
        print('No feasible policy found. Showing the last episode result.')
        env.run_and_log_trip(state)
        plot_results(history, state, scheduled_trip_time, env)


if __name__ == '__main__':
    main()
