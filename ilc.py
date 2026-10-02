
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
    def __init__(self, section_split_k=100, resistance_scale_profile=None):
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
        self.resistance_scale_profile = resistance_scale_profile
        self.global_resistance_scale = 1.0

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

    def get_resistance_scale(self, position):
        if self.resistance_scale_profile is None:
            return self.global_resistance_scale

        relative_pos = position - self.start_pos
        for item in self.resistance_scale_profile:
            if item["start"] <= relative_pos < item["end"]:
                return item["scale"]
        return 1.0

    def set_global_resistance_scale(self, scale):
        self.global_resistance_scale = float(scale)

    def calc_resistance(self, v, position):
        speed_resistance = self.res_a * (v ** 2 * 12.96) + self.res_b * v * 3.6
        base_resistance = self.res_c * self.get_resistance_scale(position)
        return speed_resistance + base_resistance

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
            run_resistance = self.calc_resistance(v, self.current_pos)
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


def decode_energy_action(action, section_count):
    transfer_action_count = section_count * section_count
    if action < transfer_action_count:
        from_idx, to_idx = divmod(action, section_count)
        return "transfer", from_idx, to_idx
    return "add", None, action - transfer_action_count


def apply_energy_action(state, action, section_count, max_total_units):
    action_type, from_idx, to_idx = decode_energy_action(action, section_count)
    next_state = np.copy(state)

    if action_type == "add":
        if to_idx < 0 or to_idx >= section_count or np.sum(next_state) >= max_total_units:
            return next_state, False
        next_state[to_idx] += 1
        return next_state, True

    if from_idx == to_idx or next_state[from_idx] <= 0:
        return next_state, False

    next_state[from_idx] -= 1
    next_state[to_idx] += 1
    return next_state, True


def sample_valid_energy_action(state, section_count, max_total_units, allow_add=True):
    valid_actions = []
    transfer_action_count = section_count * section_count

    for action in range(transfer_action_count):
        from_idx, to_idx = divmod(action, section_count)
        if from_idx != to_idx and state[from_idx] > 0:
            valid_actions.append(action)

    if allow_add and np.sum(state) < max_total_units:
        valid_actions.extend(range(transfer_action_count, transfer_action_count + section_count))

    if not valid_actions:
        return None

    return random.choice(valid_actions)


def build_feasible_initial_state(
    env,
    scheduled_trip_time,
    total_energy_units,
    max_attempts=2000,
    target_tolerance=0.5,
):
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

        if candidate_error <= target_tolerance:
            break

    if not has_feasible_state:
        raise RuntimeError("Unable to build a feasible initial energy allocation.")

    return best_state, best_time


def determine_total_energy_units(
    env,
    scheduled_trip_time,
    min_units,
    max_units,
    step_units=5,
    max_attempts_per_budget=300,
    target_tolerance=0,
):
    best_units = None
    best_state = None
    best_time = float('inf')
    best_error = float('inf')

    for total_units in range(min_units, max_units + 1, step_units):
        try:
            candidate_state, candidate_time = build_feasible_initial_state(
                env,
                scheduled_trip_time,
                total_units,
                max_attempts=max_attempts_per_budget,
                target_tolerance=target_tolerance,
            )
        except RuntimeError:
            continue

        candidate_error = abs(candidate_time - scheduled_trip_time)
        if candidate_error < best_error:
            best_units = total_units
            best_state = candidate_state
            best_time = candidate_time
            best_error = candidate_error

    if best_units is None:
        raise RuntimeError("Unable to determine a feasible total energy budget.")

    return best_units, best_state, best_time


def analyze_energy_budget_sensitivity(
    env,
    scheduled_trip_time,
    min_units=100,
    max_units=140,
    step_units=5,
    max_attempts_per_budget=500,
    target_tolerance=0.5,
):
    results = []
    print("\nEnergy budget sensitivity analysis")
    print("Units | Feasible | Time(s) | TimeDiff(s) | Allocation")
    print("-" * 78)

    for total_units in range(min_units, max_units + 1, step_units):
        try:
            state, trip_time = build_feasible_initial_state(
                env,
                scheduled_trip_time,
                total_units,
                max_attempts=max_attempts_per_budget,
                target_tolerance=target_tolerance,
            )
            feasible = not env.trip_stalled and trip_time < env.stall_time_penalty
            time_diff = trip_time - scheduled_trip_time
            allocation = [int(x) for x in state]
        except RuntimeError:
            state = None
            trip_time = float("inf")
            feasible = False
            time_diff = float("inf")
            allocation = []

        results.append(
            {
                "units": total_units,
                "feasible": feasible,
                "time": trip_time,
                "time_diff": time_diff,
                "allocation": allocation,
            }
        )

        time_text = f"{trip_time:7.1f}" if feasible else "   FAIL"
        diff_text = f"{time_diff:10.1f}" if feasible else "      FAIL"
        print(f"{total_units:5d} | {str(feasible):8s} | {time_text} | {diff_text} | {allocation}")

    return results


def compute_schedule_reward(
    current_time,
    new_time,
    scheduled_trip_time,
    energy_units,
    max_energy_units,
    terminal=False,
):
    def schedule_error(trip_time):
        if trip_time >= 1e5:
            return scheduled_trip_time * 2.0 + max(0.0, trip_time - 1e6) / 1000.0
        return abs(trip_time - scheduled_trip_time)

    current_error = schedule_error(current_time)
    new_error = schedule_error(new_time)

    progress_reward = current_error - new_error
    energy_penalty = 0.01 * (energy_units / max(max_energy_units, 1))
    reward = progress_reward - energy_penalty

    if terminal:
        normalized_energy = energy_units / max(max_energy_units, 1)
        terminal_reward = -(new_error ** 2) - 0.1 * normalized_energy
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


def get_episode_resistance_scale(episode, schedule):
    for start_ep, end_ep, scale in schedule:
        if start_ep <= episode < end_ep:
            return scale
    return schedule[-1][2]


def choose_best_energy_addition(env, state, scheduled_trip_time, max_total_units):
    if np.sum(state) >= max_total_units:
        return np.copy(state), -1, None

    best_state = None
    best_idx = -1
    best_time = None
    best_error = float('inf')

    for idx in range(env.control_section_count):
        candidate = np.copy(state)
        candidate[idx] += 1
        candidate_time, _ = env.run_and_log_trip(candidate)
        candidate_error = abs(candidate_time - scheduled_trip_time)
        if candidate_time >= env.stall_time_penalty:
            candidate_error += candidate_time / 1000.0
        if candidate_error < best_error:
            best_state = candidate
            best_idx = idx
            best_time = candidate_time
            best_error = candidate_error

    return best_state, best_idx, best_time


def choose_best_energy_removal(env, state, scheduled_trip_time):
    best_state = None
    best_idx = -1
    best_time = None
    best_error = float('inf')

    for idx in range(env.control_section_count):
        if state[idx] <= 0:
            continue

        candidate = np.copy(state)
        candidate[idx] -= 1
        candidate_time, _ = env.run_and_log_trip(candidate)
        if candidate_time >= env.stall_time_penalty:
            continue

        candidate_error = abs(candidate_time - scheduled_trip_time)
        if candidate_error < best_error:
            best_state = candidate
            best_idx = idx
            best_time = candidate_time
            best_error = candidate_error

    if best_state is None:
        return np.copy(state), -1, None

    return best_state, best_idx, best_time


def repair_unreachable_allocation(
    env,
    state,
    scheduled_trip_time,
    max_total_units,
    max_repairs=None,
    search_attempts_per_budget=120,
    search_step_units=5,
):
    original_state = np.copy(state)
    original_units = int(np.sum(original_state))
    best_state = None
    best_time = None
    best_history = None
    best_added_count = 0
    best_error = float('inf')

    for total_units in range(original_units, int(max_total_units) + 1, search_step_units):
        try:
            candidate_state, candidate_time = build_feasible_initial_state(
                env,
                scheduled_trip_time,
                total_units,
                max_attempts=search_attempts_per_budget,
                target_tolerance=0.5,
            )
        except RuntimeError:
            continue

        if candidate_time < env.stall_time_penalty:
            _, candidate_history = env.run_and_log_trip(candidate_state)
            candidate_error = abs(candidate_time - scheduled_trip_time)
            if candidate_error < best_error:
                best_state = candidate_state
                best_time = candidate_time
                best_history = candidate_history
                best_added_count = max(0, int(np.sum(candidate_state) - original_units))
                best_error = candidate_error

            if best_error <= 0.5:
                break

    if best_state is not None:
        return best_state, best_time, best_history, ["search"] * best_added_count

    repaired_state = original_state
    repaired_time, repaired_history = env.run_and_log_trip(repaired_state)
    added_sections = []
    if max_repairs is None:
        max_repairs = int(max(0, max_total_units - np.sum(repaired_state)))

    while repaired_time >= env.stall_time_penalty and len(added_sections) < max_repairs:
        repaired_state, added_idx, repaired_time = choose_best_energy_addition(
            env,
            repaired_state,
            scheduled_trip_time,
            max_total_units,
        )
        if added_idx < 0:
            break

        added_sections.append(added_idx)
        repaired_time, repaired_history = env.run_and_log_trip(repaired_state)

    return repaired_state, repaired_time, repaired_history, added_sections


def calibrate_phase_base(
    env,
    state,
    scheduled_trip_time,
    max_total_units,
    target_tolerance,
    max_attempts_per_budget=120,
):
    current_units = int(np.sum(state))
    best_state = None
    best_time = None
    best_error = float('inf')

    for total_units in range(current_units, int(max_total_units) + 1):
        try:
            candidate_state, candidate_time = build_feasible_initial_state(
                env,
                scheduled_trip_time,
                total_units,
                max_attempts=max_attempts_per_budget,
                target_tolerance=target_tolerance,
            )
        except RuntimeError:
            continue

        if candidate_time >= env.stall_time_penalty:
            continue

        candidate_error = abs(candidate_time - scheduled_trip_time)
        if candidate_error < best_error:
            best_state = candidate_state
            best_time = candidate_time
            best_error = candidate_error

        if candidate_error <= target_tolerance:
            return candidate_state, candidate_time, True

    if best_state is None:
        return np.copy(state), None, False

    return best_state, best_time, False


def plot_convergence(ep_rewards, ep_times, ep_energy, scheduled_time, base_energy=None):
    title_fontsize = 16
    label_fontsize = 14
    tick_fontsize = 12
    legend_fontsize = 12

    episodes = range(len(ep_rewards))
    plt.figure(figsize=(12, 10))

    plt.subplot(3, 1, 1)
    plt.plot(episodes, ep_rewards, 'b-', linewidth=1.5)
    plt.title('SAC Training Convergence', fontsize=title_fontsize)
    plt.ylabel('Episode Reward', fontsize=label_fontsize)
    plt.tick_params(axis='both', labelsize=tick_fontsize)
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(episodes, ep_times, 'g-', linewidth=1.5, label='Trip Time')
    plt.axhline(y=scheduled_time, color='r', linestyle='--', label='Target Time')
    plt.ylabel('Time (s)', fontsize=label_fontsize)
    plt.tick_params(axis='both', labelsize=tick_fontsize)
    plt.legend(fontsize=legend_fontsize)
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(episodes, ep_energy, 'm-', linewidth=1.5, label='Episode Final Units')
    if base_energy is not None:
        plt.plot(episodes, base_energy, 'c-', linewidth=1.5, label='Base Allocation Units')
    plt.xlabel('Episode', fontsize=label_fontsize)
    plt.ylabel('Energy Units', fontsize=label_fontsize)
    plt.tick_params(axis='both', labelsize=tick_fontsize)
    plt.legend(fontsize=legend_fontsize)
    plt.grid(True)

    plt.tight_layout()


def plot_results(history_log, state_distribution, scheduled_time, env, title_suffix=''):
    title_fontsize = 16
    label_fontsize = 14
    tick_fontsize = 12
    legend_fontsize = 12
    annotation_fontsize = 13
    grad_fontsize = 12

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
    title_prefix = f'Best Control Curve {title_suffix}'.strip()
    ax1.set_title(
        f'{title_prefix} (time: {total_time:.1f}s / target: {scheduled_time}s) | Regen: {regen_kwh:.2f} kWh',
        fontsize=title_fontsize,
    )
    ax1.set_ylabel('Speed (m/s)', fontsize=label_fontsize)
    ax1.tick_params(axis='both', labelsize=tick_fontsize)
    ax1.legend(fontsize=legend_fontsize)
    ax1.grid(True)

    ax2.plot(positions, forces, 'g-', linewidth=1.5, label='Force')
    ax2.axhline(0, color='black', linewidth=1)
    current_pos = env.start_pos
    for sec in env.sections:
        ax2.axvline(x=current_pos, color='gray', linestyle=':', alpha=0.5)
        current_pos += sec["len"]
    ax2.set_ylabel('Force (kN)', fontsize=label_fontsize)
    ax2.tick_params(axis='both', labelsize=tick_fontsize)
    ax2.legend(fontsize=legend_fontsize)
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
                    fontsize=annotation_fontsize,
                    fontweight='bold',
                    color='black',
                )
            ax3.text(
                current_pos + sec_len / 2.0,
                -max_height * 0.15,
                f'{sec["grad"]} permille',
                ha='center',
                va='top',
                fontsize=grad_fontsize,
                color='brown',
            )
            section_idx += 1
        current_pos += sec_len

    ax3.axvline(x=current_pos, color='gray', linestyle=':', alpha=0.5)
    ax3.set_xlim([env.start_pos, current_pos])
    ax3.set_ylabel('Energy Units', fontsize=label_fontsize)
    ax3.set_xlabel('Position (m)', fontsize=label_fontsize)
    ax3.tick_params(axis='both', labelsize=tick_fontsize)
    ax3.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    section_split_k = 100
    resistance_episode_schedule = [
        (0, 600, 1.0),
        (600, 1200, 3.0),
        (1200, 1800, 5.0),
    ]
    env = VisualTrainEnvironment(section_split_k=section_split_k)
    control_sections = env.control_section_count
    action_dim = control_sections * control_sections + control_sections

    agent = SACAgent(state_dim=control_sections, action_dim=action_dim)
    buffer = ReservoirReplayBuffer(capacity=5000)

    scheduled_trip_time = 200.0
    max_episodes = 1800
    total_energy_units = 118
    extra_energy_allowance = 250
    energy_search_min_units = 80
    energy_search_max_units = 360
    energy_search_step_units = 5
    max_transfer_steps = 300
    target_precision = 0.4
    initial_search_tolerance = 0.5
    accepted_error_increase = 0.1
    adaptive_add_error_threshold = 0.5
    adaptive_add_patience = 5
    adaptive_unreachable_patience = 3
    adaptive_early_rollback_threshold = 1.0
    phase_observation_episodes = 10
    stable_base_window = 10
    stable_base_unit_band = 5
    local_exploration_extra_units = 5
    phase_calibration_attempts_per_budget = 120
    initial_exploration_rate = 0.12
    min_exploration_rate = 0.02
    max_initial_attempts = 2000
    run_sensitivity_analysis = False

    if run_sensitivity_analysis:
        analyze_energy_budget_sensitivity(
            env,
            scheduled_trip_time,
            min_units=100,
            max_units=140,
            step_units=5,
            max_attempts_per_budget=1000,
            target_tolerance=initial_search_tolerance,
        )

    env.set_global_resistance_scale(resistance_episode_schedule[0][2])
    print('Searching initial feasible allocation...', flush=True)
    if total_energy_units is None:
        total_energy_units, initial_state, initial_time = determine_total_energy_units(
            env,
            scheduled_trip_time,
            energy_search_min_units,
            energy_search_max_units,
            step_units=energy_search_step_units,
            max_attempts_per_budget=max_initial_attempts,
            target_tolerance=initial_search_tolerance,
        )
    else:
        try:
            initial_state, initial_time = build_feasible_initial_state(
                env,
                scheduled_trip_time,
                total_energy_units,
                max_attempts=max_initial_attempts,
                target_tolerance=initial_search_tolerance,
            )
        except RuntimeError:
            print(
                f'Fixed energy units {total_energy_units} is not feasible. '
                f'Searching {energy_search_min_units}-{energy_search_max_units} instead...'
            )
            total_energy_units, initial_state, initial_time = determine_total_energy_units(
                env,
                scheduled_trip_time,
                energy_search_min_units,
                energy_search_max_units,
                step_units=energy_search_step_units,
                max_attempts_per_budget=max_initial_attempts,
                target_tolerance=initial_search_tolerance,
            )
    max_total_units = total_energy_units + extra_energy_allowance

    best_state = None
    best_time_error = float('inf')
    best_history = None

    ep_rewards = []
    ep_times = []
    ep_energy = []
    base_energy = []
    train_times = []
    consecutive_late_failures = 0
    consecutive_unreachable_failures = 0
    phase_observation_remaining = 0
    last_resistance_scale = None
    last_phase_start = None
    stable_base_candidates = []
    phase_calibration_pending = False
    phase_policies = []

    print('SAC training started...')
    print(
        f'Initial feasible time: {initial_time:.3f}s | InitialDiff: {initial_time - scheduled_trip_time:.3f}s '
        f'| TargetPrecision: +/-{target_precision:.3f}s | Base energy units: {int(np.sum(initial_state))} '
        f'| Max energy units: {max_total_units} '
        f'| Exploration: {initial_exploration_rate:.2f}->{min_exploration_rate:.2f}'
    )
    for ep in range(max_episodes):
        resistance_scale = get_episode_resistance_scale(ep, resistance_episode_schedule)
        env.set_global_resistance_scale(resistance_scale)
        if last_resistance_scale is None or resistance_scale != last_resistance_scale:
            if last_resistance_scale is not None:
                phase_policies.append(
                    {
                        "start": last_phase_start,
                        "end": ep,
                        "scale": last_resistance_scale,
                        "state": np.copy(initial_state),
                    }
                )
            consecutive_late_failures = 0
            consecutive_unreachable_failures = 0
            phase_observation_remaining = phase_observation_episodes
            last_resistance_scale = resistance_scale
            last_phase_start = ep
            stable_base_candidates = []
            phase_calibration_pending = ep > 0

        state = np.copy(initial_state)
        current_time, history = env.run_and_log_trip(state)
        episode_repair_text = ''
        base_unreachable = current_time >= env.stall_time_penalty

        if base_unreachable:
            state, current_time, history, added_sections = repair_unreachable_allocation(
                env,
                state,
                scheduled_trip_time,
                max_total_units,
            )
            if current_time < env.stall_time_penalty:
                episode_repair_text = (
                    f' | TempReachRepairStart: +{len(added_sections)} units'
                )
            else:
                raise RuntimeError(
                    f"Unable to repair unreachable base allocation at episode {ep}. "
                    f"Increase extra_energy_allowance or max_total_units."
                )

        base_units_for_episode = int(np.sum(initial_state))
        episode_max_units = min(
            max_total_units,
            max(int(np.sum(state)), base_units_for_episode + local_exploration_extra_units),
        )

        step_count = 0
        total_reward = 0.0
        exploration_rate = max(
            min_exploration_rate,
            initial_exploration_rate * (1.0 - ep / max_episodes),
        )

        while step_count < max_transfer_steps:
            current_time_stalled = current_time >= env.stall_time_penalty
            allow_add_action = (
                current_time_stalled
                or current_time > scheduled_trip_time + target_precision
            )

            if random.random() < exploration_rate:
                action = sample_valid_energy_action(
                    state,
                    control_sections,
                    episode_max_units,
                    allow_add=allow_add_action,
                )
                if action is None:
                    break
            else:
                action = agent.select_action(normalize_state(state, max_total_units))
                action_type, _, _ = decode_energy_action(action, control_sections)
                if action_type == "add" and not allow_add_action:
                    fallback_action = sample_valid_energy_action(
                        state,
                        control_sections,
                        episode_max_units,
                        allow_add=False,
                    )
                    if fallback_action is not None:
                        action = fallback_action

            next_state, is_valid_action = apply_energy_action(
                state,
                action,
                control_sections,
                episode_max_units,
            )

            if not is_valid_action:
                fallback_action = sample_valid_energy_action(
                    state,
                    control_sections,
                    episode_max_units,
                    allow_add=allow_add_action,
                )
                if fallback_action is not None and fallback_action != action:
                    action = fallback_action
                    next_state, is_valid_action = apply_energy_action(
                        state,
                        action,
                        control_sections,
                        episode_max_units,
                    )

            if not is_valid_action:
                reward = -5.0
                done = 1.0
                buffer.push(
                    normalize_state(state, max_total_units),
                    action,
                    reward,
                    normalize_state(next_state, max_total_units),
                    done,
                )
                agent.update(buffer, batch_size=32)
                total_reward += reward
                break

            new_time, new_history = env.run_and_log_trip(next_state)
            current_error = abs(current_time - scheduled_trip_time)
            new_error = abs(new_time - scheduled_trip_time)
            energy_units = int(np.sum(next_state))

            new_time_stalled = new_time >= env.stall_time_penalty
            if new_time_stalled:
                repaired_state, repaired_time, repaired_history, added_sections = repair_unreachable_allocation(
                    env,
                    next_state,
                    scheduled_trip_time,
                    max_total_units,
                )
                if repaired_time < env.stall_time_penalty:
                    next_state = repaired_state
                    new_time = repaired_time
                    new_history = repaired_history
                    new_error = abs(new_time - scheduled_trip_time)
                    energy_units = int(np.sum(next_state))
                    new_time_stalled = new_time >= env.stall_time_penalty
                    episode_repair_text += (
                        f' | TempReachRepairStep: +{len(added_sections)} units'
                    )

            if (new_time_stalled and new_time >= current_time) or new_error > current_error + accepted_error_increase:
                reward = compute_schedule_reward(
                    current_time,
                    new_time,
                    scheduled_trip_time,
                    energy_units,
                    max_total_units,
                    terminal=True,
                ) - 5.0
                done = 1.0
                buffer.push(
                    normalize_state(state, max_total_units),
                    action,
                    reward,
                    normalize_state(next_state, max_total_units),
                    done,
                )
                agent.update(buffer, batch_size=32)
                total_reward += reward
                break
            elif step_count + 1 >= max_transfer_steps:
                reward = compute_schedule_reward(
                    current_time,
                    new_time,
                    scheduled_trip_time,
                    energy_units,
                    max_total_units,
                    terminal=True,
                )
                done = 1.0
            else:
                reward = compute_schedule_reward(
                    current_time,
                    new_time,
                    scheduled_trip_time,
                    energy_units,
                    max_total_units,
                )
                done = 0.0

            buffer.push(
                normalize_state(state, max_total_units),
                action,
                reward,
                normalize_state(next_state, max_total_units),
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

        stable_base_text = ''
        if current_time < env.stall_time_penalty and current_time_error <= target_precision:
            stable_base_candidates.append(
                {
                    "state": np.copy(state),
                    "time": current_time,
                    "error": current_time_error,
                    "units": int(np.sum(state)),
                }
            )
            if len(stable_base_candidates) > stable_base_window:
                stable_base_candidates.pop(0)
        else:
            stable_base_candidates = []

        if len(stable_base_candidates) >= stable_base_window:
            stable_units = [item["units"] for item in stable_base_candidates]
            unit_span = max(stable_units) - min(stable_units)
            if unit_span <= stable_base_unit_band and min(stable_units) >= int(np.sum(initial_state)):
                stable_best = min(
                    stable_base_candidates,
                    key=lambda item: (item["units"], item["error"]),
                )
                if (
                    stable_best["units"] != int(np.sum(initial_state))
                    or not np.array_equal(stable_best["state"], initial_state)
                ):
                    initial_state = np.copy(stable_best["state"])
                    total_energy_units = stable_best["units"]
                    consecutive_late_failures = 0
                    consecutive_unreachable_failures = 0
                    phase_observation_remaining = 0
                    stable_base_text = (
                        f' | StableBase: units {total_energy_units}, '
                        f'time {stable_best["time"]:.3f}s, span {unit_span}'
                    )
                stable_base_candidates = []

        if base_unreachable:
            consecutive_unreachable_failures += 1
        else:
            consecutive_unreachable_failures = 0

        if current_time > scheduled_trip_time + adaptive_add_error_threshold:
            consecutive_late_failures += 1
        else:
            consecutive_late_failures = 0

        adaptive_add_text = ''
        adaptive_remove_text = ''
        phase_calibration_text = ''
        can_update_base = phase_observation_remaining <= 0
        should_add_for_unreachable = consecutive_unreachable_failures >= adaptive_unreachable_patience
        should_add_for_late = consecutive_late_failures >= adaptive_add_patience
        should_remove_for_early = (
            current_time < scheduled_trip_time - adaptive_early_rollback_threshold
            and current_time < env.stall_time_penalty
        )

        if can_update_base and phase_calibration_pending:
            calibrated_state, calibrated_time, calibrated_exact = calibrate_phase_base(
                env,
                initial_state,
                scheduled_trip_time,
                max_total_units,
                target_precision,
                max_attempts_per_budget=phase_calibration_attempts_per_budget,
            )
            if calibrated_time is not None:
                initial_state = calibrated_state
                total_energy_units = int(np.sum(initial_state))
                consecutive_late_failures = 0
                consecutive_unreachable_failures = 0
                stable_base_candidates = []
                phase_calibration_pending = False
                phase_calibration_text = (
                    f' | PhaseCalibrate: units {total_energy_units}, '
                    f'time {calibrated_time:.3f}s, exact {calibrated_exact}'
                )
            else:
                phase_calibration_pending = False
                phase_calibration_text = ' | PhaseCalibrate: failed'
        elif should_remove_for_early:
            adapted_state, removed_idx, adapted_time = choose_best_energy_removal(
                env,
                initial_state,
                scheduled_trip_time,
            )
            if removed_idx >= 0:
                initial_state = adapted_state
                total_energy_units = int(np.sum(initial_state))
                consecutive_late_failures = 0
                consecutive_unreachable_failures = 0
                stable_base_candidates = []
                adaptive_remove_text = (
                    f' | AdaptiveRemove: section {removed_idx}, '
                    f'base units {total_energy_units}, predicted {adapted_time:.1f}s'
                )
            else:
                adaptive_remove_text = ' | AdaptiveRemove: skipped'
        elif can_update_base and (should_add_for_unreachable or should_add_for_late):
            candidate_base = initial_state
            adapted_state, added_idx, adapted_time = choose_best_energy_addition(
                env,
                candidate_base,
                scheduled_trip_time,
                max_total_units,
            )
            if added_idx >= 0:
                initial_state = adapted_state
                total_energy_units = int(np.sum(initial_state))
                consecutive_late_failures = 0
                consecutive_unreachable_failures = 0
                adaptive_add_text = (
                    f' | AdaptiveAdd: section {added_idx}, '
                    f'base units {total_energy_units}, predicted {adapted_time:.1f}s'
                )
            else:
                consecutive_late_failures = 0
                consecutive_unreachable_failures = 0
                adaptive_add_text = ' | AdaptiveAdd: skipped, max energy reached'
        elif phase_observation_remaining > 0:
            phase_observation_remaining -= 1

        base_energy.append(np.sum(initial_state))

        time_diff = current_time - scheduled_trip_time
        print(
            f'Episode: {ep:03d} | Reward: {total_reward:8.3f} | Time: {current_time:9.1f}s '
            f'| TimeDiff: {time_diff:9.1f}s | EnergyUnits: {int(np.sum(state)):3d} '
            f'| BaseUnits: {int(np.sum(initial_state)):3d} | RScale: {resistance_scale:.1f} '
            f'| ExploreCap: {episode_max_units:3d} | Explore: {exploration_rate:.3f}'
            f'{episode_repair_text}{stable_base_text}{phase_calibration_text}'
            f'{adaptive_remove_text}{adaptive_add_text}'
        )

    print('Training finished. Plotting...')
    plot_convergence(ep_rewards, ep_times, ep_energy, scheduled_trip_time, base_energy)

    if last_resistance_scale is not None:
        phase_policies.append(
            {
                "start": last_phase_start,
                "end": max_episodes,
                "scale": last_resistance_scale,
                "state": np.copy(initial_state),
            }
        )

    for idx, policy in enumerate(phase_policies, start=1):
        env.set_global_resistance_scale(policy["scale"])
        policy_time, policy_history = env.run_and_log_trip(policy["state"])
        print(
            f'Phase {idx}: episodes {policy["start"]}-{policy["end"] - 1} | '
            f'RScale: {policy["scale"]:.1f} | Time: {policy_time:.3f}s | '
            f'TimeDiff: {policy_time - scheduled_trip_time:.3f}s | '
            f'EnergyUnits: {int(np.sum(policy["state"]))}'
        )
        plot_results(
            policy_history,
            policy["state"],
            scheduled_trip_time,
            env,
            title_suffix=(
                f'Phase {idx} / Episodes {policy["start"]}-{policy["end"] - 1} '
                f'/ Base Resistance x{policy["scale"]:.1f}'
            ),
        )

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
