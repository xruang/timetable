"""Bidirectional online hierarchical MAPPO with lower driving-policy rollout calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from .config import (
        DEFAULT_DOWN_CURVATURE_PATH,
        DEFAULT_DOWN_SLOPE_PATH,
        DEFAULT_LINE_DIR,
        DEFAULT_STATION_DWELL_S,
        REAL_SUPPLY_ZONE_NAMES,
        REAL_SUPPLY_ZONE_SECTION_GROUPS,
    )
    from .lower_control import (
        DQNAgent,
        DRIVING_SMOOTHNESS,
        EXPERT_KNOWLEDGE,
        LOWER_REWARD,
        LowerSectionEnvironment,
        train_lower_agents,
    )
    from .lower_continuous import (
        DDPGAgent,
        SACAgent,
        train_lower_ddpg_agents,
        train_lower_sac_agents,
    )
    from .cvxpy_lower import CvxpyLowerOptions, CvxpyLowerSolver
    from .mappo import IndependentActorMAPPOAgent, save_multi_train_checkpoint
    from .train_model import TRAIN_CONFIGS, TrainDynamics
    from .online_hierarchical import (
        OnlineHierarchicalTimetableEnvironment,
        train_online_mappo,
        write_online_operation_rows,
    )
    from .bidirectional_model import (
        build_bidirectional_model,
        down_line_profile_rows,
        seed_everything,
        station_names,
    )
    from .visualization import (
        lower_result_summary_rows,
        plot_section_operation_overview,
        plot_train_operation,
        plot_training_process,
        write_csv,
    )
except ImportError:
    from config import (
        DEFAULT_DOWN_CURVATURE_PATH,
        DEFAULT_DOWN_SLOPE_PATH,
        DEFAULT_LINE_DIR,
        DEFAULT_STATION_DWELL_S,
        REAL_SUPPLY_ZONE_NAMES,
        REAL_SUPPLY_ZONE_SECTION_GROUPS,
    )
    from lower_control import (
        DQNAgent,
        DRIVING_SMOOTHNESS,
        EXPERT_KNOWLEDGE,
        LOWER_REWARD,
        LowerSectionEnvironment,
        train_lower_agents,
    )
    from lower_continuous import (
        DDPGAgent,
        SACAgent,
        train_lower_ddpg_agents,
        train_lower_sac_agents,
    )
    from cvxpy_lower import CvxpyLowerOptions, CvxpyLowerSolver
    from mappo import IndependentActorMAPPOAgent, save_multi_train_checkpoint
    from train_model import TRAIN_CONFIGS, TrainDynamics
    from online_hierarchical import (
        OnlineHierarchicalTimetableEnvironment,
        train_online_mappo,
        write_online_operation_rows,
    )
    from bidirectional_model import (
        build_bidirectional_model,
        down_line_profile_rows,
        seed_everything,
        station_names,
    )
    from visualization import (
        lower_result_summary_rows,
        plot_section_operation_overview,
        plot_train_operation,
        plot_training_process,
        write_csv,
    )


SCRIPT_DIR = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Online bidirectional hierarchical optimization: upper MAPPO calls "
            "the lower driving policy for every station-to-station rollout."
        )
    )
    parser.add_argument("--line-dir", type=Path, default=DEFAULT_LINE_DIR)
    parser.add_argument("--down-curvature", type=Path, default=DEFAULT_DOWN_CURVATURE_PATH)
    parser.add_argument("--down-slope", type=Path, default=DEFAULT_DOWN_SLOPE_PATH)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "results_bidirectional_online")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--experiment-stage",
        choices=("lower", "hierarchical"),
        default="hierarchical",
        help="lower: train/save only the driving policy; hierarchical: also train upper MAPPO.",
    )
    parser.add_argument("--experiment-config", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default=None,
        help=(
            "Legacy shortcut. If set, applies the same device to lower driving policy "
            "and upper MAPPO."
        ),
    )
    parser.add_argument(
        "--lower-device",
        choices=("auto", "cpu", "cuda"),
        default="cpu",
        help="Torch device for lower driving-policy rollout inference.",
    )
    parser.add_argument(
        "--upper-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Torch device for upper MAPPO actor/critic updates.",
    )
    parser.add_argument("--up-trains", type=int, default=6)
    parser.add_argument("--down-trains", type=int, default=6)
    parser.add_argument(
        "--simulation-dt-s",
        type=float,
        default=0.5,
        help=(
            "列车纵向动力学仿真步长（秒）。0.1 s 更适合核对终点停车和"
            "±0.5 s 到达误差，但训练计算量约为 0.5 s 时的 5 倍。"
        ),
    )
    parser.add_argument(
        "--lower-backend",
        choices=("policy", "cvxpy"),
        default="policy",
        help=(
            "policy uses the trained DQN/DDPG/SAC lower controller; cvxpy "
            "uses the licence-free model-based lower optimizer."
        ),
    )
    parser.add_argument(
        "--lower-algorithm",
        choices=("dqn", "ddpg", "sac"),
        default="dqn",
        help="Lower driving policy algorithm used for online rollout calls.",
    )
    parser.add_argument(
        "--convex-spatial-segments",
        type=int,
        default=600,
        help="Spatial cells used by the CVXPY lower optimizer.",
    )
    parser.add_argument(
        "--convex-speed-limit-ratio",
        type=float,
        default=0.8,
        help="Speed-limit reduction used by the CVXPY lower optimizer.",
    )
    parser.add_argument(
        "--convex-solver",
        choices=("CLARABEL", "SCS"),
        default="CLARABEL",
        help="Open-source CVXPY solver; CLARABEL is recommended.",
    )
    parser.add_argument(
        "--convex-time-limit-s",
        type=float,
        default=None,
        help="Optional time limit for one lower convex solve.",
    )
    parser.add_argument(
        "--convex-runtime-search-step-s",
        type=float,
        default=1.0,
        help=(
            "Step used to raise a baseline runtime that is infeasible after "
            "the speed-limit reduction."
        ),
    )
    parser.add_argument(
        "--convex-runtime-search-max-extra-s",
        type=float,
        default=30.0,
        help="Maximum automatic increase of an infeasible baseline runtime.",
    )
    parser.add_argument(
        "--lower-sac-gamma",
        type=float,
        default=0.99,
        help=(
            "SAC 折扣因子。0.9998 可显著减弱长回合中终点准点奖励的衰减；"
            "仅对 SAC 生效。"
        ),
    )
    parser.add_argument(
        "--lower-sac-initial-alpha",
        type=float,
        default=0.02,
        help="SAC初始熵温度；覆盖预热阶段保持固定。",
    )
    parser.add_argument(
        "--lower-sac-min-alpha",
        type=float,
        default=0.01,
        help="SAC自动温度调节的下限。",
    )
    parser.add_argument(
        "--lower-sac-alpha-learning-rate",
        type=float,
        default=3e-5,
        help="SAC温度参数学习率。",
    )
    parser.add_argument(
        "--lower-sac-alpha-warmup-episodes",
        type=int,
        default=1026,
        help="固定alpha的覆盖预热回合数；1026等于342个组合完整三轮。",
    )
    parser.add_argument(
        "--lower-schedule-deviation-state",
        action="store_true",
        help=(
            "在连续下层策略状态末尾加入 travelled_ratio - elapsed_ratio；"
            "正值表示运行进度领先，负值表示落后。"
        ),
    )
    parser.add_argument(
        "--lower-schedule-deviation-scale",
        type=float,
        default=1.0,
        help="有符号运行进度偏差的输入缩放倍数；仅在启用该状态时生效。",
    )
    parser.add_argument(
        "--lower-shared-context-state",
        action="store_true",
        help=(
            "为连续共享策略加入方向符号和归一化站间位置两个状态量，"
            "避免不同方向、站间出现状态混叠。"
        ),
    )
    parser.add_argument(
        "--lower-replay-capacity",
        type=int,
        default=100000,
        help="下层经验回放容量（按仿真步计数）。",
    )
    parser.add_argument(
        "--share-lower-continuous",
        action="store_true",
        help=(
            "Use one shared continuous lower controller for all trains. This "
            "is the preferred option for DDPG/SAC with mass-aware state input."
        ),
    )
    parser.add_argument(
        "--share-lower-dqn",
        action="store_true",
        help=(
            "Use one shared DQN lower controller for all trains. This is useful "
            "when DQN receives mass-aware state input and random-mass training."
        ),
    )
    parser.add_argument(
        "--share-lower-ddpg",
        action="store_true",
        help=(
            "Backward-compatible alias for --share-lower-continuous when "
            "--lower-algorithm ddpg is used."
        ),
    )
    parser.add_argument(
        "--ddpg-random-mass-training",
        action="store_true",
        help=(
            "During lower DDPG pretraining, randomize train mass and derive "
            "traction constraints from the configured mass anchor table."
        ),
    )
    parser.add_argument("--ddpg-mass-min-t", type=float, default=235.0)
    parser.add_argument("--ddpg-mass-max-t", type=float, default=275.0)
    parser.add_argument("--ddpg-anchor-mass-probability", type=float, default=0.15)
    parser.add_argument("--ddpg-boundary-mass-probability", type=float, default=0.05)
    parser.add_argument("--lower-episodes", type=int, default=3000)
    parser.add_argument(
        "--lower-bc-episodes",
        type=int,
        default=0,
        help=(
            "Behavior-cloning pretraining episodes for continuous lower "
            "controllers before RL fine-tuning. Use with DDPG for "
            "pretrain+fine-tune experiments."
        ),
    )
    parser.add_argument(
        "--lower-log-interval",
        type=int,
        default=10,
        help=(
            "Print lower-policy training diagnostics every N episodes. "
            "Use 0 to disable periodic lower training logs."
        ),
    )
    parser.add_argument(
        "--lower-checkpoint-interval",
        type=int,
        default=0,
        help=(
            "Autosave the current lower controller every N training episodes. "
            "Use 0 to disable; 500 is recommended for server runs."
        ),
    )
    parser.add_argument(
        "--lower-plot-interval",
        type=int,
        default=100,
        help=(
            "Save one lower continuous-control speed trajectory figure every "
            "N fine-tuning episodes. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--lower-policy-mode",
        choices=("fresh", "continue", "reuse"),
        default="fresh",
    )
    parser.add_argument("--lower-policy-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--upper-policy-mode",
        choices=("fresh", "continue", "reuse"),
        default="fresh",
        help=(
            "Load upper MAPPO weights from a checkpoint. continue also restores "
            "optimizer states when they are present; reuse loads only networks."
        ),
    )
    parser.add_argument("--upper-policy-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--repeat-lower-policy-weights",
        action="store_true",
        help=(
            "Allow a checkpoint with one or two lower policies to be reused as "
            "templates for all trains. With two policies, the first is used for "
            "up trains and the second for down trains."
        ),
    )
    # 下层训练目标时间扰动：训练时使用“真实站间时间 ± 本值”生成样本。
    # 这不是上层时刻表动作；设为 0 表示只按 config 中的真实站间时间训练。
    parser.add_argument(
        "--lower-target-span",
        type=int,
        default=0,
        help=(
            "下层训练的目标时间扰动范围（秒）：目标时间 = config 真实站间时间 "
            "+ [-N, N] 内的随机整数；不是上层动作。设为 0 表示不扰动。"
        ),
    )
    parser.add_argument("--upper-episodes", type=int, default=1000)
    parser.add_argument(
        "--lower-arrival-tolerance",
        type=float,
        default=0.5,
        help=(
            "Lower-layer feasible arrival error in seconds. For example, 0.5 "
            "means actual section time may differ from the target by +/-0.5 s."
        ),
    )
    parser.add_argument(
        "--lower-speed-limit-ratio",
        type=float,
        default=1.0,
        help=(
            "下层控制采用的线路限速比例。1.0 表示使用线路真实限速；"
            "安全动作屏蔽仍负责阻止超速动作。"
        ),
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=10,
        help=(
            "Print one online event progress line every N station-to-station "
            "events. Use 0 to disable."
        ),
    )
    parser.add_argument("--initial-headway", type=float, default=480.0)
    parser.add_argument("--minimum-headway", type=float, default=120.0)
    parser.add_argument("--station-dwell-s", type=float, default=DEFAULT_STATION_DWELL_S)
    # 上层动作 2：每个站间运行时间相对 config 基准值的最大调整幅度。
    # 如果站间运行时间视为固定，应设为 0；与 lower-target-span 无关。
    parser.add_argument(
        "--runtime-adjustment-span",
        type=float,
        default=0.0,
        help=(
            "上层对站间运行时间的最大调整幅度（秒），实际动作范围约为 "
            "[-N, N]。若站间时间固定，设为 0。"
        ),
    )
    # 上层动作 1：列车到站后的停站时间调整幅度，会影响后续所有到发时刻。
    parser.add_argument(
        "--dwell-adjustment-span",
        type=float,
        default=10.0,
        help=(
            "上层对停站时间的最大调整幅度（秒），实际动作范围约为 "
            "[-N, N]；设为 0 表示停站时间固定。"
        ),
    )
    parser.add_argument(
        "--rollout-cache-quantum",
        type=float,
        default=0.1,
        help="Quantization in seconds for lower rollout cache target runtimes.",
    )
    parser.add_argument("--traction-energy-weight", type=float, default=0.5)
    parser.add_argument("--regenerative-match-weight", type=float, default=0.5)
    return parser


def configure_run_outputs(args: argparse.Namespace) -> None:
    base_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = base_name
    run_dir = args.output_dir / run_name
    suffix = 2
    while run_dir.exists():
        run_name = f"{base_name}_{suffix:02d}"
        run_dir = args.output_dir / run_name
        suffix += 1
    args.run_name = run_name
    args.output_dir = run_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.checkpoint is None:
        checkpoint_name = (
            "lower_policy_checkpoint.pt"
            if args.experiment_stage == "lower"
            else "hierarchical_mappo_bidirectional_online_checkpoint.pt"
        )
        args.checkpoint = run_dir / checkpoint_name


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def source_fingerprints() -> dict[str, str]:
    """Return an exact code fingerprint even when Git metadata is unavailable."""
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(SCRIPT_DIR.glob("*.py"))
    }


def write_run_metadata(
    args: argparse.Namespace,
    train_configs: Sequence[Any],
    initial_timetables: np.ndarray,
    direction_labels: Sequence[str],
) -> None:
    experiment_config = None
    if args.experiment_config is not None:
        with args.experiment_config.open("r", encoding="utf-8") as handle:
            experiment_config = json.load(handle)
        with (args.output_dir / "experiment_config.json").open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(experiment_config, handle, ensure_ascii=False, indent=2)
    metadata = {
        "run_name": args.run_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": Path(__file__).name,
        "experiment_type": "online_hierarchical_lower_policy_rollout",
        "experiment_stage": args.experiment_stage,
        "experiment_config": experiment_config,
        "source_sha256": source_fingerprints(),
        "lower_algorithm": (
            f"cvxpy_{args.convex_solver.lower()}"
            if args.lower_backend == "cvxpy"
            else args.lower_algorithm
        ),
        "lower_backend": args.lower_backend,
        "model_based_lower": (
            {
                "solver": args.convex_solver,
                "spatial_segments": args.convex_spatial_segments,
                "speed_limit_ratio": args.convex_speed_limit_ratio,
                "time_limit_s": args.convex_time_limit_s,
                "curve_resistance": False,
                "symmetric_constant_power_braking": True,
                "outputs_are_physically_recomputed": True,
            }
            if args.lower_backend == "cvxpy"
            else None
        ),
        "share_lower_ddpg": bool(getattr(args, "share_lower_ddpg", False)),
        "share_lower_continuous": bool(
            getattr(args, "share_lower_continuous", False)
            or getattr(args, "share_lower_ddpg", False)
        ),
        "lower_behavior_clone_pretrain_episodes": getattr(
            args,
            "lower_bc_episodes",
            0,
        ),
        "upper_policy_mode": getattr(args, "upper_policy_mode", "fresh"),
        "upper_policy_checkpoint": getattr(args, "upper_policy_checkpoint", None),
        "lower_training_speed_snapshot_interval": getattr(
            args,
            "lower_plot_interval",
            0,
        ),
        "ddpg_mass_training": {
            "random_mass_training": bool(
                getattr(args, "ddpg_random_mass_training", False)
            ),
            "mass_min_t": getattr(args, "ddpg_mass_min_t", None),
            "mass_max_t": getattr(args, "ddpg_mass_max_t", None),
            "anchor_mass_probability": getattr(
                args,
                "ddpg_anchor_mass_probability",
                None,
            ),
            "boundary_mass_probability": getattr(
                args,
                "ddpg_boundary_mass_probability",
                None,
            ),
        },
        "arguments": vars(args),
        "lower_policy_weight_reuse_note": (
            "No learned lower policy is used; CVXPY solves each cache miss."
            if args.lower_backend == "cvxpy"
            else (
                "A small lower-policy checkpoint is repeated across trains; this "
                "is a warm-start comparison rather than a fully train-specific "
                "lower-policy pretraining run."
                if getattr(args, "repeat_lower_policy_weights", False)
                else "Lower policies are loaded one-to-one or trained in this run."
            )
        ),
        "direction_labels": list(direction_labels),
        "initial_timetables_s": initial_timetables,
        "train_configs": [asdict(config) for config in train_configs],
        "lower_reward": asdict(LOWER_REWARD),
        "driving_smoothness": asdict(DRIVING_SMOOTHNESS),
        "expert_knowledge_action_shield": asdict(EXPERT_KNOWLEDGE),
        "lower_state_design": (
            {
                "algorithm": f"cvxpy_{args.convex_solver.lower()}",
                "state_dim": 0,
                "features": [
                    "section geometry and gradient",
                    "train mass and Davis resistance parameters",
                    "traction/braking force and power limits",
                    "target interstation runtime",
                ],
                "note": (
                    "No learned lower-policy state is used. The optimizer "
                    "returns a speed/force trajectory and physically recomputed "
                    "time and energy."
                ),
            }
            if args.lower_backend == "cvxpy"
            else
            {
                "algorithm": args.lower_algorithm,
                "state_dim": (
                    LowerSectionEnvironment.LEGACY_COMPACT_STATE_DIM
                    + int(args.lower_schedule_deviation_state)
                    + 2 * int(args.lower_shared_context_state)
                ),
                "features": [
                    "travelled_ratio",
                    "speed_mps_over_40",
                    "previous_acceleration_mps2_over_config_limit",
                    "elapsed_time_ratio",
                    "target_average_speed_mps_over_40",
                    "motor_car_mass_kg_over_200000",
                    "trailer_car_mass_kg_over_150000",
                    "previous_force_ratio",
                    "current_speed_limit_kmh_over_125",
                    "current_gradient_per_mille_over_30",
                    "current_curve_radius_feature",
                    "preview_safe_speed_mps_over_40",
                    "preview_limit_distance_ratio",
                ] + (
                    ["schedule_deviation_travelled_minus_elapsed"]
                    if args.lower_schedule_deviation_state
                    else []
                ) + (
                    ["direction_sign", "section_index_normalized"]
                    if args.lower_shared_context_state
                    else []
                ),
                "note": (
                    "Line conditions are direct lower-policy inputs in this "
                    "line-aware continuous-control state. "
                    + (
                        "Signed schedule deviation is explicit: positive means the "
                        "train is ahead of schedule, negative means it is behind. "
                        if args.lower_schedule_deviation_state
                        else "Signed schedule deviation is not included. "
                    )
                    + (
                        "Direction and normalized section position explicitly "
                        "condition the shared policy. "
                        if args.lower_shared_context_state
                        else "Shared-policy direction/section context is not included. "
                    )
                    +
                    "Traction force, braking force and traction power are not "
                    "state features; they remain train-model constraints."
                ),
            }
            if args.lower_algorithm in {"ddpg", "sac"}
            else {
                "algorithm": "dqn",
                "state_dim": LowerSectionEnvironment.STATE_DIM,
                "features": "legacy 15-dimensional lower observation",
            }
        ),
        "supply_zone_names": REAL_SUPPLY_ZONE_NAMES,
        "supply_zone_section_groups": REAL_SUPPLY_ZONE_SECTION_GROUPS,
        "online_information_flow": {
            "upper_to_lower": [
                "target runtime for the next interstation section",
                "station departure adjustment within the dwell-adjustment span",
                "current train/station decision context through the upper state",
            ],
            "lower_to_upper": [
                "speed trajectory",
                "traction and regenerative energy trajectories",
                "actual arrival time",
                "feasibility and speed-limit compliance indicators",
            ],
        },
    }
    with (args.output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(json_ready(metadata), handle, ensure_ascii=False, indent=2)


def save_lower_only_checkpoint(
    output_path: Path,
    lower_agents: Sequence[Any],
    train_configs: Sequence[Any],
    lower_algorithm: str,
) -> None:
    """Save lower controllers without creating or training an upper controller."""
    unique_agents: list[Any] = []
    seen_ids: set[int] = set()
    for agent in lower_agents:
        if id(agent) not in seen_ids:
            seen_ids.add(id(agent))
            unique_agents.append(agent)

    controller_states = []
    for controller in unique_agents:
        state: dict[str, Any] = {"policy": controller.policy.state_dict()}
        for attribute in (
            "target",
            "actor",
            "target_actor",
            "critic",
            "target_critic",
            "critic1",
            "critic2",
            "target_critic1",
            "target_critic2",
            "optimizer",
            "actor_optimizer",
            "critic_optimizer",
            "critic1_optimizer",
            "critic2_optimizer",
            "alpha_optimizer",
        ):
            item = getattr(controller, attribute, None)
            if item is not None and hasattr(item, "state_dict"):
                state[attribute] = item.state_dict()
        log_alpha = getattr(controller, "log_alpha", None)
        if log_alpha is not None:
            state["log_alpha"] = log_alpha.detach().cpu()
        if hasattr(controller, "update_count"):
            state["update_count"] = int(controller.update_count)
        controller_states.append(state)

    checkpoint_data = {
        "experiment_stage": "lower",
        "lower_algorithm": lower_algorithm,
        "shared_lower_policy": len(unique_agents) == 1,
        "lower_policies": [agent.policy.state_dict() for agent in unique_agents],
        "lower_controller_states": controller_states,
        "lower_action_semantics": getattr(unique_agents[0], "action_semantics", "discrete"),
        "train_parameters": [asdict(config) for config in train_configs],
        "driving_smoothness_parameters": asdict(DRIVING_SMOOTHNESS),
    }
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(checkpoint_data, temporary_path)
    temporary_path.replace(output_path)
    print(f"Saved lower-only checkpoint: {output_path.resolve()}")


def load_repeated_lower_policies_from_checkpoint(
    checkpoint_path: Path,
    lower_agents: Sequence[Any],
    train_configs: Sequence[Any],
) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Lower-policy checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    _validate_lower_action_semantics(checkpoint, lower_agents)
    controller_states = checkpoint.get("lower_controller_states")
    policies = checkpoint.get("lower_policies")
    if not policies and not controller_states:
        raise ValueError(
            f"Checkpoint does not contain lower_policies: {checkpoint_path}"
        )
    templates = controller_states or policies
    if len(templates) not in {1, 2, len(lower_agents)}:
        raise ValueError(
            "Repeated lower-policy loading only supports 1 policy, 2 "
            "direction-template policies, or one policy per train; checkpoint "
            f"has {len(templates)}, current experiment needs {len(lower_agents)}."
        )
    for train_idx, (agent, config) in enumerate(zip(lower_agents, train_configs)):
        if len(templates) == len(lower_agents):
            state_dict = templates[train_idx]
        elif len(templates) == 1:
            state_dict = templates[0]
        else:
            state_dict = templates[0] if str(config.name).startswith("Up ") else templates[1]
        _load_lower_controller_state(agent, state_dict)
    print(
        f"Loaded repeated lower policies from: {checkpoint_path} "
        f"({len(templates)} templates -> {len(lower_agents)} trains)"
    )


def load_exact_lower_controllers_from_checkpoint(
    checkpoint_path: Path,
    lower_agents: Sequence[Any],
) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Lower-policy checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    _validate_lower_action_semantics(checkpoint, lower_agents)
    controller_states = checkpoint.get("lower_controller_states")
    policies = checkpoint.get("lower_policies")
    templates = controller_states or policies
    if templates is None:
        raise ValueError(
            f"Checkpoint does not contain lower controller states: {checkpoint_path}"
        )
    if len(templates) != len(lower_agents):
        raise ValueError(
            "Lower-controller count does not match current train count: "
            f"checkpoint has {len(templates)}, current experiment needs "
            f"{len(lower_agents)}"
        )
    for agent, state_entry in zip(lower_agents, templates):
        _load_lower_controller_state(agent, state_entry)
    print(f"Loaded exact lower controllers from: {checkpoint_path}")


def load_shared_lower_policy_from_checkpoint(
    checkpoint_path: Path,
    shared_agent: Any,
) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Lower-policy checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    _validate_lower_action_semantics(checkpoint, [shared_agent])
    controller_states = checkpoint.get("lower_controller_states")
    if controller_states:
        _load_lower_controller_state(shared_agent, controller_states[0])
        print(f"Loaded shared lower controller from: {checkpoint_path}")
        return
    policies = checkpoint.get("lower_policies")
    if policies is None and checkpoint.get("lower_controller_states") is not None:
        policies = [item["policy"] for item in checkpoint["lower_controller_states"]]
    if not policies:
        raise ValueError(
            f"Checkpoint does not contain lower_policies: {checkpoint_path}"
        )
    state_dict = policies[0]
    if isinstance(state_dict, dict) and "policy" in state_dict:
        state_dict = state_dict["policy"]
    state_dict, adapted = adapt_lower_policy_state_dict(
        shared_agent.policy,
        state_dict,
    )
    shared_agent.policy.load_state_dict(state_dict)
    if hasattr(shared_agent, "target_actor"):
        shared_agent.target_actor.load_state_dict(state_dict)
    if hasattr(shared_agent, "target"):
        shared_agent.target.load_state_dict(state_dict)
    print(f"Loaded shared lower policy from: {checkpoint_path}")
    if adapted:
        print(
            "Adapted shared lower-policy input layer; new line-condition "
            "feature weights were initialized to zero."
        )


def _validate_lower_action_semantics(checkpoint: dict, agents: Sequence[Any]) -> None:
    for agent in agents:
        expected = getattr(agent, "action_semantics", None)
        if expected is not None and checkpoint.get("lower_action_semantics") != expected:
            raise ValueError(
                "Incompatible lower checkpoint: old critics use executed actions or "
                "old terminal dynamics. Start with --lower-policy-mode fresh; "
                "do not continue/reuse the old checkpoint."
            )


def _load_lower_controller_state(agent: Any, state_entry: Any) -> None:
    if isinstance(state_entry, dict) and "policy" in state_entry:
        policy_state = state_entry["policy"]
    else:
        policy_state = state_entry
    policy_state, adapted = adapt_lower_policy_state_dict(agent.policy, policy_state)
    agent.policy.load_state_dict(policy_state)
    for attribute in (
        "target",
        "actor",
        "target_actor",
        "critic",
        "target_critic",
        "critic1",
        "critic2",
        "target_critic1",
        "target_critic2",
    ):
        module = getattr(agent, attribute, None)
        if (
            module is not None
            and hasattr(module, "load_state_dict")
            and isinstance(state_entry, dict)
            and attribute in state_entry
        ):
            module.load_state_dict(state_entry[attribute])
    for attribute in (
        "optimizer",
        "actor_optimizer",
        "critic_optimizer",
        "critic1_optimizer",
        "critic2_optimizer",
        "alpha_optimizer",
    ):
        optimizer = getattr(agent, attribute, None)
        if (
            optimizer is not None
            and hasattr(optimizer, "load_state_dict")
            and isinstance(state_entry, dict)
            and attribute in state_entry
        ):
            optimizer.load_state_dict(state_entry[attribute])
            _move_optimizer_state_to_device(optimizer, agent.device)
    if isinstance(state_entry, dict) and "log_alpha" in state_entry and hasattr(agent, "log_alpha"):
        with torch.no_grad():
            agent.log_alpha.copy_(state_entry["log_alpha"].to(agent.device))
    if isinstance(state_entry, dict) and "update_count" in state_entry and hasattr(agent, "update_count"):
        agent.update_count = int(state_entry["update_count"])
    if adapted:
        print(
            "Adapted lower-policy input layer; new line-condition feature "
            "weights were initialized to zero."
        )


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def load_upper_controller_from_checkpoint(
    checkpoint_path: Path,
    upper_agent: IndependentActorMAPPOAgent,
    load_optimizers: bool,
) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Upper-policy checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    upper_actors = checkpoint.get("upper_actors")
    upper_critic = checkpoint.get("upper_critic")
    if upper_actors is None or upper_critic is None:
        raise ValueError(
            f"Checkpoint does not contain upper MAPPO weights: {checkpoint_path}"
        )
    if len(upper_actors) != len(upper_agent.actors):
        raise ValueError(
            "Upper actor count does not match current train count: "
            f"checkpoint has {len(upper_actors)}, current experiment needs "
            f"{len(upper_agent.actors)}"
        )
    for actor, state_dict in zip(upper_agent.actors, upper_actors):
        actor.load_state_dict(state_dict)
    upper_agent.critic.load_state_dict(upper_critic)
    if load_optimizers:
        actor_optimizer_states = checkpoint.get("upper_actor_optimizers")
        critic_optimizer_state = checkpoint.get("upper_critic_optimizer")
        if actor_optimizer_states is not None:
            if len(actor_optimizer_states) != len(upper_agent.actor_optimizers):
                raise ValueError(
                    "Upper actor-optimizer count does not match current train count: "
                    f"checkpoint has {len(actor_optimizer_states)}, current experiment needs "
                    f"{len(upper_agent.actor_optimizers)}"
                )
            for optimizer, optimizer_state in zip(
                upper_agent.actor_optimizers,
                actor_optimizer_states,
            ):
                optimizer.load_state_dict(optimizer_state)
                _move_optimizer_state_to_device(optimizer, upper_agent.device)
        if critic_optimizer_state is not None:
            upper_agent.critic_optimizer.load_state_dict(critic_optimizer_state)
            _move_optimizer_state_to_device(
                upper_agent.critic_optimizer,
                upper_agent.device,
            )
    print(
        f"Loaded upper MAPPO controller from: {checkpoint_path} "
        f"(optimizers={'yes' if load_optimizers else 'no'})"
    )


def adapt_lower_policy_state_dict(
    policy: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], bool]:
    current_state = policy.state_dict()
    saved_input = state_dict.get("layers.0.weight")
    current_input = current_state.get("layers.0.weight")
    if (
        saved_input is None
        or current_input is None
        or saved_input.shape == current_input.shape
    ):
        return state_dict, False
    can_adapt_input = (
        saved_input.ndim == current_input.ndim == 2
        and saved_input.shape[0] == current_input.shape[0]
        and saved_input.shape[1] < current_input.shape[1]
    )
    if not can_adapt_input:
        return state_dict, False
    adapted_state = dict(state_dict)
    adapted_input = current_input.clone()
    adapted_input.zero_()
    adapted_input[:, : saved_input.shape[1]] = saved_input
    adapted_state["layers.0.weight"] = adapted_input
    return adapted_state, True


def build_lower_agents(
    args: argparse.Namespace,
    train_configs: Sequence[Any],
    environments_by_train: Sequence[Sequence[LowerSectionEnvironment]],
    initial_timetables: np.ndarray,
    direction_labels: Optional[Sequence[str]] = None,
) -> tuple[list[Any], list[dict[str, Any]]]:
    if not 0.0 < args.lower_speed_limit_ratio <= 1.0:
        raise ValueError("lower_speed_limit_ratio must be in (0, 1].")
    # 所有下层算法使用同一线路限速比例；超速风险由环境中的安全动作屏蔽处理。
    for train_environments in environments_by_train:
        for environment in train_environments:
            environment.speed_limit_penalty_ratio = args.lower_speed_limit_ratio
    if args.lower_algorithm == "dqn":
        agent_cls = DQNAgent
    elif args.lower_algorithm == "ddpg":
        agent_cls = DDPGAgent
    elif args.lower_algorithm == "sac":
        agent_cls = SACAgent
    else:
        raise ValueError(f"Unsupported lower algorithm: {args.lower_algorithm}")
    lower_state_dim = (
        LowerSectionEnvironment.STATE_DIM
        if args.lower_algorithm == "dqn"
        else (
            LowerSectionEnvironment.LEGACY_COMPACT_STATE_DIM
            + int(args.lower_schedule_deviation_state)
            + 2 * int(args.lower_shared_context_state)
        )
    )
    if not 0.0 < args.lower_sac_gamma <= 1.0:
        raise ValueError("lower_sac_gamma must be in (0, 1].")
    if args.lower_sac_initial_alpha <= 0.0:
        raise ValueError("lower_sac_initial_alpha must be positive.")
    if not 0.0 < args.lower_sac_min_alpha <= args.lower_sac_initial_alpha:
        raise ValueError(
            "lower_sac_min_alpha must be positive and <= lower_sac_initial_alpha."
        )
    if args.lower_sac_alpha_learning_rate <= 0.0:
        raise ValueError("lower_sac_alpha_learning_rate must be positive.")
    if args.lower_sac_alpha_warmup_episodes < 0:
        raise ValueError("lower_sac_alpha_warmup_episodes must be >= 0.")
    if args.lower_schedule_deviation_scale <= 0.0:
        raise ValueError("lower_schedule_deviation_scale must be positive.")
    if args.lower_replay_capacity < 128:
        raise ValueError("lower_replay_capacity must be at least 128.")
    lower_agent_kwargs: dict[str, Any] = {
        "state_dim": lower_state_dim,
        "action_dim": (
            LowerSectionEnvironment.ACTION_DIM
            if args.lower_algorithm == "dqn"
            else 1
        ),
        "device_name": args.lower_device,
        "replay_capacity": args.lower_replay_capacity,
    }
    if args.lower_algorithm in {"ddpg", "sac"}:
        lower_agent_kwargs.update(
            {
                "include_schedule_deviation_state": args.lower_schedule_deviation_state,
                "schedule_deviation_scale": args.lower_schedule_deviation_scale,
                "include_shared_context_state": args.lower_shared_context_state,
            }
        )
    if args.lower_algorithm == "sac":
        lower_agent_kwargs.update(
            {
                "gamma": args.lower_sac_gamma,
                "initial_alpha": args.lower_sac_initial_alpha,
                "min_alpha": args.lower_sac_min_alpha,
                "alpha_learning_rate": args.lower_sac_alpha_learning_rate,
            }
        )
        for train_environments in environments_by_train:
            for environment in train_environments:
                environment.schedule_shaping_gamma = args.lower_sac_gamma
    share_lower_policy = (
        (
            args.lower_algorithm == "dqn"
            and args.share_lower_dqn
        )
        or (
            args.lower_algorithm in {"ddpg", "sac"}
            and (args.share_lower_continuous or args.share_lower_ddpg)
        )
    )
    lower_agents = [
        agent_cls(**lower_agent_kwargs)
        for _ in train_configs
    ]
    if share_lower_policy:
        shared_agent = lower_agents[0]
        lower_agents = [shared_agent for _ in train_configs]
        if direction_labels is not None:
            direction_counts: dict[str, int] = {}
            for label, environments in zip(direction_labels, environments_by_train):
                direction_counts[str(label)] = direction_counts.get(str(label), 0) + len(environments)
            coverage = ", ".join(
                f"{direction}={count} section-envs"
                for direction, count in sorted(direction_counts.items())
            )
            print(
                f"Shared {args.lower_algorithm.upper()} lower training covers both directions: "
                f"{coverage}"
            )
    lower_history: list[dict[str, Any]] = []

    def autosave_lower_training(
        phase: str,
        completed_episodes: int,
        current_history: list[dict[str, Any]],
    ) -> None:
        autosave_path = args.output_dir / "lower_policy_autosave.pt"
        save_lower_only_checkpoint(
            autosave_path,
            lower_agents,
            train_configs,
            args.lower_algorithm,
        )
        write_csv(
            args.output_dir / "lower_training_autosave_history.csv",
            [*lower_history, *current_history],
        )
        with (args.output_dir / "lower_autosave_status.json").open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                {
                    "phase": phase,
                    "completed_episodes": completed_episodes,
                    "saved_at": datetime.now().isoformat(timespec="seconds"),
                    "checkpoint": str(autosave_path),
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )
    if args.lower_policy_mode in {"continue", "reuse"}:
        if args.lower_policy_checkpoint is None:
            raise ValueError(
                "--lower-policy-checkpoint is required for continue/reuse mode"
            )
        if share_lower_policy:
            load_shared_lower_policy_from_checkpoint(
                args.lower_policy_checkpoint,
                lower_agents[0],
            )
        elif args.repeat_lower_policy_weights:
            load_repeated_lower_policies_from_checkpoint(
                args.lower_policy_checkpoint,
                lower_agents,
                train_configs,
            )
        else:
            load_exact_lower_controllers_from_checkpoint(
                args.lower_policy_checkpoint,
                lower_agents,
            )
    def environment_tags_for_train(train_idx: int) -> list[dict[str, Any]]:
        direction = (
            direction_labels[train_idx]
            if direction_labels is not None
            else ""
        )
        return [
            {
                "train_id": train_idx + 1,
                "train_name": train_configs[train_idx].name,
                "direction": direction,
                "section_name": environment.section.name,
            }
            for environment in environments_by_train[train_idx]
        ]

    if share_lower_policy:
        # 共享策略不再受 12 个列车模板的质量重复次数影响。对每个方向，
        # 使用该方向的一套线路环境，并与 9 个质量锚点做笛卡尔积：
        # 2 directions * 19 sections * 9 masses = 342 combinations/cycle.
        representative_by_direction: dict[str, int] = {}
        for train_idx, direction in enumerate(direction_labels or []):
            representative_by_direction.setdefault(str(direction), train_idx)
        if not representative_by_direction:
            representative_by_direction[""] = 0
        flat_environments: list[LowerSectionEnvironment] = []
        flat_timetable_values: list[float] = []
        flat_tags: list[dict[str, Any]] = []
        for direction, representative_idx in representative_by_direction.items():
            base_environments = environments_by_train[representative_idx]
            base_timetable = initial_timetables[representative_idx]
            for mass_config in TRAIN_CONFIGS:
                mass_t = mass_config.mass_kg / 1000.0
                sampled_config = replace(
                    mass_config,
                    name=f"Shared {direction or 'line'} {mass_t:.1f}t",
                    dt_s=float(base_environments[0].dynamics.config.dt_s),
                )
                for section_idx, base_environment in enumerate(base_environments):
                    flat_environments.append(
                        LowerSectionEnvironment(
                            base_environment.section,
                            TrainDynamics(sampled_config),
                            speed_limit_penalty_ratio=base_environment.speed_limit_penalty_ratio,
                            arrival_tolerance_s=base_environment.arrival_tolerance_s,
                            direction_sign=base_environment.direction_sign,
                            section_index_normalized=base_environment.section_index_normalized,
                            schedule_shaping_gamma=(
                                args.lower_sac_gamma
                                if args.lower_algorithm == "sac"
                                else 1.0
                            ),
                        )
                    )
                    flat_timetable_values.append(float(base_timetable[section_idx]))
                    flat_tags.append(
                        {
                            "train_id": "",
                            "train_name": f"Mass anchor {mass_t:.1f}t",
                            "direction": direction,
                            "section_name": base_environment.section.name,
                            "mass_anchor_t": mass_t,
                        }
                    )
        flat_timetable = np.asarray(flat_timetable_values, dtype=np.float32)
        print(
            "Balanced shared-policy grid: "
            f"{len(representative_by_direction)} directions * "
            f"{len(environments_by_train[0])} sections * "
            f"{len(TRAIN_CONFIGS)} masses = {len(flat_environments)} combinations"
        )
        training_items = [
            (
                0,
                train_configs[0],
                flat_environments,
                flat_timetable,
                flat_tags,
                f"Shared {args.lower_algorithm.upper()} lower policy",
            )
        ]
    else:
        training_items = [
            (
                train_idx,
                config,
                environments,
                initial_timetables[train_idx],
                environment_tags_for_train(train_idx),
                config.name,
            )
            for train_idx, (config, environments) in enumerate(
                zip(train_configs, environments_by_train)
            )
        ]
    for item in training_items:
        (
            train_idx,
            config,
            environments,
            timetable_for_training,
            environment_tags,
            train_name,
        ) = item
        if args.lower_policy_mode == "reuse":
            print(f"Reusing lower {args.lower_algorithm.upper()} policy for {train_name}; lower training skipped.")
            continue
        if args.lower_episodes <= 0:
            print(f"Lower training skipped for {train_name}: lower_episodes=0")
            continue
        if args.lower_algorithm == "dqn":
            lower_history.extend(
                train_lower_agents(
                    environments,
                    lower_agents[train_idx],
                    timetable_for_training,
                    episodes=args.lower_episodes,
                    target_span_s=args.lower_target_span,
                    train_name=train_name,
                    random_mass_training=args.ddpg_random_mass_training,
                    mass_min_t=args.ddpg_mass_min_t,
                    mass_max_t=args.ddpg_mass_max_t,
                    anchor_mass_probability=args.ddpg_anchor_mass_probability,
                    boundary_mass_probability=args.ddpg_boundary_mass_probability,
                    log_interval=args.lower_log_interval,
                    environment_tags=environment_tags,
                    checkpoint_interval=args.lower_checkpoint_interval,
                    checkpoint_callback=autosave_lower_training,
                )
            )
        else:
            trainer = (
                train_lower_ddpg_agents
                if args.lower_algorithm == "ddpg"
                else train_lower_sac_agents
            )
            lower_history.extend(
                trainer(
                    environments,
                    lower_agents[train_idx],
                    timetable_for_training,
                    episodes=args.lower_episodes,
                    target_span_s=args.lower_target_span,
                    train_name=train_name,
                    random_mass_training=args.ddpg_random_mass_training,
                    mass_min_t=args.ddpg_mass_min_t,
                    mass_max_t=args.ddpg_mass_max_t,
                    anchor_mass_probability=args.ddpg_anchor_mass_probability,
                    boundary_mass_probability=args.ddpg_boundary_mass_probability,
                    log_interval=args.lower_log_interval,
                    pretrain_episodes=args.lower_bc_episodes,
                    plot_interval=args.lower_plot_interval,
                    output_dir=args.output_dir,
                    environment_tags=environment_tags,
                    alpha_auto_tune_start_episode=(
                        args.lower_sac_alpha_warmup_episodes
                        if args.lower_algorithm == "sac"
                        else 0
                    ),
                    checkpoint_interval=args.lower_checkpoint_interval,
                    checkpoint_callback=autosave_lower_training,
                )
            )
    return lower_agents, lower_history


def initial_station_schedule(
    initial_timetable: Sequence[float],
    base_departure_s: float,
    station_count: int,
    station_dwell_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    arrivals = np.full(station_count, np.nan, dtype=float)
    departures = np.full(station_count, np.nan, dtype=float)
    current_s = float(base_departure_s)
    arrivals[0] = current_s
    departures[0] = current_s
    for section_idx, runtime_s in enumerate(initial_timetable):
        current_s += float(runtime_s)
        station_idx = section_idx + 1
        arrivals[station_idx] = current_s
        if station_idx < station_count - 1:
            current_s += station_dwell_s
            departures[station_idx] = current_s
    return arrivals, departures


def online_station_schedule(
    step_rows: Sequence[dict[str, Any]],
    train_idx: int,
    base_departure_s: float,
    station_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows_by_section = {
        int(row["section_index"]): row
        for row in step_rows
        if int(row["train_index"]) == train_idx
    }
    arrivals = np.full(station_count, np.nan, dtype=float)
    departures = np.full(station_count, np.nan, dtype=float)
    dwells = np.full(station_count, np.nan, dtype=float)
    arrivals[0] = float(base_departure_s)
    if 0 in rows_by_section:
        departures[0] = float(rows_by_section[0]["departure_time_s"])
    for station_idx in range(1, station_count):
        previous_row = rows_by_section.get(station_idx - 1)
        if previous_row is not None:
            arrivals[station_idx] = float(previous_row["arrival_time_s"])
        current_row = rows_by_section.get(station_idx)
        if current_row is not None:
            departures[station_idx] = float(current_row["departure_time_s"])
        if np.isfinite(arrivals[station_idx]) and np.isfinite(departures[station_idx]):
            dwells[station_idx] = departures[station_idx] - arrivals[station_idx]
    return arrivals, departures, dwells


def export_online_timetable(
    step_rows: Sequence[dict[str, Any]],
    initial_timetables: np.ndarray,
    base_departures: np.ndarray,
    station_dwell_s: float,
    train_configs: Sequence[Any],
    direction_labels: Sequence[str],
    up_profile,
    down_profile,
    output_dir: Path,
) -> None:
    if not step_rows:
        return
    station_rows: list[dict[str, Any]] = []
    section_rows: list[dict[str, Any]] = []
    schedule_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]] = {}
    for train_idx, config in enumerate(train_configs):
        profile = up_profile if direction_labels[train_idx] == "up" else down_profile
        names = station_names(profile)
        initial_arrivals, initial_departures = initial_station_schedule(
            initial_timetables[train_idx],
            float(base_departures[train_idx]),
            len(names),
            station_dwell_s,
        )
        online_arrivals, online_departures, online_dwells = online_station_schedule(
            step_rows,
            train_idx,
            float(base_departures[train_idx]),
            len(names),
        )
        schedule_cache[train_idx] = (
            initial_arrivals,
            initial_departures,
            online_arrivals,
            online_departures,
            online_dwells,
            names,
        )
        for station_idx, station_name in enumerate(names):
            initial_arrival = initial_arrivals[station_idx]
            initial_departure = initial_departures[station_idx]
            online_arrival = online_arrivals[station_idx]
            online_departure = online_departures[station_idx]
            station_rows.append(
                {
                    "train_index": train_idx + 1,
                    "train_name": config.name,
                    "direction": direction_labels[train_idx],
                    "station_index": station_idx + 1,
                    "station_name": station_name,
                    "initial_arrival_time_s": initial_arrival,
                    "optimized_arrival_time_s": online_arrival,
                    "arrival_offset_s": (
                        online_arrival - initial_arrival
                        if np.isfinite(online_arrival) and np.isfinite(initial_arrival)
                        else ""
                    ),
                    "initial_departure_time_s": initial_departure if np.isfinite(initial_departure) else "",
                    "optimized_departure_time_s": online_departure if np.isfinite(online_departure) else "",
                    "departure_offset_s": (
                        online_departure - initial_departure
                        if np.isfinite(online_departure) and np.isfinite(initial_departure)
                        else ""
                    ),
                    "initial_dwell_s": (
                        station_dwell_s
                        if 0 < station_idx < len(names) - 1
                        else 0.0
                    ),
                    "optimized_dwell_s": online_dwells[station_idx] if np.isfinite(online_dwells[station_idx]) else "",
                }
            )
    for row in step_rows:
        train_idx = int(row["train_index"])
        section_idx = int(row["section_index"])
        section_rows.append(
            {
                "train_index": train_idx + 1,
                "train_name": train_configs[train_idx].name,
                "direction": direction_labels[train_idx],
                "section_index": section_idx + 1,
                "section_name": row["section_name"],
                "physical_section_name": row["physical_section_name"],
                "initial_runtime_s": float(initial_timetables[train_idx, section_idx]),
                "requested_target_runtime_s": row.get(
                    "requested_target_runtime_s",
                    row["target_runtime_s"],
                ),
                "target_runtime_s": row["target_runtime_s"],
                "runtime_adjustment_s": row["runtime_adjustment_s"],
                "departure_time_s": row["departure_time_s"],
                "arrival_time_s": row["arrival_time_s"],
                "dwell_adjustment_s": row["dwell_adjustment_s"],
                "traction_kwh": row["traction_kwh"],
                "regenerative_kwh": row["regenerative_kwh"],
                "feasible": row["feasible"],
                "safety_filtered": row.get("safety_filtered", False),
            }
        )
    write_csv(output_dir / "online_timetable_schedule.csv", station_rows)
    write_csv(output_dir / "online_timetable_section_adjustments.csv", section_rows)
    plot_online_timetable(
        schedule_cache,
        direction_labels,
        up_profile,
        down_profile,
        output_dir,
    )


def plot_online_timetable(
    schedule_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]],
    direction_labels: Sequence[str],
    up_profile,
    down_profile,
    output_dir: Path,
) -> None:
    up_positions = [up_profile.sections[0].display_start_m] + [
        section.display_end_m for section in up_profile.sections
    ]
    down_positions = [down_profile.sections[0].display_start_m] + [
        section.display_end_m for section in down_profile.sections
    ]
    fig, axes = plt.subplots(2, 1, figsize=(16, 11), sharex=True)
    direction_titles = {"up": "Up direction", "down": "Down direction"}
    colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(schedule_cache))))
    for axis, direction in zip(axes, ("up", "down")):
        positions = up_positions if direction == "up" else down_positions
        station_labels = None
        for train_idx, color in zip(schedule_cache, colors):
            if direction_labels[train_idx] != direction:
                continue
            (
                initial_arrivals,
                initial_departures,
                online_arrivals,
                online_departures,
                _online_dwells,
                station_labels,
            ) = schedule_cache[train_idx]
            initial_times = []
            initial_positions = []
            online_times = []
            online_positions = []
            for station_idx, position_m in enumerate(positions):
                if np.isfinite(initial_arrivals[station_idx]):
                    initial_times.append(initial_arrivals[station_idx] / 60.0)
                    initial_positions.append(position_m / 1000.0)
                if (
                    station_idx < len(positions) - 1
                    and np.isfinite(initial_departures[station_idx])
                    and initial_departures[station_idx] != initial_arrivals[station_idx]
                ):
                    initial_times.append(initial_departures[station_idx] / 60.0)
                    initial_positions.append(position_m / 1000.0)
                if np.isfinite(online_arrivals[station_idx]):
                    online_times.append(online_arrivals[station_idx] / 60.0)
                    online_positions.append(position_m / 1000.0)
                if (
                    station_idx < len(positions) - 1
                    and np.isfinite(online_departures[station_idx])
                    and online_departures[station_idx] != online_arrivals[station_idx]
                ):
                    online_times.append(online_departures[station_idx] / 60.0)
                    online_positions.append(position_m / 1000.0)
            axis.plot(
                initial_times,
                initial_positions,
                color=color,
                linestyle="--",
                alpha=0.35,
                linewidth=1.0,
            )
            axis.plot(
                online_times,
                online_positions,
                color=color,
                linewidth=1.4,
                alpha=0.95,
                label=f"T{train_idx + 1}",
            )
        axis.set_title(f"{direction_titles[direction]}: initial dashed vs optimized solid")
        axis.set_ylabel("Physical mileage (km)")
        axis.set_yticks(np.asarray(positions) / 1000.0)
        if station_labels is not None:
            axis.set_yticklabels(station_labels)
        axis.grid(alpha=0.25)
        axis.legend(ncol=6, fontsize=8, loc="best")
    axes[-1].set_xlabel("Time from first departure (min)")
    fig.suptitle("Online MAPPO timetable: station departure and arrival adjustments")
    fig.tight_layout()
    fig.savefig(output_dir / "online_timetable_optimization.png", dpi=180)
    plt.close(fig)


def export_online_outputs(
    up_profile,
    down_profile,
    train_configs: Sequence[Any],
    direction_labels: Sequence[str],
    results_by_train,
    output_dir: Path,
) -> None:
    summary_rows: list[dict[str, Any]] = []
    direction_counts = {"up": 0, "down": 0}
    for train_idx, results in enumerate(results_by_train):
        direction = direction_labels[train_idx]
        direction_counts[direction] += 1
        profile = up_profile if direction == "up" else down_profile
        stem = f"{direction}_train_{direction_counts[direction]:02d}_online_operation"
        plot_train_operation(profile, results, output_dir, stem)
        plot_section_operation_overview(
            profile,
            results,
            output_dir,
            f"{direction}_train_{direction_counts[direction]:02d}_online",
        )
        for row in lower_result_summary_rows(results, train_configs[train_idx].name):
            row["direction"] = direction
            summary_rows.append(row)
    write_csv(output_dir / "online_train_operation_summary.csv", summary_rows)
    write_online_operation_rows(
        output_dir / "online_lower_rollout_trajectories.csv",
        results_by_train,
        [config.name for config in train_configs],
    )


def main(argv: Optional[Sequence[str]] = None) -> Path:
    args = build_parser().parse_args(argv)
    if args.device is not None:
        args.lower_device = args.device
        args.upper_device = args.device
    if args.lower_arrival_tolerance < 0.0:
        raise ValueError("--lower-arrival-tolerance must be >= 0")
    if not 0.0 < args.simulation_dt_s <= 0.5:
        raise ValueError("--simulation-dt-s must be in (0, 0.5]")
    if args.lower_checkpoint_interval < 0:
        raise ValueError("--lower-checkpoint-interval must be >= 0")
    if args.runtime_adjustment_span > 0.5:
        raise ValueError("--runtime-adjustment-span should be <= 0.5 s")
    if args.lower_backend == "cvxpy":
        if args.experiment_stage != "hierarchical":
            raise ValueError(
                "--lower-backend cvxpy has no lower-policy training stage; "
                "use --experiment-stage hierarchical"
            )
        if args.lower_policy_mode != "fresh" or args.lower_policy_checkpoint is not None:
            raise ValueError(
                "Lower policy checkpoints do not apply to --lower-backend cvxpy"
            )
        if args.convex_spatial_segments < 2:
            raise ValueError("--convex-spatial-segments must be at least 2")
        if not 0.0 < args.convex_speed_limit_ratio <= 1.0:
            raise ValueError("--convex-speed-limit-ratio must be in (0, 1]")
        if args.convex_time_limit_s is not None and args.convex_time_limit_s <= 0.0:
            raise ValueError("--convex-time-limit-s must be positive")
        if args.convex_runtime_search_step_s <= 0.0:
            raise ValueError("--convex-runtime-search-step-s must be positive")
        if args.convex_runtime_search_max_extra_s < 0.0:
            raise ValueError("--convex-runtime-search-max-extra-s must be nonnegative")
    configure_run_outputs(args)
    seed_everything(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    (
        up_profile,
        down_profile,
        train_configs,
        environments_by_train,
        initial_timetables,
        base_departures,
        direction_groups,
        physical_sections,
        direction_labels,
    ) = build_bidirectional_model(args)
    for train_environments in environments_by_train:
        for environment in train_environments:
            environment.arrival_tolerance_s = args.lower_arrival_tolerance
    write_run_metadata(args, train_configs, initial_timetables, direction_labels)
    write_csv(
        args.output_dir / "down_line_profile.csv",
        down_line_profile_rows(up_profile, down_profile),
    )
    write_csv(
        args.output_dir / "train_parameters.csv",
        [
            {
                **asdict(config),
                "direction": direction_labels[idx],
            }
            for idx, config in enumerate(train_configs)
        ],
    )
    lower_model_solver = None
    if args.lower_backend == "cvxpy":
        lower_agents = []
        lower_history = []
        lower_model_solver = CvxpyLowerSolver(
            CvxpyLowerOptions(
                spatial_segments=args.convex_spatial_segments,
                speed_limit_ratio=args.convex_speed_limit_ratio,
                arrival_tolerance_s=args.lower_arrival_tolerance,
                output_flag=False,
                time_limit_s=args.convex_time_limit_s,
                solver=args.convex_solver,
                # Near the minimum feasible runtime SCS may return a highly
                # inaccurate trajectory.  Clarabel gives a definite
                # infeasible status, allowing the runtime calibrator below to
                # search for the nearest physically valid value.
                fallback_solver=None,
            )
        )
        print(
            "Using licence-free model-based lower layer: "
            f"CVXPY/{args.convex_solver}, "
            f"segments={args.convex_spatial_segments}, "
            f"speed-limit ratio={args.convex_speed_limit_ratio:.3f}"
        )
    else:
        lower_agents, lower_history = build_lower_agents(
            args,
            train_configs,
            environments_by_train,
            initial_timetables,
            direction_labels,
        )
    if args.experiment_stage == "lower":
        save_lower_only_checkpoint(
            args.checkpoint,
            lower_agents,
            train_configs,
            args.lower_algorithm,
        )
        plot_training_process(lower_history, [], args.output_dir)
        feasible_values = [
            1.0
            if str(row.get("feasible", "")).strip().lower() in {"1", "true", "yes"}
            else 0.0
            for row in lower_history
            if str(row.get("phase", "")) != "behavior_clone_pretrain"
        ]
        timetable_errors = [
            abs(float(row["timetable_error_s"]))
            for row in lower_history
            if str(row.get("phase", "")) != "behavior_clone_pretrain"
            and row.get("timetable_error_s") not in (None, "")
        ]
        write_csv(
            args.output_dir / "lower_experiment_summary.csv",
            [
                {
                    "experiment_stage": "lower",
                    "lower_algorithm": args.lower_algorithm,
                    "training_records": len(feasible_values),
                    "feasible_rate": (
                        float(np.mean(feasible_values)) if feasible_values else ""
                    ),
                    "mean_abs_timetable_error_s": (
                        float(np.mean(timetable_errors)) if timetable_errors else ""
                    ),
                    "arrival_tolerance_s": args.lower_arrival_tolerance,
                    "target_time_random_span_s": args.lower_target_span,
                }
            ],
        )
        print(f"Saved lower-only experiment results to: {args.output_dir.resolve()}")
        return args.output_dir

    online_environment = OnlineHierarchicalTimetableEnvironment(
        initial_timetables_by_train=initial_timetables,
        environments_by_train=environments_by_train,
        lower_agents=lower_agents,
        base_departures_s=base_departures,
        direction_groups=direction_groups,
        physical_section_names_by_train=physical_sections,
        supply_zone_section_groups=REAL_SUPPLY_ZONE_SECTION_GROUPS,
        supply_zone_names=REAL_SUPPLY_ZONE_NAMES,
        station_dwell_s=args.station_dwell_s,
        minimum_headway_s=args.minimum_headway,
        runtime_adjustment_span_s=args.runtime_adjustment_span,
        dwell_adjustment_span_s=args.dwell_adjustment_span,
        traction_energy_weight=args.traction_energy_weight,
        regenerative_match_weight=args.regenerative_match_weight,
        rollout_cache_quantum_s=args.rollout_cache_quantum,
        lower_model_solver=lower_model_solver,
        model_runtime_search_step_s=args.convex_runtime_search_step_s,
        model_runtime_search_max_extra_s=args.convex_runtime_search_max_extra_s,
    )
    if args.lower_backend == "cvxpy":
        args.requested_initial_timetables_s = initial_timetables.tolist()
        initial_timetables = online_environment.initial_timetables.copy()
        write_run_metadata(args, train_configs, initial_timetables, direction_labels)
        write_csv(
            args.output_dir / "convex_runtime_adjustments.csv",
            online_environment.model_runtime_adjustments,
        )
    upper_agent = IndependentActorMAPPOAgent(
        train_count=online_environment.train_count,
        observation_dim=online_environment.local_observation_dim,
        action_dim=online_environment.action_dim,
        central_state_dim=online_environment.central_state_dim,
        active_agent_mask=online_environment.active_agent_mask,
        device_name=args.upper_device,
    )
    if args.upper_policy_mode != "fresh":
        if args.upper_policy_checkpoint is None:
            raise ValueError(
                "--upper-policy-checkpoint is required when "
                "--upper-policy-mode is continue/reuse"
            )
        load_upper_controller_from_checkpoint(
            args.upper_policy_checkpoint,
            upper_agent,
            load_optimizers=args.upper_policy_mode == "continue",
        )
    result, upper_history, best_results_by_train, best_step_rows = train_online_mappo(
        online_environment,
        upper_agent,
        episodes=args.upper_episodes,
        progress_interval=args.progress_interval,
    )
    save_multi_train_checkpoint(
        args.checkpoint,
        lower_agents,
        upper_agent,
        result,
        [asdict(config) for config in train_configs],
        asdict(DRIVING_SMOOTHNESS),
        lower_algorithm=(
            f"cvxpy_{args.convex_solver.lower()}"
            if args.lower_backend == "cvxpy"
            else args.lower_algorithm
        ),
    )
    write_csv(args.output_dir / "online_station_decision_steps.csv", best_step_rows)
    export_online_timetable(
        best_step_rows,
        initial_timetables,
        base_departures,
        args.station_dwell_s,
        train_configs,
        direction_labels,
        up_profile,
        down_profile,
        args.output_dir,
    )
    write_csv(
        args.output_dir / "online_experiment_summary.csv",
        [
            {
                "up_train_count": args.up_trains,
                "down_train_count": args.down_trains,
                "station_dwell_s": args.station_dwell_s,
                "initial_headway_s_each_direction": args.initial_headway,
                "minimum_headway_s": args.minimum_headway,
                "lower_algorithm": (
                    f"cvxpy_{args.convex_solver.lower()}"
                    if args.lower_backend == "cvxpy"
                    else args.lower_algorithm
                ),
                "lower_backend": args.lower_backend,
                "convex_solver": (
                    args.convex_solver if args.lower_backend == "cvxpy" else ""
                ),
                "convex_spatial_segments": (
                    args.convex_spatial_segments
                    if args.lower_backend == "cvxpy"
                    else ""
                ),
                "convex_speed_limit_ratio": (
                    args.convex_speed_limit_ratio
                    if args.lower_backend == "cvxpy"
                    else ""
                ),
                "traction_kwh": result.traction_kwh,
                "regenerative_kwh": result.regenerative_kwh,
                "matched_regenerative_kwh": result.matched_regenerative_kwh,
                "regenerative_utilization": result.regenerative_utilization,
                "normalized_weighted_objective": result.objective,
                "runtime_adjustment_span_s": args.runtime_adjustment_span,
                "dwell_adjustment_span_s": args.dwell_adjustment_span,
                "rollout_cache_quantum_s": args.rollout_cache_quantum,
                "rollout_cache_hits": online_environment.rollout_cache.hits,
                "rollout_cache_misses": online_environment.rollout_cache.misses,
                "rollout_cache_size": len(online_environment.rollout_cache.items),
                "convex_baseline_runtime_adjustment_count": len(
                    online_environment.model_runtime_adjustments
                ),
                "convex_baseline_runtime_adjustment_s": float(
                    sum(
                        row["adjustment_s"]
                        for row in online_environment.model_runtime_adjustments
                    )
                ),
            }
        ],
    )
    export_online_outputs(
        up_profile,
        down_profile,
        train_configs,
        direction_labels,
        best_results_by_train,
        args.output_dir,
    )
    plot_training_process(lower_history, upper_history, args.output_dir)
    print(f"Saved online hierarchical experiment results to: {args.output_dir.resolve()}")
    return args.output_dir


if __name__ == "__main__":
    main()
