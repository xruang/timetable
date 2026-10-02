# 广州地铁 21 号线 6+6 双向协同优化实验

当前工作区只保留上下行 6+6 列车相关代码版本，以及它们依赖的公共模块。

2026-09-17下层动作一致性及终点保护修复见[修复说明](LOWER_FIXES_20260917.md)。新实验使用 `lower_sac_action_consistent_dt_0_2`，必须fresh训练，不能继续加载旧连续控制critic。

## 项目内线路数据

线路 Excel 已保存在项目内，主程序默认直接读取以下文件，不再依赖桌面目录：

```text
timetable/data/up/slope.xlsx
timetable/data/up/curvature.xlsx
timetable/data/down/slope.xlsx
timetable/data/down/curvature.xlsx
```

如需临时使用其他线路文件，仍可通过 `--line-dir`、`--down-slope` 和
`--down-curvature` 覆盖默认路径。

## 三个容易混淆的时间参数

| 参数 | 属于哪一层 | 含义 | 设为 0 的效果 |
|---|---|---|---|
| `--lower-target-span` | 下层训练 | 训练样本的目标站间时间在真实基准值两侧随机变化的范围，不是上层动作 | 下层只学习 config 中的真实站间时间 |
| `--runtime-adjustment-span` | 上层 | 上层对站间运行时间的最大调整幅度 | 上层不能调整站间运行时间 |
| `--dwell-adjustment-span` | 上层 | 上层对停站时间的最大调整幅度，调整后会影响后续到发时刻 | 上层不能调整停站时间 |

例如，若采用“站间时间基本固定、主要调整停站时间”的建模假设，可使用：

```powershell
--lower-target-span 0 --lower-arrival-tolerance 0.5 --runtime-adjustment-span 0 --dwell-adjustment-span 10
```

其中 `1` 和 `10` 只是便于理解的示例，应按真实数据允许的波动范围确定。

## 保留的实验入口

### 1. 当前主版本：在线上下层协同

```text
timetable/timetable_0601_bidirectional_online.py
```

用途：

- 上行 6 列车 + 下行 6 列车。
- 上层按事件步调整时刻表。
- 下层在每个站间被调用，生成该站间速度轨迹和牵引/制动力曲线。
- 下层算法支持 `DQN`、`DDPG`、`SAC`。
- 支持共享下层策略、随机载重训练、读取下层预训练权重。
- 使用真实供电分区计算再生制动匹配。

常用运行指令示例：

```powershell
python -m timetable.timetable_0601_bidirectional_online --lower-algorithm sac --share-lower-continuous --lower-policy-mode fresh --upper-policy-mode fresh --lower-bc-episodes 1000 --lower-episodes 5000 --upper-episodes 500 --lower-log-interval 10 --lower-plot-interval 100 --progress-interval 10 --runtime-adjustment-span 0.5 --ddpg-random-mass-training --output-dir timetable/results_bidirectional_online/shared_sac_bc1000_finetune5000_upper500
```

DQN 对比：

```powershell
python -m timetable.timetable_0601_bidirectional_online --lower-algorithm dqn --share-lower-dqn --lower-policy-mode fresh --upper-policy-mode fresh --lower-episodes 5000 --upper-episodes 500 --lower-log-interval 10 --progress-interval 10 --runtime-adjustment-span 0.5 --ddpg-random-mass-training --output-dir timetable/results_bidirectional_online/shared_dqn_lower5000_upper500
```

DDPG 对比：

```powershell
python -m timetable.timetable_0601_bidirectional_online --lower-algorithm ddpg --share-lower-continuous --lower-policy-mode fresh --upper-policy-mode fresh --lower-bc-episodes 1000 --lower-episodes 5000 --upper-episodes 500 --lower-log-interval 10 --lower-plot-interval 100 --progress-interval 10 --runtime-adjustment-span 0.5 --ddpg-random-mass-training --output-dir timetable/results_bidirectional_online/shared_ddpg_bc1000_finetune5000_upper500
```

### 2. 双向模型公共模块

```text
timetable/bidirectional_model.py
```

用途：

- 构建上下行线路、列车和下层站间环境。
- 生成双向真实基准时刻表、发车间隔和方向分组。
- 提供在线主程序使用的站名与下行线路导出函数。
- 只保存公共模型逻辑，不再包含旧版独立训练入口。

## 保留的公共模块

```text
timetable/config.py
timetable/bidirectional_model.py
timetable/line_profile.py
timetable/train_model.py
timetable/lower_control.py
timetable/lower_continuous.py
timetable/mappo.py
timetable/online_hierarchical.py
timetable/visualization.py
timetable/redraw_operation_plots.py
```

## 已移除的旧版本

以下非 6+6 双向版本已从源码中移除：

```text
timetable/timetable_0601.py
timetable/timetable_0601_backup.py
timetable/timetable_0601_disturbance.py
timetable/timetable_0601_power_zone.py
timetable/train_real_strategy.py
timetable/compare_real_strategy.py
timetable/real_strategy.py
timetable/calibrate_real_dynamics.py
timetable/timetable_0601_bidirectional.py
PINN_demo.py
```

说明：历史结果目录和旧 checkpoint 文件未自动删除，以免误删已有实验结果。
## 统一实验入口

以后不需要重复编写很长的 PowerShell 命令。实验参数统一保存在 `experiments/*.json`，统一入口为：

```powershell
python run_experiment.py list
python run_experiment.py run lower_sac_smoke
python run_experiment.py run lower_sac_baseline
```

也可以在 VS Code 的“运行和调试”中直接选择以下项目：

- `Experiment: SAC smoke (2 episodes)`
- `Experiment: lower DQN baseline`
- `Experiment: lower DDPG baseline`
- `Experiment: lower SAC baseline`

所有新结果固定保存在：

```text
results/<实验名>/<运行时间>_seed<随机种子>/
```

每次运行会自动保存：

- `experiment_config.json`：本次实际采用的完整实验配置；
- `run_metadata.json`：所有参数、列车参数和 Python 源码 SHA-256；
- `run_status.json`：运行成功或失败状态；
- `lower_policy_checkpoint.pt`：仅下层实验的策略及优化器状态；
- 训练历史 CSV、分区统计和收敛图。

当前三组下层对比配置采用同一语义：

- `lower_target_span = 0`：按 config 中真实站间时间训练，不随机改变目标；
- `lower_arrival_tolerance = 0.5`：下层允许 ±0.5 s 到达误差；
- `runtime_adjustment_span = 0`：上层不调整站间运行时间；
- `dwell_adjustment_span = 10`：以后训练上层时，只调整停站时间。

临时改一个参数可使用 `--set`，且改动只作用于本次运行并被记录：

```powershell
python run_experiment.py run lower_sac_smoke --set lower_episodes=20 --set seed=8
```

## Linux 服务器训练

服务器打包、CUDA 环境检查、后台训练、自动 checkpoint 和续训方式见：

```text
SERVER_TRAINING.md
```

## 单情景 SAC 诊断接口

原多情景入口仍为 `run_experiment.py`。单情景接口复用同一环境、奖励、安全层和SAC更新，不训练上层，也不加载旧权重。

```bash
source .venv/bin/activate
python run_single_scenario.py --list-sections
python run_single_scenario.py --direction up --section HC-DGNL --mass-t 255 --alpha 0.02 --episodes 1000
```

默认目标使用所选站间的真实运营时间；可用 `--target-time-s 87` 显式指定固定时间。默认步长0.2秒、gamma0.9998、16维观测、准点容差0.5秒、限速比例1.0、无BC。质量和目标不随机变化，alpha始终固定；训练仍采样随机动作，每100回合用策略均值评估并保存checkpoint。`--dry-run`只检查情景；`--eval-interval`改变评估保存间隔；`--device cpu`指定CPU。

结果独立保存到 `results/lower_sac_single_scenario/<时间>_seed7/`，包含配置及源码哈希、训练CSV、确定性评估CSV、轨迹图、checkpoint和运行状态。评估记录严格准点、≤5秒成功、失败原因、初始Q、实际折扣回报以及终止样本TD误差。失败的elapsed不是到站时间；确定性回报和SAC soft Q不同，应结合终止TD误差判断价值偏差。

服务器后台运行：

```bash
mkdir -p server_logs
nohup python -u run_single_scenario.py --direction up --section HC-DGNL --mass-t 255 --alpha 0.02 --episodes 1000 > server_logs/single_scenario.log 2>&1 &
tail -f server_logs/single_scenario.log
```

重复运行会创建新的结果目录。后台日志示例使用固定文件名，下一次运行前应更换文件名以保留已有日志。

## SAC 价值学习诊断对照

```bash
bash server/run_sac_diagnostics.sh
```

依次运行两个1000回合的全新单情景实验：A组保留均匀回放，B组仅将终止转移的批次采样占比设为25%，终止样本不足时从普通样本补齐且不重复采样。其他参数相同，初始种子相同；采样改变后两组随机序列和轨迹会自然不同。此对照改变了训练样本权重，不做重要性校正，不是最终算法方案；一个种子只用于初步诊断。

独立运行：`python run_sac_diagnostics.py --terminal-fraction 0` 或 `python run_sac_diagnostics.py --terminal-fraction 0.25`。支持单情景入口的方向、站间、质量、目标、设备等参数。输出分别位于 `results/sac_diagnostics/uniform/` 和 `results/sac_diagnostics/terminal_0.25/`。

`update_diagnostics.csv`记录实际更新批次的普通/终止奖励与Q1/Q2/target范围、TD误差、终止样本占比及裁剪前梯度范数；`deterministic_evaluation.csv`包含固定评估和整个终止经验池的两个critic误差；`transition_audit.jsonl`保留前三条及每次终止转移；`fixed_state_action_scan.csv`记录固定观测上不同提出动作的Q值，仅是价值探测，不代表实测动作回报。每个训练转移和更新均检查数据一致性、有限值和终止目标，无需额外开启开关。断言失败会停止实验，错误写入运行状态。
