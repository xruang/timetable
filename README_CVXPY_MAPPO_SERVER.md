# CVXPY 下层 + MAPPO 上层服务器训练包

该版本将原在线分层框架中的学习型下层替换为无商业许可证的凸优化下层：

- 上层：独立 Actor、集中式 Critic 的 MAPPO；
- 下层：CVXPY 建模，默认由开源 Clarabel 求解；
- 上层输入：列车载重对应的动力参数、区间和目标运行时间；
- 下层输出：实际复算时间、牵引/再生/净能耗、速度轨迹、优化力轨迹和实际复算力轨迹；
- 约束：限速折减 0.8、对称恒功率牵引/制动、牵引/制动力、加减速度、坡度与 Davis 阻力；
- 曲线阻力不进入凸优化模型；
- 相同列车参数、区间和离散目标时间使用内存缓存，不重复求解。
- 若 0.8 限速使原始基准运行时间物理不可行，启动时按 1 秒步长向上搜索最近的可行时间，并将调整记录到 `convex_runtime_adjustments.csv`。例如当前上行 `CP-JK` 的 251 秒会自动校准为 255 秒。

## 1. 安装

建议 Python 3.9 或兼容版本。先按照服务器 CUDA 版本安装 PyTorch，再安装其余依赖：

```bash
python -m pip install -r requirements.txt
```

无需安装 `gurobipy`，也无需 Gurobi 许可证。

验证开源下层：

```bash
python server/check_cvxpy_lower.py
```

## 2. 服务器训练

```bash
bash run_cvxpy_mappo_server.sh
```

等价的完整命令为：

```bash
python -m timetable.timetable_0601_bidirectional_online \
  --lower-backend cvxpy \
  --convex-solver CLARABEL \
  --convex-spatial-segments 600 \
  --convex-speed-limit-ratio 0.8 \
  --lower-arrival-tolerance 0.5 \
  --rollout-cache-quantum 0.1 \
  --runtime-adjustment-span 0.5 \
  --dwell-adjustment-span 10 \
  --up-trains 6 \
  --down-trains 6 \
  --upper-episodes 500 \
  --upper-device auto \
  --progress-interval 10 \
  --run-name cvxpy_mappo_6x6 \
  --output-dir timetable/results_cvxpy_mappo
```

`--lower-episodes`、`--lower-policy-mode` 和下层策略 checkpoint 对 CVXPY 后端无效；该后端没有下层神经网络训练阶段。

## 3. 快速冒烟测试

正式训练前可先运行：

```bash
bash smoke_test_cvxpy_mappo.sh
```

冒烟测试使用双向各 1 列车、40 个空间网格和 1 个 MAPPO 回合，只用于检查数据、接口、求解器、checkpoint 和结果导出链路。正式实验请使用 600 个空间网格。

## 4. 计算量说明

CVXPY 下层不是神经网络推理。第一次遇到某个“列车参数 + 区间 + 目标时间”组合时需要求解，之后从缓存直接读取。`--rollout-cache-quantum 0.1` 表示目标时间按 0.1 秒离散；增大到 0.5 秒可显著减少首次求解数量，但会降低上层运行时间动作的分辨率。

训练结果目录会包含：

- `hierarchical_mappo_bidirectional_online_checkpoint.pt`：上层 MAPPO 权重和优化器状态；
- `online_lower_rollout_trajectories.csv`：所有最优速度/力轨迹；
- `online_train_operation_summary.csv`：实际复算能耗和时间；
- `online_station_decision_steps.csv`：上下层每次交互；
- `online_experiment_summary.csv`：目标值、再生利用率和缓存统计；
- `run_metadata.json`：求解器、网格、限速比例和完整运行参数。

继续训练上层时使用：

```bash
python -m timetable.timetable_0601_bidirectional_online \
  --lower-backend cvxpy \
  --convex-spatial-segments 600 \
  --convex-speed-limit-ratio 0.8 \
  --upper-policy-mode continue \
  --upper-policy-checkpoint /path/to/hierarchical_mappo_bidirectional_online_checkpoint.pt \
  --upper-episodes 500 \
  --runtime-adjustment-span 0.5 \
  --run-name cvxpy_mappo_continue \
  --output-dir timetable/results_cvxpy_mappo
```

缓存目前属于单次进程内缓存；重新启动继续训练时会重新计算首次出现的下层组合。

当 Clarabel 明确判定某个临界运行时间不可行时，正式训练不会再调用 SCS 接受低精度轨迹，而是进入上述运行时间校准流程。默认最多向上搜索 30 秒，可通过 `--convex-runtime-search-max-extra-s` 修改。
