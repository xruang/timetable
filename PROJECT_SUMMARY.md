# 轨道交通列车节能控制与时刻表协同优化项目总结

本文档记录当前项目的主要建模内容、代码结构、上下层强化学习框架、列车与线路参数、训练方式、输出文件以及常用运行指令。当前项目根目录为：

```text
E:\时刻表\SAC
```

## 1. 项目目标

本项目面向广州地铁 21 号线站间运行数据，构建列车节能驾驶与多车时刻表协同优化实验平台。

核心目标包括：

- 根据坡度、曲线半径、限速等线路条件建立列车动力学模型。
- 下层智能体生成站间速度轨迹和牵引/制动力曲线。
- 上层智能体调整多列车时刻表，优化系统能耗与再生制动利用。
- 支持上行/下行双向 6+6 列车运行。
- 支持不同列车质量、牵引力、功率和戴维斯阻力参数。
- 支持共享 DQN / SAC / DDPG 下层策略，用于多载重泛化实验。
- 输出速度轨迹、牵引力曲线、时刻表优化过程、训练奖励曲线、再生制动利用结果等可视化文件。

## 2. 主要代码文件

```text
timetable/config.py
```

保存全局线路站间名、初始运行时间、真实供电分区、默认站停时间等。

```text
timetable/line_profile.py
```

读取并构建线路条件，包括坡度、曲线半径、限速、上下行线路映射。

```text
timetable/train_model.py
```

列车动力学与列车配置。包括质量、最大牵引力、最大牵引功率、最大制动力、戴维斯阻力参数、坡道阻力、曲线阻力、牵引功率约束等。

```text
timetable/lower_control.py
```

下层离散控制环境与 DQN。负责站间仿真、离散动作、经验池、奖励计算、DQN 训练、响应结果生成。

```text
timetable/lower_continuous.py
```

下层连续控制算法，包括 SAC、DDPG、BC 预训练、RL 微调、训练过程速度轨迹快照。

```text
timetable/mappo.py
```

传统响应表式 MAPPO 环境、上层多智能体 PPO、checkpoint 保存。

```text
timetable/online_hierarchical.py
```

在线层级协同环境。上层在事件步调用下层策略生成当前站间速度轨迹，再计算系统能耗和再生制动匹配。

```text
timetable/timetable_0601_bidirectional_online.py
```

当前主要实验入口：双向 6+6 列车、在线上层 MAPPO、下层 DQN/SAC/DDPG 对比、共享下层策略、随机载重训练。

```text
timetable/visualization.py
```

所有 CSV 和图像输出，包括全线速度轨迹、站间速度轨迹、牵引力曲线、训练奖励曲线、时刻表图、再生制动匹配图等。

```text
timetable/redraw_operation_plots.py
```

用于已有结果的单独重画图。

## 3. 线路与供电分区

全线共有 19 个站间：

```text
THP-TD
TD-HC
HC-DGNL
DGNL-THSC
THSC-SZL
SZL-SC
SC-SY
SY-SX
SX-CP
CP-JK
JK-ZLX
ZLX-ZL
ZL-ZX
ZX-KB
KB-FG
FG-ZC
ZC-ST
ST-ZG
ZG-ZCP
```

初始站间运行时间为：

```text
[184, 131, 87, 121, 185, 122, 142, 148, 160, 251, 260, 120, 130, 164, 154, 176, 221, 236, 144] s
```

默认站停时间：

```text
60 s
```

真实供电分区：

| 分区 | 站间 |
|---|---|
| THP-DGNL | THP-TD, TD-HC, HC-DGNL |
| DGNL-SY | DGNL-THSC, THSC-SZL, SZL-SC, SC-SY |
| SY-ZL | SY-SX, SX-CP, CP-JK, JK-ZLX, ZLX-ZL |
| ZL-ST | ZL-ZX, ZX-KB, KB-FG, FG-ZC, ZC-ST |
| ST-ZCP | ST-ZG, ZG-ZCP |

上行线路数据默认读取：

```text
timetable/data/up/slope.xlsx
timetable/data/up/curvature.xlsx
```

下行线路数据默认读取：

```text
timetable/data/down/slope.xlsx
timetable/data/down/curvature.xlsx
```

说明：

- `Change mileage (m)` 表示坡度改变点或站间开始坐标。
- `Gradient` 表示坡度。
- `Slope length (m)` 表示该坡度段长度。
- 曲线表中起始里程、终点里程和曲线半径用于曲线阻力和限速继承。
- 上下行公里标统一，但下行坡度方向反向处理。

## 4. 列车动力学模型

列车纵向动力学为：

```text
a = (F - W) / m
```

其中：

```text
W = W_davis + W_gradient + W_curve
```

戴维斯阻力：

```text
W_davis = a + b * v_kmh + c * v_kmh^2
```

当前戴维斯阻力参数由动车质量和拖车质量确定：

```text
a = (1.65 * 动车质量 + 0.78 * 拖车质量) * 9.8
b = (0.0247 * 动车质量 + 0.0028 * 拖车质量) * 9.8
c = 0.6566
```

质量单位为吨。

坡道阻力：

```text
W_gradient = m * g * gradient / 1000
```

曲线阻力：

```text
W_curve = m * g * (600 / R) / 1000
```

最大牵引力受力和功率共同约束：

```text
F_traction_max(v) = min(F_max, P_max / v)
```

能量计算：

```text
E_traction = max(F * ds, 0) / (eta_t * 3.6e6)
E_regen    = max(-F * ds, 0) * eta_r / 3.6e6
```

默认牵引效率：

```text
0.90
```

默认再生制动效率：

```text
0.60
```

动力学仿真步长：

```text
0.5 s
```

## 5. 列车质量与牵引约束

当前提供 9 组质量-牵引力-功率锚点：

| 总质量/t | 最大牵引力/kN | 最大牵引功率/kW |
|---:|---:|---:|
| 235 | 260 | 2770 |
| 240 | 265 | 2850 |
| 245 | 270 | 2900 |
| 250 | 275 | 2970 |
| 255 | 280 | 3020 |
| 260 | 285 | 3070 |
| 265 | 290 | 3140 |
| 270 | 295 | 3180 |
| 275 | 300 | 3230 |

若只给总质量，动车质量默认按 70% 处理，拖车质量为剩余 30%。

中间质量使用线性插值获取最大牵引力和最大牵引功率。超出锚点范围时，当前采用保守裁剪到最近锚点。

12 列车默认从 9 组配置中循环选取。

## 6. 下层驾驶策略

下层负责在单个站间内生成速度轨迹和牵引/制动力曲线。

输入包括：

- 当前站间位置与速度。
- 目标运行时间。
- 剩余距离与时间。
- 当前限速与前方限速。
- 坡度、曲线半径。
- 动车质量、拖车质量。
- 当前工况和上一时刻控制信息。

输出包括：

- 实际运行时间。
- 速度轨迹。
- 牵引/制动力曲线。
- 牵引能耗。
- 再生制动能量。
- 到站误差。
- 是否可行。
- jerk、牵引力波动、工况切换等指标。

### 6.1 DQN

DQN 使用离散动作：

| 动作 | 含义 |
|---:|---|
| 0 | 制动 |
| 1 | 惰行 |
| 2 | 半牵引 |
| 3 | 全牵引 |

当前 DQN 状态已加入：

```text
动车质量
拖车质量
```

因此支持质量条件化的共享 DQN。

DQN 可选择：

- 12 个独立 DQN：每辆车单独训练。
- 共享 DQN：12 个列车执行体共享同一套 Q 网络权重。

共享 DQN 开关：

```text
--share-lower-dqn
```

### 6.2 SAC

SAC 使用连续动作：

```text
force_ratio in [-1, 1]
```

其中：

- 正值表示牵引。
- 负值表示制动。
- 接近 0 表示惰行或小力控制。

SAC 当前支持：

- BC 预训练。
- RL 微调。
- 共享策略。
- 随机载重训练。
- 训练过程速度轨迹快照。

共享 SAC 开关：

```text
--share-lower-continuous
```

### 6.3 DQN 与 SAC 的区别

| 项目 | DQN | SAC |
|---|---|---|
| 动作空间 | 离散工况 | 连续牵引/制动力比例 |
| 网络输出 | 各动作 Q 值 | 连续动作分布 |
| 探索方式 | epsilon-greedy | 熵正则随机策略 |
| 控制精细度 | 较粗 | 较细 |
| 常见问题 | 工况切换明显 | 牵引力小幅波动 |
| 对照意义 | 离散控制基准 | 连续控制主实验 |

## 7. 下层奖励函数

当前下层奖励重点保留几项核心目标：

- 牵引能耗。
- 准时性。
- jerk 舒适性。
- 牵引力变化平滑项。
- 限速、超时、提前停车等必要约束。

训练中还包含专家知识或动作过滤：

- 到站之前速度不能提前变为 0；若提前为 0，则结束该回合并惩罚。
- 接近限速或前方安全限速时，不允许继续牵引。
- 到站速度必须为 0。
- 牵引/制动力做量化、死区和最小保持时间处理，以减少高频波动。

## 8. 上层 MAPPO 时刻表优化

上层负责多列车协同决策，当前主要采用在线事件步方式：

1. 判断哪些列车到达站点或进入下一站间。
2. 上层根据全局状态调整当前列车的时刻表/发车相关变量。
3. 调用下层共享或独立驾驶策略，生成当前站间速度轨迹。
4. 记录牵引能耗和再生制动能量时间序列。
5. 在真实供电分区内进行再生制动匹配。
6. 根据系统目标计算上层奖励。
7. 进入下一事件步。

上层目标包含两项：

```text
总牵引能耗
可利用再生制动能量
```

系统目标形式：

```text
J = w_E * E_traction_ratio - w_R * regen_utilization
```

默认权重：

```text
w_E = 0.5
w_R = 0.5
```

真实供电区间内，同一时间片存在一列车制动、另一列车牵引时，才认为再生制动能量可被吸收。

## 9. 上下层信息交换

上层传给下层：

- 当前列车。
- 当前站间。
- 目标运行时间。
- 当前列车质量和动力参数。
- 线路坡度、曲线、限速。
- 当前事件步时刻。

下层返回给上层：

- 实际运行时间。
- 到站误差。
- 牵引能耗。
- 再生制动能量。
- 速度轨迹。
- 牵引/制动力曲线。
- 牵引/制动能量时间序列。
- 是否可行。

当前在线版本不再只是静态 `E_min(T)` 响应表，而是在上层每个事件步直接调用下层策略生成当前站间轨迹。

## 10. 共享策略与随机载重训练

考虑后续要增加多站间载重变化，当前推荐使用：

```text
质量条件化共享下层策略
```

含义：

- 系统中仍有 12 个列车执行体。
- 但下层不是 12 个完全独立网络。
- 12 个列车共享一套训练得到的 DQN 或 SAC 权重。
- 每个列车在执行时根据自身状态、当前站间、目标运行时间、动车质量和拖车质量调用共享网络。

这样可以支持：

- 同一列车不同站间载重变化。
- 多列车不同载重。
- 随机载重训练。
- 新增列车数量时不必新增下层网络。

随机载重训练使用参数：

```text
--ddpg-random-mass-training
```

该参数名称历史上来自 DDPG，但当前已同步作用于 DQN/SAC/DDPG。

默认随机质量范围：

```text
235 t 至 275 t
```

训练覆盖统计会输出：

```text
lower_training_coverage_by_direction.csv
lower_training_coverage_by_mass_bin.csv
```

用于检查：

- 上行是否被采样。
- 下行是否被采样。
- 每个站间训练次数。
- 每个载重区间训练次数。
- 每个方向/站间/载重段的可行率。

如果下行没有被训练到，会明显影响下行执行效果，因为下行的坡度方向、限速继承方向、制动点和牵引点与上行不同。

## 11. 输出文件

常见输出目录：

```text
timetable/results_bidirectional_online/<run_name或自动时间目录>
```

主要输出包括：

```text
hierarchical_mappo_bidirectional_online_checkpoint.pt
```

保存上层 MAPPO 与下层控制器权重。

```text
lower_training_history.csv
```

下层训练历史。

```text
lower_training_process.png
```

下层训练过程，包括奖励、能耗、准时误差、jerk 等。

```text
lower_reward_training_process.png
```

专门的下层奖励过程图，包括 reward/km、总 reward、可行率、奖励分量。

```text
lower_reward_mass_sensitivity.png
```

载重与 reward/km 的关系散点图。

```text
lower_reward_by_section_summary.csv
```

按站间统计 reward/km、可行率、平均载重。

```text
lower_training_coverage_by_direction.csv
lower_training_coverage_by_mass_bin.csv
```

训练覆盖统计。

```text
mappo_training_process.png
```

上层 MAPPO 训练过程。

```text
online_timetable_optimization.png
online_timetable_schedule.csv
online_timetable_section_adjustments.csv
```

在线时刻表优化结果。

```text
up_train_XX_online_operation.png
down_train_XX_online_operation.png
```

单列车全线运行图，包含速度、限速、坡度高度、牵引/制动力。

```text
up_train_XX_online_section_speed_trajectories.png
down_train_XX_online_section_speed_trajectories.png
```

各站间速度轨迹图。

```text
up_train_XX_online_section_force_curves.png
down_train_XX_online_section_force_curves.png
```

各站间牵引/制动力曲线。

```text
run_metadata.json
train_parameters.csv
online_experiment_summary.csv
```

实验参数、列车配置和结果摘要。

## 12. 常用运行指令

以下指令均在项目根目录执行：

```text
E:\时刻表\SAC
```

### 12.1 共享 SAC 对比实验

```powershell
python -m timetable.timetable_0601_bidirectional_online --lower-algorithm sac --share-lower-continuous --lower-policy-mode fresh --upper-policy-mode fresh --lower-bc-episodes 1000 --lower-episodes 5000 --upper-episodes 500 --lower-log-interval 10 --lower-plot-interval 100 --progress-interval 10 --runtime-adjustment-span 0.5 --ddpg-random-mass-training --output-dir timetable/results_bidirectional_online/shared_sac_bc1000_finetune5000_upper500_massaware_compare
```

### 12.2 共享 DQN 对比实验

```powershell
python -m timetable.timetable_0601_bidirectional_online --lower-algorithm dqn --share-lower-dqn --lower-policy-mode fresh --upper-policy-mode fresh --lower-episodes 5000 --upper-episodes 500 --lower-log-interval 10 --progress-interval 10 --runtime-adjustment-span 0.5 --ddpg-random-mass-training --output-dir timetable/results_bidirectional_online/shared_dqn_lower5000_upper500_massaware_compare
```

### 12.3 固定载重 DQN 对比实验

```powershell
python -m timetable.timetable_0601_bidirectional_online --lower-algorithm dqn --lower-policy-mode fresh --upper-policy-mode fresh --lower-episodes 5000 --upper-episodes 500 --lower-log-interval 10 --progress-interval 10 --runtime-adjustment-span 0.5 --output-dir timetable/results_bidirectional_online/dqn_lower5000_upper500_fixed_mass_compare
```

### 12.4 快速检查共享 DQN 流程

```powershell
python -m timetable.timetable_0601_bidirectional_online --lower-algorithm dqn --share-lower-dqn --lower-policy-mode fresh --upper-policy-mode fresh --lower-episodes 200 --upper-episodes 50 --lower-log-interval 10 --progress-interval 10 --runtime-adjustment-span 0.5 --ddpg-random-mass-training --output-dir timetable/results_bidirectional_online/debug_shared_dqn_compare
```

### 12.5 快速检查共享 SAC 流程

```powershell
python -m timetable.timetable_0601_bidirectional_online --lower-algorithm sac --share-lower-continuous --lower-policy-mode fresh --upper-policy-mode fresh --lower-bc-episodes 20 --lower-episodes 50 --upper-episodes 20 --lower-log-interval 10 --lower-plot-interval 10 --progress-interval 10 --runtime-adjustment-span 0.5 --ddpg-random-mass-training --output-dir timetable/results_bidirectional_online/debug_shared_sac_compare
```

## 13. 当前注意事项

1. DQN 状态维度已从 15 改为 17，因此旧 DQN checkpoint 不能直接复用。

2. SAC/DDPG 的 BC 预训练轨迹由专家启发式生成，不是 SAC 自己探索得到。后续可将专家轨迹保存为数据集，复用进行 BC 训练。

3. `--runtime-adjustment-span` 是站间运行时间微调范围，当前应保持不超过：

```text
0.5 s
```

4. 发车时间调整和站间运行时间调整不是同一个概念。不要用 `--runtime-adjustment-span 10` 表示发车时间调整 10 秒。

5. 当前 `--ddpg-random-mass-training` 参数名较旧，但已被 DQN/SAC/DDPG 共同使用，表示训练时随机采样载重并重建列车动力学参数。

6. 下行必须被训练覆盖。共享策略训练结束后应检查：

```text
lower_training_coverage_by_direction.csv
```

确认 `down` 方向有足够样本。

7. 如果后续引入站间客流导致的载重变化，推荐继续使用共享 DQN 或共享 SAC，并在每个站间更新当前列车的 `TrainConfig`。

## 14. 推荐实验对比

建议至少保留以下实验组：

| 实验组 | 下层算法 | 策略结构 | 载重 |
|---|---|---|---|
| A | DQN | 12 个独立网络 | 固定载重 |
| B | DQN | 共享网络 | 随机载重 |
| C | SAC | 共享网络 | 随机载重 |
| D | SAC | 共享网络 + BC 预训练 | 随机载重 |

主要比较指标：

- 下层可行率。
- 到站误差。
- 牵引能耗。
- 再生制动利用率。
- 速度轨迹平滑性。
- 牵引/制动力波动。
- 上层 MAPPO 奖励收敛性。
- 训练覆盖均衡性。
