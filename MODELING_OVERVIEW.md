# 当前建模说明

本文档对应当前保留的上下行 6+6 双向实验版本：

```text
timetable/timetable_0601_bidirectional_online.py
timetable/bidirectional_model.py
```

## 线路与供电分区

线路共 19 个站间，上下行使用统一物理公里标。下行线路条件由下行坡度/曲率文件读取或由上行线路反向生成，坡度方向按行车方向处理。

真实供电分区配置保留在：

```text
timetable/config.py
```

当前供电分区：

```text
1: THP-TD, TD-HC, HC-DGNL
2: DGNL-THSC, THSC-SZL, SZL-SC, SC-SY
3: SY-SX, SX-CP, CP-JK, JK-ZLX, ZLX-ZL
4: ZL-ZX, ZX-KB, KB-FG, FG-ZC, ZC-ST
5: ST-ZG, ZG-ZCP
```

## 列车动力学

动力学模型位于：

```text
timetable/train_model.py
```

纵向动力学：

```text
a = (F - W) / m
W = W_davis + W_gradient + W_curve
```

戴维斯阻力：

```text
W_davis = a + b * v_kmh + c * v_kmh^2
a = (1.65 * 动车质量 + 0.78 * 拖车质量) * 9.8
b = (0.0247 * 动车质量 + 0.0028 * 拖车质量) * 9.8
c = 0.6566
```

牵引力约束：

```text
F_traction_max(v) = min(F_max, P_max / v)
```

当前已加入最大加速度/减速度约束：

```text
max_acceleration_mps2 = 1.0
max_deceleration_mps2 = 1.0
```

## 下层控制

DQN 模块：

```text
timetable/lower_control.py
```

SAC/DDPG 模块：

```text
timetable/lower_continuous.py
```

下层目标是在给定站间、给定目标运行时间、给定列车质量和线路条件下，生成速度轨迹与牵引/制动力曲线。

当前下层约束包括：

- 到终点速度为 0。
- 到站前速度不能提前为 0，否则终止回合并惩罚。
- 动作预测会导致超速时，安全层把它最小幅度投影到安全边界；安全动作原样通过。
- 当前新实验使用 `1.0 * 线路限速`，不再写死为 `0.85 * 限速`。
- 动作空间按当前阻力、质量、最大牵引力、最大制动力、最大加速度/减速度动态限制。
- SAC路径不使用手写时刻跟踪和主动限速保持；平滑只限制过大的牵引/制动力跳变。

## 上层协同

在线协同环境位于：

```text
timetable/online_hierarchical.py
```

主入口：

```text
timetable/timetable_0601_bidirectional_online.py
```

上层在事件步上决策，向下层传递当前列车、当前站间、目标运行时间、站停/发车约束等信息；下层返回当前站间速度轨迹、牵引能耗、再生制动能量和到站时间。上层再根据系统牵引能耗和同供电分区内可利用再生制动进行优化。

## 可视化输出

可视化模块：

```text
timetable/visualization.py
```

主要输出包括：

- 上层训练过程。
- 下层训练奖励过程。
- 全线速度、限速、坡度高度、牵引/制动力曲线。
- 各站间速度轨迹。
- 各站间牵引/制动力曲线。
- 时刻表优化结果。
- 再生制动匹配结果。
