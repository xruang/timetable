# Linux 服务器训练说明（RTX PRO 5000 Blackwell）

本文针对本项目和实验室的新服务器。推荐系统为 Ubuntu Server 24.04 LTS；项目在
2 TB NVMe 上运行，8 TB HDD 用于归档历史结果。不要把 Windows 的 `.venv`、旧
checkpoint 或历史 `results` 上传到服务器。

## 0. 裸机系统与显卡驱动

安装 Ubuntu Server 时勾选 OpenSSH Server。首次登录后安装基础工具：

```bash
sudo apt update
sudo apt full-upgrade -y
sudo apt install -y openssh-server git unzip tmux htop nvtop \
  build-essential python3 python3-venv python3-pip fonts-noto-cjk \
  linux-headers-$(uname -r)
```

先运行 `nvidia-smi`。如果整机厂已经装好驱动且能识别显卡，不要重复安装。如果没有
驱动，按 NVIDIA 官方 Ubuntu 仓库方式安装 open kernel modules：

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y nvidia-open
sudo reboot
```

重启后确认：

```bash
nvidia-smi
```

RTX PRO 5000 48 GB 是 Blackwell 显卡。使用当前 PyTorch CUDA 13 wheel 时，驱动版本
应不低于 `580.65.06`。这里不需要单独安装完整 CUDA Toolkit 或 cuDNN；本项目不编译
自定义 CUDA 扩展，PyTorch wheel 会携带所需的用户态 CUDA 库。若 `nvidia-smi` 仍
找不到设备，优先检查 Secure Boot 是否阻止了内核模块加载。

## 1. 在本机生成上传包

在项目目录运行：

```powershell
python server/package_project.py
```

将生成 `SAC_server_bundle.zip`。压缩包只包含当前源码、实验配置和四个线路
Excel 文件，不包含历史结果、缓存和旧 checkpoint。

从 Windows PowerShell 上传：

```powershell
scp "E:\时刻表\SAC\SAC_server_bundle.zip" 用户名@服务器IP:~/
```

## 2. 在服务器解压

```bash
unzip SAC_server_bundle.zip
cd SAC
```

Ubuntu 24.04 自带的 Python 3.12 可以直接使用。项目代码最低支持 Python 3.9。

## 3. 创建环境

默认安装 PyPI 当前稳定版 PyTorch。当前稳定 wheel 支持 Blackwell；不要使用本机旧的
PyTorch 2.6.0 + CUDA 11.8 组合：

```bash
bash server/setup_linux.sh
```

如果以后需要严格复现实验，可以用安装后生成的 `environment.lock.txt` 记录版本。
只有明确要测试旧版 CUDA 12.8 wheel 时才覆盖，例如：

```bash
TORCH_PACKAGE=torch==2.7.0 \
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 \
bash server/setup_linux.sh
```

环境检查会验证 PyTorch、CUDA、GPU、四个线路文件以及上下行各 19 个站间。

## 4. 先做环境和最小训练检查

```bash
source .venv/bin/activate
python server/check_environment.py --require-cuda
bash server/train_detached.sh lower_sac_smoke
```

`lower_sac_smoke` 只有 2 回合，用于检查代码、线路数据、结果保存和 checkpoint 链路，
不用于判断可行率或收敛性。日志无报错并生成结果目录后，再开始正式实验。

## 5. 后台训练单个算法

```bash
bash server/train_detached.sh lower_sac_baseline
```

其他下层对比实验：

```bash
bash server/train_detached.sh lower_dqn_baseline
bash server/train_detached.sh lower_ddpg_baseline
```

脚本会自动覆盖为：

```text
lower_device = cuda
upper_device = cuda
lower_checkpoint_interval = 500
```

配置文件中的研究参数不会被修改。本次实际参数仍会写入结果目录的
`experiment_config.json`。

## 6. 查看状态

启动脚本会打印日志文件和 PID。例如：

```bash
tail -f server_logs/lower_sac_baseline_YYYYMMDD_HHMMSS.log
watch -n 2 nvidia-smi
free -h
df -h
```

结果目录结构：

```text
results/<实验名>/<时间>_seed<随机种子>/
```

长时间训练期间每 500 回合更新：

```text
lower_policy_autosave.pt
lower_training_autosave_history.csv
lower_autosave_status.json
```

完整训练结束后保存：

```text
lower_policy_checkpoint.pt
run_status.json
全部训练 CSV 和图像
```

停止日志查看使用 `Ctrl+C`，不会终止后台训练。需要确认进程时可运行：

```bash
ps -fp PID
```

## 7. 从自动保存点继续

先查看 `lower_autosave_status.json` 中已完成的回合数，然后将
`lower_episodes` 设置为剩余回合数：

```bash
bash server/train_detached.sh lower_sac_baseline \
  --set lower_policy_mode=continue \
  --set lower_policy_checkpoint=/absolute/path/to/lower_policy_autosave.pt \
  --set lower_episodes=REMAINING_EPISODES
```

策略、价值网络和优化器状态会恢复。新的续训结果会进入一个新运行目录，不会覆盖
中断前的实验目录。随机采样序列和 DQN/DDPG 探索衰减会从新运行重新开始，因此它是
参数级续训，不是逐随机数完全复现。

## 8. 顺序执行三种算法

单张 GPU 建议顺序执行，避免三个进程同时抢占显存：

```bash
nohup bash server/train_all_baselines.sh \
  > server_logs/all_baselines.log 2>&1 &
```

当前仿真环境本身仍是单进程 CPU 逐步计算，GPU主要加速神经网络前向和参数更新。
如果 GPU 利用率不高但 CPU 单核占用较高，这是当前代码结构的预期表现。

## 9. 结果归档和下载

训练时优先把项目和当前结果放在 NVMe 上。完成一组实验后，再复制到 8 TB HDD 的
归档目录（下面路径可按实际挂载点修改）：

```bash
mkdir -p /data/SAC_archive
rsync -a --info=progress2 results/ /data/SAC_archive/results/
```

从 Windows 下载某次结果：

```powershell
scp -r 用户名@服务器IP:~/SAC/results/实验名/时间_seed编号 "E:\时刻表\服务器结果\"
```

本项目目前每个训练任务是单进程。24 核 CPU 和 192 GB 内存有余量，但三种算法不要
同时占用同一张 GPU；基线脚本默认按 SAC、DQN、DDPG 顺序运行。

## 10. 从已有 CSV 重新生成中文训练图

更新代码或安装中文字体后，不需要重新训练。指定一次实验结果目录即可重新绘图：

```bash
python server/regenerate_plots.py results/lower_sac_baseline/时间_seed7
```
