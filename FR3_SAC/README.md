# FR3_SAC

基于 `FR3_PPO` 框架改写的 Soft Actor-Critic (SAC) 实现，用于 Franka FR3 到达目标任务。

快速上手

1. 创建并激活你的 Python 环境（推荐 conda）：

```bash
conda create -n RL_MUJOCO python=3.10 -y
conda activate RL_MUJOCO
pip install -r requirements.txt
```

2. 在 `FR3_SAC` 下运行快速 smoke test：

```bash
cd fr3_reach_rl_ppo/FR3_SAC
python test.py
```

3. 开始训练：

```bash
python train.py
```

关键文件

- `config.py`：训练与环境配置
- `fr3_env.py`：MuJoCo 环境实现（与 `FR3_PPO` 保持一致）
- `train.py`：使用 `stable-baselines3.SAC` 的训练脚本
- `test.py`：快速的 smoke test

依赖（示例）

- mujoco（及其 Python 绑定）
- gymnasium
- stable-baselines3
- torch

注意

- 确保你的 MuJoCo license 与环境变量已正确配置。
- 如果使用 GPU，请确保 `torch` 与 CUDA 版本匹配。
