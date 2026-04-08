"""配置文件 - 基于 Franka FR3 的 DDPG 训练配置"""

import os

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "fr3_reach.xml")
LOG_DIR = os.path.join(BASE_DIR, "logs")
SAVE_DIR = os.path.join(BASE_DIR, "saved_models")

# 训练配置（DDPG 为 off-policy actor-critic）
TRAIN_CONFIG = {
    "total_timesteps": 1000000,
    "n_envs": 14,
    "learning_rate": 1e-3,
    "buffer_size": 500000,
    "batch_size": 256,
    "train_freq": 1,
    "gradient_steps": 1,
    "tau": 0.005,
    "gamma": 0.99,
    "max_grad_norm": 0.5,
    # 噪声参数（用于探索）
    "action_noise_std": 0.1,
}

# 环境配置
ENV_CONFIG = {"max_steps": 300, "n_substeps": 10}

# 测试配置
TEST_CONFIG = {"n_episodes": 100, "max_steps": 300, "render": True, "n_substeps": 5}
