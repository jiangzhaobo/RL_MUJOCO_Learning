"""配置文件 - 基于 Franka FR3 的 SAC 训练配置"""

import os

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "fr3_reach.xml")
LOG_DIR = os.path.join(BASE_DIR, "logs")
SAVE_DIR = os.path.join(BASE_DIR, "saved_models")

# 训练配置（SAC 为离线/离策略算法，通常使用较大的重放池）
TRAIN_CONFIG = {
    "total_timesteps": 1500000,
    "n_envs": 15,
    "learning_rate": 3e-4,
    "buffer_size": 1000000,
    "batch_size": 256,
    "train_freq": 1,
    "gradient_steps": 1,
    "tau": 0.005,
    "gamma": 0.99,
    "ent_coef": "auto",
    "target_update_interval": 1,
    "max_grad_norm": 0.5,
}

# 环境配置
ENV_CONFIG = {
    "max_steps": 300,
    "n_substeps": 10,
}

# 测试配置
TEST_CONFIG = {"n_episodes": 100, "max_steps": 300, "render": True, "n_substeps": 5}
