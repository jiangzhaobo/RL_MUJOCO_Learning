"""配置文件 - 基于 Franka FR3 的 A2C 训练配置"""

import os

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "fr3_reach.xml")
LOG_DIR = os.path.join(BASE_DIR, "logs")
SAVE_DIR = os.path.join(BASE_DIR, "saved_models")

# 训练配置（A2C 为同步 on-policy 算法）
TRAIN_CONFIG = {
    "total_updates": 1000_000,
    "n_envs": 15,
    "steps_per_update": 5,
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "value_coef": 0.5,
    "entropy_coef": 0.01,
    "max_grad_norm": 0.5,
    "save_interval": 1000,
}

# 环境配置
ENV_CONFIG = {
    "max_steps": 300,
    "n_substeps": 10,
}

# 测试配置
TEST_CONFIG = {"n_episodes": 100, "max_steps": 300, "render": True, "n_substeps": 5}
