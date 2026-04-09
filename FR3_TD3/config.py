"""配置文件 - 基于 Franka FR3 的 TD3 训练配置"""

import os

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "fr3_reach.xml")
LOG_DIR = os.path.join(BASE_DIR, "logs")
SAVE_DIR = os.path.join(BASE_DIR, "saved_models")

# 训练配置（TD3 双重延迟策略，含目标策略噪声）
TRAIN_CONFIG = {
    "total_timesteps": 1500000,
    "n_envs": 16,
    "learning_rate": 1e-3,
    "buffer_size": 500000,
    "batch_size": 256,
    "tau": 0.005,
    "gamma": 0.99,
    "policy_delay": 2,
    "target_policy_noise": 0.2,
    "target_noise_clip": 0.5,
    "action_noise_std": 0.1,
}

# 环境配置
ENV_CONFIG = {"max_steps": 300, "n_substeps": 10}

# 测试配置
TEST_CONFIG = {"n_episodes": 100, "max_steps": 300, "render": True, "n_substeps": 5}
