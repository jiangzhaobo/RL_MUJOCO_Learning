"""配置文件 - 基于 Franka FR3 机械臂"""

import os

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "fr3_reach.xml")
LOG_DIR = os.path.join(BASE_DIR, "logs")
SAVE_DIR = os.path.join(BASE_DIR, "saved_models")

# 训练配置
TRAIN_CONFIG = {
    "total_timesteps": 1500000,  # 从500k增加到2M，机械臂任务至少需要1M+
    "n_envs": 10,  # 并行环境数，大幅提升样本效率
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 256,  # 并行环境后batch_size相应增大
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.15,
    "ent_coef": 0.005,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
}

# 环境配置
ENV_CONFIG = {
    "max_steps": 300,  # 训练和测试统一使用200步
    "n_substeps": 10,  # 每个RL step执行20个物理步，控制频率25Hz
}

# 测试配置
TEST_CONFIG = {
    "n_episodes": 100,
    "max_steps": 300,  # 修复：与训练保持一致，避免分布偏移
    "render": True,
}
