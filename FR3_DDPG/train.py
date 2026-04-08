"""DDPG 训练脚本（参考 FR3_SAC 结构）"""

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import argparse
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from utils.fr3_env import FR3ReachEnv
from config import TRAIN_CONFIG, ENV_CONFIG, LOG_DIR, SAVE_DIR


def make_env():
    def _init():
        env = FR3ReachEnv(
            render_mode=None,
            max_steps=ENV_CONFIG["max_steps"],
            n_substeps=ENV_CONFIG["n_substeps"],
        )
        env = Monitor(env)
        return env

    return _init


def train():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(SAVE_DIR, exist_ok=True)

    n_envs = TRAIN_CONFIG.get("n_envs", 1)

    if n_envs > 1:
        env = SubprocVecEnv([make_env() for _ in range(n_envs)])
    else:
        env = DummyVecEnv([make_env()])

    # 获取动作维度以创建噪声
    action_dim = env.action_space.shape[0]
    noise_std = TRAIN_CONFIG.get("action_noise_std", 0.1)
    action_noise = NormalActionNoise(
        mean=np.zeros(action_dim), sigma=noise_std * np.ones(action_dim)
    )

    model = DDPG(
        "MlpPolicy",
        env,
        learning_rate=TRAIN_CONFIG["learning_rate"],
        buffer_size=TRAIN_CONFIG["buffer_size"],
        batch_size=TRAIN_CONFIG["batch_size"],
        tau=TRAIN_CONFIG["tau"],
        gamma=TRAIN_CONFIG["gamma"],
        train_freq=TRAIN_CONFIG.get("train_freq", 1),
        gradient_steps=TRAIN_CONFIG.get("gradient_steps", 1),
        action_noise=action_noise,
        verbose=1,
        tensorboard_log=LOG_DIR,
    )

    print("开始 DDPG 训练...")
    print(f"总训练步数: {TRAIN_CONFIG['total_timesteps']}")
    print(f"并行环境数: {n_envs}")
    print(f"日志目录: {LOG_DIR}  →  tensorboard --logdir={LOG_DIR}")

    model.learn(total_timesteps=TRAIN_CONFIG["total_timesteps"], progress_bar=True)

    final_model_path = os.path.join(SAVE_DIR, "fr3_reach_ddpg_final.zip")
    model.save(final_model_path)
    print(f"训练完成！最终模型已保存到: {final_model_path}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练 FR3 Reach 任务（DDPG）")
    parser.parse_args()
    train()
