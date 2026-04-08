"""A2C 训练脚本（基于 stable_baselines3）"""

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import argparse
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from utils.fr3_env import FR3ReachEnv
from config import TRAIN_CONFIG, ENV_CONFIG, LOG_DIR, SAVE_DIR


def make_env():
    def _init():
        env = FR3ReachEnv(
            render_mode=None,
            max_steps=ENV_CONFIG["max_steps"],
            n_substeps=ENV_CONFIG["n_substeps"],
        )
        return Monitor(env)

    return _init


def train():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(SAVE_DIR, exist_ok=True)

    n_envs = TRAIN_CONFIG.get("n_envs", 1)
    if n_envs > 1:
        env = SubprocVecEnv([make_env() for _ in range(n_envs)])
    else:
        env = DummyVecEnv([make_env()])

    total_timesteps = TRAIN_CONFIG.get("total_updates", 20000) * TRAIN_CONFIG.get(
        "steps_per_update", 5
    )

    model = A2C(
        "MlpPolicy",
        env,
        learning_rate=TRAIN_CONFIG.get("learning_rate", 3e-4),
        n_steps=TRAIN_CONFIG.get("steps_per_update", 5),
        gamma=TRAIN_CONFIG.get("gamma", 0.99),
        vf_coef=TRAIN_CONFIG.get("value_coef", 0.5),
        ent_coef=TRAIN_CONFIG.get("entropy_coef", 0.01),
        max_grad_norm=TRAIN_CONFIG.get("max_grad_norm", 0.5),
        verbose=1,
        tensorboard_log=LOG_DIR,
    )

    print("开始 A2C 训练...")
    print(f"总训练步数: {total_timesteps}")

    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    final_model_path = os.path.join(SAVE_DIR, "fr3_reach_a2c_final.zip")
    model.save(final_model_path)
    print(f"训练完成！最终模型已保存到: {final_model_path}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练 FR3 到达任务（A2C）")
    parser.parse_args()
    train()
