"""SAC 训练脚本（基于 FR3_PPO 框架改写）"""

import os
import argparse
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from fr3_env import FR3ReachEnv
from config import TRAIN_CONFIG, ENV_CONFIG, LOG_DIR, SAVE_DIR


def make_env():
    def _init():
        env = FR3ReachEnv(
            render_mode="human",
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

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=TRAIN_CONFIG["learning_rate"],
        buffer_size=TRAIN_CONFIG["buffer_size"],
        batch_size=TRAIN_CONFIG["batch_size"],
        tau=TRAIN_CONFIG["tau"],
        gamma=TRAIN_CONFIG["gamma"],
        train_freq=TRAIN_CONFIG.get("train_freq", 1),
        gradient_steps=TRAIN_CONFIG.get("gradient_steps", 1),
        ent_coef=TRAIN_CONFIG.get("ent_coef", "auto"),
        verbose=1,
        tensorboard_log=LOG_DIR,
    )

    # checkpoint_callback = CheckpointCallback(
    #     save_freq=10000,
    #     save_path=SAVE_DIR,
    #     name_prefix="fr3_sac_model",
    # )

    print("开始 SAC 训练...")
    print(f"总训练步数: {TRAIN_CONFIG['total_timesteps']}")
    print(f"并行环境数: {n_envs}")
    print(f"日志目录: {LOG_DIR}  →  tensorboard --logdir={LOG_DIR}")

    model.learn(total_timesteps=TRAIN_CONFIG["total_timesteps"], progress_bar=True)

    final_model_path = os.path.join(SAVE_DIR, "fr3_reach_sac_final.zip")
    model.save(final_model_path)
    print(f"训练完成！最终模型已保存到: {final_model_path}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练 FR3 Reach 任务（SAC）")
    parser.parse_args()
    train()
