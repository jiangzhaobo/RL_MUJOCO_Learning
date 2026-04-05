"""训练脚本"""

import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from fr3_env import FR3ReachEnv
from config import TRAIN_CONFIG, ENV_CONFIG, LOG_DIR, SAVE_DIR


def make_env():
    """创建单个环境的工厂函数"""

    def _init():
        env = FR3ReachEnv(
            max_steps=ENV_CONFIG["max_steps"],
            n_substeps=ENV_CONFIG["n_substeps"],
        )
        env = Monitor(env)
        return env

    return _init


def train():
    """训练函数"""
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(SAVE_DIR, exist_ok=True)

    n_envs = TRAIN_CONFIG["n_envs"]

    # 使用SubprocVecEnv并行多个环境，大幅提升样本效率
    env = SubprocVecEnv([make_env() for _ in range(n_envs)])

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=TRAIN_CONFIG["learning_rate"],
        n_steps=TRAIN_CONFIG["n_steps"],
        batch_size=TRAIN_CONFIG["batch_size"],
        n_epochs=TRAIN_CONFIG["n_epochs"],
        gamma=TRAIN_CONFIG["gamma"],
        gae_lambda=TRAIN_CONFIG["gae_lambda"],
        clip_range=TRAIN_CONFIG["clip_range"],
        ent_coef=TRAIN_CONFIG["ent_coef"],
        vf_coef=TRAIN_CONFIG["vf_coef"],
        max_grad_norm=TRAIN_CONFIG["max_grad_norm"],
        verbose=1,
        tensorboard_log=LOG_DIR,
    )

    # checkpoint_callback = CheckpointCallback(
    #     save_freq=max(10000 // n_envs, 1),
    #     save_path=SAVE_DIR,
    #     name_prefix="fr3_reach_model",
    # )

    print("开始训练...")
    print(f"总训练步数: {TRAIN_CONFIG['total_timesteps']}")
    print(f"并行环境数: {n_envs}")
    print(
        f"每step子步数: {ENV_CONFIG['n_substeps']}（控制频率 {1/(0.002*ENV_CONFIG['n_substeps']):.0f}Hz）"
    )
    print(f"日志目录: {LOG_DIR}  →  tensorboard --logdir={LOG_DIR}")

    model.learn(
        total_timesteps=TRAIN_CONFIG["total_timesteps"],
        # callback=checkpoint_callback,
        progress_bar=True,
    )

    final_model_path = os.path.join(SAVE_DIR, "fr3_reach_final.zip")
    model.save(final_model_path)
    print(f"训练完成！最终模型已保存到: {final_model_path}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练FR3到达任务")
    parser.parse_args()
    train()
