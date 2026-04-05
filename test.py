"""测试脚本"""

import os
import argparse
import numpy as np
import time
from stable_baselines3 import PPO
from fr3_env import FR3ReachEnv
from config import TEST_CONFIG, SAVE_DIR
import mujoco.viewer


def test(args):
    """测试函数"""
    model_path = (
        args.model_path
        if args.model_path
        else os.path.join(SAVE_DIR, "fr3_reach_final.zip")
    )

    if not os.path.exists(model_path):
        print(f"错误：模型文件不存在: {model_path}")
        return

    print(f"加载模型: {model_path}")
    model = PPO.load(model_path)

    # 创建环境（带可视化）
    env = FR3ReachEnv(render_mode="human", max_steps=TEST_CONFIG.get("max_steps", 200))

    # 测试统计
    n_episodes = args.n_episodes if args.n_episodes else TEST_CONFIG["n_episodes"]
    success_count = 0
    total_rewards = []
    total_steps = []
    distances = []

    print(f"\n开始测试 {n_episodes} 轮...")
    print("=" * 50)

    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_steps += 1
            done = terminated or truncated

            if TEST_CONFIG["render"]:
                env.render()
                time.sleep(0.01)

        # 记录统计信息
        final_distance = info["distance"]
        is_success = final_distance < 0.05

        if is_success:
            success_count += 1

        total_rewards.append(episode_reward)
        total_steps.append(episode_steps)
        distances.append(final_distance)

        print(
            f"Episode {episode + 1}/{n_episodes}: "
            f"Reward={episode_reward:.2f}, Steps={episode_steps}, "
            f"Distance={final_distance:.4f}, Success={'true' if is_success else 'false'}"
        )

    # 打印总结
    print("\n" + "=" * 50)
    print("测试总结:")
    print(f"成功率: {100 * success_count / n_episodes:.1f}%")
    print(f"平均奖励: {np.mean(total_rewards):.2f}")
    print(f"平均步数: {np.mean(total_steps):.1f}")
    print(f"平均距离: {np.mean(distances):.4f}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试UR5e到达任务")
    parser.add_argument("--model_path", type=str, default=None, help="模型路径")
    parser.add_argument("--n_episodes", type=int, default=None, help="测试轮数")
    args = parser.parse_args()
    test(args)
