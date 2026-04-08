import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import os


class FR3ReachEnv(gym.Env):
    """Franka FR3机械臂到达目标点的强化学习环境"""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None, max_steps=200, n_substeps=None):
        super().__init__()

        # 加载MuJoCo模型（FR3场景）
        model_path = os.path.join(
            os.path.dirname(__file__), "..", "models", "fr3_reach.xml"
        )
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # 每个RL step执行的物理仿真子步数
        # timestep=0.002s，n_substeps=5 → 控制频率100Hz，物理频率500Hz
        self.n_substeps = n_substeps if n_substeps is not None else 5

        # FR3 Home位置（7个关节，来自fr3.xml的keyframe）
        self.home_qpos = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])

        # 目标球的可达范围（FR3臂展约0.855m）
        self.target_range = {"x": [0.2, 0.6], "y": [-0.5, 0.5], "z": [0.1, 0.8]}

        # 动作空间：关节角增量（rad）。将尺度略微减小以提高稳定性。
        self.action_space = spaces.Box(
            low=-0.03, high=0.03, shape=(7,), dtype=np.float32
        )

        # 观测空间：关节位置(7) + 关节速度(7) + 末端位置(3) + 目标位置(3) + 距离(1)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32
        )

        self.max_steps = max_steps
        self.current_step = 0

        # 渲染相关
        self.render_mode = render_mode
        self.viewer = None

        # 获取重要的ID
        self.ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site"
        )
        self.target_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target"
        )

        assert self.ee_site_id != -1, "找不到 attachment_site，请检查fr3.xml"
        assert self.target_body_id != -1, "找不到 target body，请检查fr3_reach.xml"

        # 成功判定与稳定性相关阈值
        self.success_threshold = 0.04  # m
        self.success_consecutive = 2  # 连续满足阈值的步数
        self.ee_vel_threshold = 0.01  # m/s，末端速度阈值

    def _get_target_pos(self):
        """从仿真数据中读取目标位置（修复3：不依赖Python变量）"""
        return self.data.xpos[self.target_body_id].copy()

    def _get_obs(self):
        """获取观测"""
        qpos = self.data.qpos[:7].copy()
        qvel = self.data.qvel[:7].copy()
        ee_pos = self.data.site_xpos[self.ee_site_id].copy()
        target_pos = self._get_target_pos()
        distance = np.linalg.norm(ee_pos - target_pos)

        obs = np.concatenate([qpos, qvel, ee_pos, target_pos, [distance]])
        return obs.astype(np.float32)

    def _sample_target(self):
        """随机生成目标位置"""
        x = np.random.uniform(*self.target_range["x"])
        y = np.random.uniform(*self.target_range["y"])
        z = np.random.uniform(*self.target_range["z"])
        return np.array([x, y, z])

    def reset(self, seed=None, options=None):
        """重置环境到home位置"""
        super().reset(seed=seed)

        # 重置关节到home位置，速度清零
        self.data.qpos[:7] = self.home_qpos
        self.data.qvel[:7] = 0.0

        # 重置 ctrl 为 home（只写入前7个控制通道，避免长度不匹配）
        self.data.ctrl[:7] = self.home_qpos

        # 随机生成目标位置
        target_pos = self._sample_target()
        self.model.body_pos[self.target_body_id] = target_pos

        # 前向运动学，更新所有xpos/site_xpos
        mujoco.mj_forward(self.model, self.data)

        self.current_step = 0

        # 初始化上一步距离与末端位置，用于基于距离差的奖励和速度估计
        ee_pos = self.data.site_xpos[self.ee_site_id].copy()
        target_pos = self._get_target_pos()
        self.prev_distance = float(np.linalg.norm(ee_pos - target_pos))
        self.prev_ee_pos = ee_pos.copy()

        # 稳定计数（连续满足成功条件的计数器）
        self.stable_count = 0

        return self._get_obs(), {}

    def step(self, action):
        """执行一步动作"""
        # 应用动作：当前关节角 + 增量，裁剪到关节范围内
        # 目标关节角 = 当前关节角 + 增量
        target_qpos = self.data.qpos[:7] + action

        # 将目标裁剪到 actuator 控制范围（按前7个通道），并只写入对应 ctrl 切片
        low = self.model.actuator_ctrlrange[:7, 0]
        high = self.model.actuator_ctrlrange[:7, 1]
        target_qpos = np.clip(target_qpos, low, high)
        self.data.ctrl[:7] = target_qpos

        # 执行多个物理仿真子步，让控制器真正跟上目标
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        # 获取新的观测
        obs = self._get_obs()
        ee_pos = self.data.site_xpos[self.ee_site_id].copy()
        target_pos = self._get_target_pos()
        distance = np.linalg.norm(ee_pos - target_pos)

        # 估计末端线速度（使用上一个 RL step 的末端位置差分）
        # dt = 模型 timestep * 子步数
        dt = float(self.model.opt.timestep) * float(self.n_substeps)
        ee_vel_vec = (ee_pos - self.prev_ee_pos) / (dt + 1e-12)
        ee_vel_norm = float(np.linalg.norm(ee_vel_vec))

        # 奖励设计（改进）
        # - 基础项：负距离（稠密）
        # - 进展奖励：上一时刻到当前时刻的距离差（靠近则为正）
        # - 动作惩罚：抑制过大/剧烈动作，帮助平滑性
        # - 成功奖励：到达阈值给大额奖励并终止

        progress = self.prev_distance - distance
        action_penalty = 0.01 * np.linalg.norm(action)
        reward = (
            -distance
            + 0.5 * np.exp(-10.0 * distance)
            + 1.2 * progress
            - action_penalty
            - 0.001 * self.current_step
        )

        # 稳定性判定：距离与速度都满足时计数，否则清零
        if distance < self.success_threshold and ee_vel_norm < self.ee_vel_threshold:
            self.stable_count += 1
        else:
            self.stable_count = 0

        # 当连续满足稳定条件到达设置步数时视为成功
        if self.stable_count >= self.success_consecutive:
            reward += 20.0
            terminated = True
        else:
            terminated = False

        # 更新 prev_distance 与 prev_ee_pos
        self.prev_distance = distance
        self.prev_ee_pos = ee_pos.copy()

        self.current_step += 1
        truncated = self.current_step >= self.max_steps

        info = {
            "distance": distance,
            "ee_pos": ee_pos,
            "target_pos": target_pos,
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        """渲染环境"""
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()

    def close(self):
        """关闭环境"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
