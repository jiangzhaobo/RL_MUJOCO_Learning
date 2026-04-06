import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import os


class FR3ReachEnv(gym.Env):
    """Franka FR3 到达目标的环境（与 FR3_SAC 中一致）"""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None, max_steps=200, n_substeps=None):
        super().__init__()

        model_path = os.path.join(
            os.path.dirname(__file__), "..", "models", "fr3_reach.xml"
        )
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.n_substeps = n_substeps

        self.home_qpos = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])

        self.target_range = {"x": [0.2, 0.6], "y": [-0.5, 0.5], "z": [0.1, 0.8]}

        self.action_space = spaces.Box(
            low=-0.03, high=0.03, shape=(7,), dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32
        )

        self.max_steps = max_steps
        self.current_step = 0

        self.render_mode = render_mode
        self.viewer = None

        self.ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site"
        )
        self.target_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target"
        )

        assert self.ee_site_id != -1, "找不到 attachment_site，请检查fr3.xml"
        assert self.target_body_id != -1, "找不到 target body，请检查fr3_reach.xml"

        self.success_threshold = 0.05
        self.success_consecutive = 2
        self.ee_vel_threshold = 0.01

    def _get_target_pos(self):
        return self.data.xpos[self.target_body_id].copy()

    def _get_obs(self):
        qpos = self.data.qpos[:7].copy()
        qvel = self.data.qvel[:7].copy()
        ee_pos = self.data.site_xpos[self.ee_site_id].copy()
        target_pos = self._get_target_pos()
        distance = np.linalg.norm(ee_pos - target_pos)
        obs = np.concatenate([qpos, qvel, ee_pos, target_pos, [distance]])
        return obs.astype(np.float32)

    def _sample_target(self):
        x = np.random.uniform(*self.target_range["x"])
        y = np.random.uniform(*self.target_range["y"])
        z = np.random.uniform(*self.target_range["z"])
        return np.array([x, y, z])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.data.qpos[:7] = self.home_qpos
        self.data.qvel[:7] = 0.0
        self.data.ctrl[:7] = self.home_qpos

        target_pos = self._sample_target()
        self.model.body_pos[self.target_body_id] = target_pos

        mujoco.mj_forward(self.model, self.data)

        self.current_step = 0

        ee_pos = self.data.site_xpos[self.ee_site_id].copy()
        target_pos = self._get_target_pos()
        self.prev_distance = float(np.linalg.norm(ee_pos - target_pos))
        self.prev_ee_pos = ee_pos.copy()

        self.stable_count = 0

        return self._get_obs(), {}

    def step(self, action):
        target_qpos = self.data.qpos[:7] + action
        low = self.model.actuator_ctrlrange[:7, 0]
        high = self.model.actuator_ctrlrange[:7, 1]
        target_qpos = np.clip(target_qpos, low, high)
        self.data.ctrl[:7] = target_qpos

        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        ee_pos = self.data.site_xpos[self.ee_site_id].copy()
        target_pos = self._get_target_pos()
        distance = np.linalg.norm(ee_pos - target_pos)

        dt = float(self.model.opt.timestep) * float(self.n_substeps)
        ee_vel_vec = (ee_pos - self.prev_ee_pos) / (dt + 1e-12)
        ee_vel_norm = float(np.linalg.norm(ee_vel_vec))

        progress = self.prev_distance - distance
        action_penalty = 0.05 * np.linalg.norm(action)
        reward = (
            -distance + 0.5 * np.exp(-10.0 * distance) + 1.0 * progress - action_penalty
        )

        if distance < self.success_threshold and ee_vel_norm < self.ee_vel_threshold:
            self.stable_count += 1
        else:
            self.stable_count = 0

        if self.stable_count >= self.success_consecutive:
            reward += 15.0
            terminated = True
        else:
            terminated = False

        self.prev_distance = distance
        self.prev_ee_pos = ee_pos.copy()

        self.current_step += 1
        truncated = self.current_step >= self.max_steps

        info = {"distance": distance, "ee_pos": ee_pos, "target_pos": target_pos}

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
