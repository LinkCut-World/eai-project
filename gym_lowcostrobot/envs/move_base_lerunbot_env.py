import os
import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import Env, spaces

from gym_lowcostrobot import ASSETS_PATH

XML_MODEL_PATH = os.path.join(ASSETS_PATH, "..", "soarm101", "move_base.xml")


class MoveBaseLerunbotEnv(Env):
    """
    By simulating the contact and friction between the end of the robotic arm and the ground,
    the robotic arm is driven to move on the ground.

    Action Mode: "joint"
    Observation Mode: "state"
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    ARM_DOF = 6
    BASE_QPOS_DIM = 7
    BASE_QVEL_DIM = 6
    DEG_TO_RAD = np.pi / 180

    RESET_QPOS_DEG = np.array([0, 28.0, -10.0, 61.0, 0.0, 5.6])
    RESET_QPOS_RAD = RESET_QPOS_DEG * DEG_TO_RAD

    SAFETY_QPOS_DEG = np.array(
        [
            [-52.0, -100.0, -100.0, -100.0, -100.0, -100.0],
            [38.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        ]
    )
    SAFETY_QPOS_RAD = SAFETY_QPOS_DEG * DEG_TO_RAD

    def __init__(
        self,
        observation_mode="state",
        action_mode="joint",
        reward_type="dense",
        distance_threshold=0.1,
        rotation_threshold=0.1,
        target_x_range=None,
        target_y_range=None,
        target_r_range=None,
        target_t_range=None,
        target_yaw_range=None,
        relative_yaw=False,
        n_substeps=10,
        render_mode=None,
        camera_name=None,
        translation_weight: float = 1.0,
        moving_weight: float = 0.0,
        rotation_weight: float = 0.2,
        action_penalty_weight: float = 0.1,
        tracking_penalty_weight: float = 0.1,
        target_body_name: str = "gripper",
        detail: bool = False,
        action_repeat: int = 20,
        obs_scale: dict = None,
        max_step=None,
        key_qpos: list = None,
        action_noise: float = 0.0,
    ):
        """
        Initialize the MoveBaseLerunbotEnv simulation environment.

        Args:
            observation_mode (str): Mode of observation, usually "state".
            action_mode (str): Mode of action, usually "joint".
            reward_type (str): Type of reward, e.g., "dense".
            distance_threshold (float): Distance threshold to consider the target reached.
            rotation_threshold (float): Rotation threshold to consider the target reached.
            target_x_range (tuple): Range of target x coordinates.
            target_y_range (tuple): Range of target y coordinates.
            target_r_range (tuple): Range of target polar radius (used if x, y range is not provided).
            target_t_range (tuple): Range of target polar angle (used if x, y range is not provided).
            target_yaw_range (tuple): Range of target yaw angles.
            relative_yaw (bool): Whether the target yaw is relative to the target position's direction.
            n_substeps (int): Number of MuJoCo physics steps per control step (decimation).
            render_mode (str): Rendering mode, e.g., "human" or "rgb_array".
            camera_name (str): Name of the camera to use for rendering.
            translation_weight (float): Weight for translation progress reward.
            moving_weight (float): Penalty weight for base movement (encourages minimal unnecessary movement).
            rotation_weight (float): Weight for rotation progress reward.
            action_penalty_weight (float): Penalty weight for large action changes (encourages smoothness).
            tracking_penalty_weight (float): Penalty weight for tracking errors between expected and true joint positions.
            target_body_name (str): Name of the body representing the end-effector (e.g., "gripper").
            detail (bool): If True, returns detailed trajectory info in the info dict.
            action_repeat (int): Number of times an action is repeated (or length of action sequence).
            obs_scale (dict): Dictionary for scaling observation values.
            max_step (int): Maximum number of steps per episode.
            key_qpos (list): List of joint indices to control (allows restricting control to specific joints).
            action_noise (float): Standard deviation of Gaussian noise added to actions.
        """
        if not os.path.exists(XML_MODEL_PATH):
            print(f"Warning: XML file not found at {XML_MODEL_PATH}. Using PushCubeEnv placeholder.")
            self.model = mujoco.MjModel.from_xml_path(os.path.join(ASSETS_PATH, "push_cube.xml"))
        else:
            self.model = mujoco.MjModel.from_xml_path(XML_MODEL_PATH)

        self.data = mujoco.MjData(self.model)

        self.distance_threshold = distance_threshold
        self.rotation_threshold = rotation_threshold
        self.target_x_range = target_x_range
        self.target_y_range = target_y_range
        self.target_r_range = target_r_range
        self.target_t_range = target_t_range
        self.target_yaw_range = target_yaw_range
        self.relative_yaw = relative_yaw
        self.control_decimation = n_substeps
        self.translation_weight = translation_weight
        self.moving_weight = moving_weight
        self.rotation_weight = rotation_weight
        self.target_body_name = target_body_name
        self.action_penalty_weight = action_penalty_weight
        self.tracking_penalty_weight = tracking_penalty_weight
        self.detail = detail
        self.expected_arm = np.zeros_like(self.RESET_QPOS_RAD, dtype=np.float32)
        self.true_arm = np.zeros_like(self.RESET_QPOS_RAD, dtype=np.float32)
        self.obs_scale = obs_scale
        self.max_step = max_step
        self.key_qpos = key_qpos
        self.action_noise = action_noise

        if self.key_qpos is None:
            self.key_qpos = list(range(self.ARM_DOF))
        if self.obs_scale is not None and "arm_qpos" in self.obs_scale:
            raise NotImplementedError("Scaling 'arm_qpos' is not supported in MoveBaseLerunbotEnv.")
            # TODO: when scaling arm_qpos, need to scale action accordingly.
        if (
            int(target_x_range is not None) + int(target_y_range is not None) == 1
            or int(target_r_range is not None) + int(target_t_range is not None) == 1
            or int(target_x_range is not None) + int(target_r_range is not None) != 1
        ):
            raise ValueError(
                "Either (target_x_range and target_y_range) or (target_r_range and target_t_range) must be provided."
            )
        if self.target_yaw_range is None:
            raise ValueError("target_yaw_range must be provided.")

        if action_mode != "joint":
            raise ValueError("MoveBaseEnv only support 'joint' action mode.")
        self.action_repeat = action_repeat
        self.single_action_space = spaces.Box(
            low=(self.SAFETY_QPOS_RAD[0] - self.RESET_QPOS_RAD)[self.key_qpos],
            high=(self.SAFETY_QPOS_RAD[1] - self.RESET_QPOS_RAD)[self.key_qpos],
            shape=(len(self.key_qpos),),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.tile(self.single_action_space.low, self.action_repeat),
            high=np.tile(self.single_action_space.high, self.action_repeat),
            shape=(self.single_action_space.shape[0] * self.action_repeat,),
            dtype=np.float32,
        )

        observation_space_dict = {
            "target_trans": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "target_rot": spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32),
        }

        if self.action_repeat == 1:
            observation_space_dict.update(
                {
                    "base_qpos": spaces.Box(low=-np.inf, high=np.inf, shape=(self.BASE_QPOS_DIM,)),
                    "base_qvel": spaces.Box(low=-np.inf, high=np.inf, shape=(self.BASE_QVEL_DIM,)),
                    "arm_qpos": spaces.Box(
                        low=-np.pi - self.RESET_QPOS_RAD, high=np.pi - self.RESET_QPOS_RAD, shape=(self.ARM_DOF,)
                    ),
                    "arm_qvel": spaces.Box(low=-10.0, high=10.0, shape=(self.ARM_DOF,)),
                    "last_action": spaces.Box(
                        low=self.single_action_space.low,
                        high=self.single_action_space.high,
                        shape=(self.single_action_space.shape[0],),
                    ),
                }
            )
            if self.max_step is not None:
                observation_space_dict["time"] = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        else:
            self.max_step = 1

        if self.obs_scale is not None:
            for key, scale in self.obs_scale.items():
                if key in observation_space_dict:
                    space = observation_space_dict[key]
                    if isinstance(space, spaces.Box):
                        new_low = space.low * scale
                        new_high = space.high * scale
                        observation_space_dict[key] = spaces.Box(
                            low=new_low, high=new_high, shape=space.shape, dtype=space.dtype
                        )

        self.observation_space = gym.spaces.Dict(observation_space_dict)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.camera_name = camera_name
        if self.render_mode == "human":
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data, show_left_ui=False, show_right_ui=False)
            self.viewer.cam.azimuth = -45.0
            self.viewer.cam.distance = 1.5
            self.viewer.cam.elevation = -20.0
            self.viewer.cam.lookat = np.array([0.0, 0.0, 0.0])
        elif self.render_mode == "rgb_array":
            self.rgb_array_renderer = mujoco.Renderer(self.model, height=720, width=1280)

    def apply_action(self, action):
        """
        Step the simulation forward based on the 6D joint action.
        """
        action = np.clip(action, self.single_action_space.low, self.single_action_space.high)

        full_target = np.zeros(self.RESET_QPOS_RAD.shape, dtype=np.float32)
        full_target[self.key_qpos] = action
        if self.action_noise:
            noised_full_target = full_target + self.np_random.normal(0, self.action_noise, size=full_target.shape)
        else:
            noised_full_target = full_target
        previous_target = self.expected_arm.copy()

        arm_qpos_start_idx = self.BASE_QPOS_DIM
        arm_ctrl_idx = slice(0, self.ARM_DOF)
        for i in range(1, self.control_decimation + 1):
            qpos_target = noised_full_target * (i / self.control_decimation) + previous_target * (
                1.0 - i / self.control_decimation
            )

            self.data.ctrl[arm_ctrl_idx] = qpos_target + self.RESET_QPOS_RAD

            mujoco.mj_step(self.model, self.data)

            if self.render_mode == "human":
                self.viewer.sync()

        self.expected_arm = full_target
        self.true_arm = self.data.qpos[self.BASE_QPOS_DIM : self.BASE_QPOS_DIM + self.ARM_DOF] - self.RESET_QPOS_RAD

    def get_gripper_pose(self):
        body_id = self.model.body(self.target_body_name).id

        translation = self.data.body(body_id).xpos.copy()
        rotation_mujoco = self.data.body(body_id).xquat.copy()

        rotation_lerunbot_format = np.array(
            [rotation_mujoco[1], rotation_mujoco[2], rotation_mujoco[3], rotation_mujoco[0]]
        )

        return {
            "translation": translation.astype(np.float32),
            "rotation": rotation_lerunbot_format.astype(np.float32),
        }

    def get_base_yaw(self):
        quat = self.data.qpos[3:7].copy()

        # quaternion -> yaw
        qw, qx, qy, qz = quat
        yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        return float(yaw)

    def get_observation(self):
        observation = {
            "target_trans": np.array(self.target_position["translation"], dtype=np.float32),
            "target_rot": np.array(self.target_position["rotation"], dtype=np.float32),
        }

        if self.action_repeat == 1:
            observation.update(
                {
                    "base_qpos": np.array(self.data.qpos[: self.BASE_QPOS_DIM], dtype=np.float32),
                    "base_qvel": np.array(self.data.qvel[: self.BASE_QVEL_DIM], dtype=np.float32),
                    "arm_qpos": np.array(
                        self.data.qpos[self.BASE_QPOS_DIM : self.BASE_QPOS_DIM + self.ARM_DOF], dtype=np.float32
                    )
                    - self.RESET_QPOS_RAD,
                    "arm_qvel": np.array(
                        self.data.qvel[self.BASE_QVEL_DIM : self.BASE_QVEL_DIM + self.ARM_DOF], dtype=np.float32
                    ),
                    "last_action": np.array(self.expected_arm[self.key_qpos], dtype=np.float32),
                }
            )
            if self.max_step is not None:
                observation["time"] = np.array([self.step_count / self.max_step], dtype=np.float32)

        if self.obs_scale is not None:
            for key, scale in self.obs_scale.items():
                if key in observation:
                    observation[key] *= scale

        return observation

    def reset(self, seed=None, options=None, target=None):
        super().reset(seed=seed, options=options)
        self.step_count = 0

        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.qpos[: self.BASE_QPOS_DIM] = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        self.data.qpos[self.BASE_QPOS_DIM : self.BASE_QPOS_DIM + self.ARM_DOF] = self.RESET_QPOS_RAD
        self.expected_arm = np.zeros_like(self.RESET_QPOS_RAD, dtype=np.float32)

        if target is None:
            if self.target_x_range is not None:
                random_x = self.np_random.uniform(self.target_x_range[0], self.target_x_range[1])
                random_y = self.np_random.uniform(self.target_y_range[0], self.target_y_range[1])
            else:
                random_r = self.np_random.uniform(self.target_r_range[0], self.target_r_range[1])
                random_t = self.np_random.uniform(self.target_t_range[0], self.target_t_range[1])
                random_x = random_r * np.cos(random_t * self.DEG_TO_RAD)
                random_y = random_r * np.sin(random_t * self.DEG_TO_RAD)

            initial_z = self.data.qpos[2]

            random_trans = np.array([random_x, random_y, initial_z], dtype=np.float32)

            random_yaw = float(self.np_random.uniform(self.target_yaw_range[0], self.target_yaw_range[1]))
            if self.relative_yaw:
                random_yaw += np.arctan2(random_y, random_x)
        else:
            random_trans = np.array(target["target_trans"], dtype=np.float32)
            random_yaw = float(target["target_rot"])

        self.target_position = {
            "translation": random_trans,
            "rotation": random_yaw,
        }
        self.distance = np.linalg.norm(random_trans)
        self.last_trans = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.rotation = abs(random_yaw)



        try:
            self.model.geom("target_region").pos[:3] = random_trans
        except Exception:
            print("Warning: 'target_region' geom not found in the model XML.")
            pass

        mujoco.mj_forward(self.model, self.data)
        self.true_arm = self.data.qpos[self.BASE_QPOS_DIM : self.BASE_QPOS_DIM + self.ARM_DOF] - self.RESET_QPOS_RAD

        return self.get_observation(), {}

    def step(self, actions):
        self.step_count += 1

        action_dim = len(self.key_qpos)
        if isinstance(actions, np.ndarray) and actions.ndim == 1:
            assert (
                actions.size % action_dim == 0
            ), f"1D actions size {actions.size} must be divisible by action_dim={action_dim}"

            num = actions.size // action_dim
            actions = actions.reshape(num, action_dim)
        elif isinstance(actions, np.ndarray) and actions.ndim == 2:
            pass
        else:
            assert isinstance(actions, (list, tuple)), "Actions must be a single array or a list/array of arrays."

        if self.detail:
            details = {
                "qpos": [self.data.qpos.copy()],
                "qvel": [self.data.qvel.copy()],
            }
        gripper_reward = 0.0
        expected_arms = [self.expected_arm.copy()]
        true_arms = [self.true_arm.copy()]
        for action in actions:
            action_rad = np.clip(action, self.single_action_space.low, self.single_action_space.high)
            self.apply_action(action_rad)
            if self.detail:
                details["qpos"].append(self.data.qpos.copy())
                details["qvel"].append(self.data.qvel.copy())

            current_pose = self.get_gripper_pose()
            gripper_reward -= abs(current_pose["translation"][2]) * 0.01

            expected_arms.append(self.expected_arm.copy())
            true_arms.append(self.true_arm.copy())

        if self.step_count == self.max_step:
            expected_arms.append(np.zeros_like(self.RESET_QPOS_RAD, dtype=np.float32))
            true_arms.append(np.zeros_like(self.RESET_QPOS_RAD, dtype=np.float32))
        expected_arms = np.array(expected_arms)
        true_arms = np.array(true_arms)

        observation = self.get_observation()

        truncated, reward, reward_info = self.compute_reward(actions, expected_arms=expected_arms, true_arms=true_arms)

        terminated = self.max_step is not None and self.step_count >= self.max_step

        info = {"is_success": self.is_success(self.target_position)}
        info.update(reward_info)
        if self.detail:
            details["qpos"] = np.array(details["qpos"])
            details["qvel"] = np.array(details["qvel"])
            details["expected_arms"] = expected_arms
            details["true_arms"] = true_arms
            details["error"] = np.linalg.norm(expected_arms - true_arms, axis=1)
            info.update(details)

        for key in info:
            info[key] = np.array(info[key])
        return observation, reward, terminated, truncated, info

    def goal_distance(self, achieved_trans: np.ndarray, desired_trans: np.ndarray):
        return np.linalg.norm(achieved_trans - desired_trans)

    def is_success(self, desired_pose: dict):
        if desired_pose is None:
            return False

        # translation check (use the base position stored in qpos[:3])
        achieved_trans = np.array(self.data.qpos[:3].copy())
        desired_trans = np.array(desired_pose["translation"])

        distance = self.goal_distance(achieved_trans, desired_trans)

        # rotation: compare base quaternion in qpos[3:7] (w,x,y,z) to desired quaternion
        achieved_rot = self.get_base_yaw()
        desired_rot = np.array(desired_pose["rotation"])
        yaw_diff = achieved_rot - desired_rot
        rot_angle = (yaw_diff + np.pi) % (2 * np.pi) - np.pi

        return (distance < self.distance_threshold) and (rot_angle < self.rotation_threshold)

    def compute_reward(
        self, actions, expected_arms=None, true_arms=None, gripper_reward=None
    ) -> tuple[bool, float, dict]:

        target_trans = np.array(self.target_position["translation"])
        target_rot = np.array(self.target_position["rotation"])
        current_trans = self.data.qpos[:3].copy()
        current_rot = self.get_base_yaw()

        rewards = {}

        if self.translation_weight:
            distance = np.linalg.norm(current_trans - target_trans)
            rewards["translation_reward"] = float(self.translation_weight) * (self.distance - distance)
            self.distance = distance

        if self.moving_weight:
            moving_distance = np.linalg.norm(current_trans - self.last_trans)
            rewards["moving_penalty"] = -float(self.moving_weight) * moving_distance
            self.last_trans = current_trans

        if self.rotation_weight:
            rewards["rotation_reward"] = -float(self.rotation_weight) * (
                abs((current_rot - target_rot + np.pi) % (2 * np.pi) - np.pi) - self.rotation
            )
            self.rotation = abs((current_rot - target_rot + np.pi) % (2 * np.pi) - np.pi)



        if true_arms is not None and self.action_penalty_weight:
            diffs = np.diff(true_arms, axis=0)
            rewards["action_penalty"] = -self.action_penalty_weight * np.sum(diffs**2)

        if true_arms is not None and expected_arms is not None and self.tracking_penalty_weight:
            rewards["tracking_penalty"] = -self.tracking_penalty_weight * np.sum((true_arms - expected_arms)[1:] ** 2)

        if gripper_reward is not None:
            rewards["gripper_reward"] = gripper_reward

        reward = sum(rewards.values())
        info = rewards

        return False, float(reward), info

    def render(self, render_mode=None, camera_name=None):
        if render_mode is None:
            render_mode = self.render_mode
        if camera_name is None:
            camera_name = self.camera_name
        if render_mode == "human":
            self.viewer.sync()
        elif render_mode == "rgb_array":
            self.rgb_array_renderer.update_scene(self.data, camera=camera_name)
            return self.rgb_array_renderer.render()

    def close(self):
        if self.render_mode == "human":
            self.viewer.close()

        if self.render_mode == "rgb_array":
            self.rgb_array_renderer.close()


if __name__ == "__main__":
    print("Init MoveBaseLerunbotEnv...")

    env = MoveBaseLerunbotEnv(render_mode="human", n_substeps=10)

    num_episodes = 2

    TRAJECTORY_LENGTH = 500

    for episode in range(num_episodes):
        observation, info = env.reset()

        step_count = 0

        target_trans = env.target_position["translation"]
        print(f"\n--- Episode {episode + 1} ---")
        print(f"Target E.E. Pos (X, Y, Z): {target_trans[0]:.3f}, {target_trans[1]:.3f}, {target_trans[2]:.3f}")

        actions_sequence = []

        for i in range(TRAJECTORY_LENGTH):
            low_deg = env.SAFETY_QPOS_RAD[0]
            high_deg = env.SAFETY_QPOS_RAD[1]

            if i == 0:
                next_action_deg = np.random.uniform(low_deg * 0.5, high_deg * 0.5, size=(env.ARM_DOF,)).astype(
                    np.float32
                )
            else:
                noise = np.random.uniform(-5.0, 5.0, size=(env.ARM_DOF,))
                next_action_deg = np.clip(actions_sequence[-1] + noise, low_deg, high_deg).astype(np.float32)

            actions_sequence.append(next_action_deg)

        print(f"Generated a batch of {TRAJECTORY_LENGTH} actions.")

        observation, total_reward, terminated, truncated, info = env.step(actions_sequence)

        step_count = TRAJECTORY_LENGTH

        print(f"Sequence Execution Complete.")
        print(f"  Total steps executed: {step_count}")
        print(f"  Final Reward (based on final pose): {total_reward:.3f}")

    print("\ntest complete. Closing environment.")
    env.close()
