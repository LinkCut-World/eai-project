import os
from typing import Optional, Union, Dict, Any
import gymnasium as gym
from gymnasium import Env, spaces
import numpy as np
from scipy.spatial.transform import Rotation as R
import zmq
import time
import json
from colorama import Fore, Style
import argparse

DEMO_FORWARD = [
    {"shoulder_pan.pos": -6.0, "shoulder_lift.pos": 28.0, "elbow_flex.pos": -10.0, "wrist_flex.pos": 61.0, "wrist_roll.pos": 0.0, "gripper.pos": 5.6},
    {"shoulder_pan.pos": -6.0, "shoulder_lift.pos": 31.23076923076923, "elbow_flex.pos": -28.802197802197803, "wrist_flex.pos": 61.84615384615385, "wrist_roll.pos": 85.05494505494505, "gripper.pos": 6.149903907751442},
    {"shoulder_pan.pos": -6.0, "shoulder_lift.pos": 31.23076923076923, "elbow_flex.pos": -40.802197802197803, "wrist_flex.pos": 55.43956043956044, "wrist_roll.pos": 85.4065934065934, "gripper.pos": 6.149903907751442},
    {"shoulder_pan.pos": -6.0, "shoulder_lift.pos": 50.91208791208791, "elbow_flex.pos": -40.802197802197803, "wrist_flex.pos": 55.43956043956044, "wrist_roll.pos": 85.4065934065934, "gripper.pos": 6.149903907751442},
    {"shoulder_pan.pos": -6.0, "shoulder_lift.pos": 69.91208791208791, "elbow_flex.pos": -40.802197802197803, "wrist_flex.pos": 50.43956043956044, "wrist_roll.pos": 85.4065934065934, "gripper.pos": 6.149903907751442},
    {"shoulder_pan.pos": -6.0, "shoulder_lift.pos": 75.32967032967033, "elbow_flex.pos": -40.12087912087912, "wrist_flex.pos": 72.2967032967033, "wrist_roll.pos": 85.05494505494505, "gripper.pos": 5.829596412556054},
    {"shoulder_pan.pos": -6.0, "shoulder_lift.pos": 63.32967032967033, "elbow_flex.pos": -31.12087912087912, "wrist_flex.pos": 93.2967032967033, "wrist_roll.pos": 85.05494505494505, "gripper.pos": 5.829596412556054},
    {"shoulder_pan.pos": -6.0, "shoulder_lift.pos": 52.32967032967033, "elbow_flex.pos": -26.12087912087912, "wrist_flex.pos": 98.2967032967033, "wrist_roll.pos": 85.05494505494505, "gripper.pos": 5.829596412556054},
    {"shoulder_pan.pos": -6.0, "shoulder_lift.pos": 41.34065934065934, "elbow_flex.pos": -21.89010989010989, "wrist_flex.pos": 102.24175824175825, "wrist_roll.pos": 85.05494505494505, "gripper.pos": 6.2139654067905195},
    {"shoulder_pan.pos": -6.0, "shoulder_lift.pos": 41.51648351648352, "elbow_flex.pos": -33.40659340659341, "wrist_flex.pos": 82.32967032967034, "wrist_roll.pos": 85.05494505494505, "gripper.pos": 6.2139654067905195},
    {"shoulder_pan.pos": -6.0, "shoulder_lift.pos": 28.0, "elbow_flex.pos": -10.0, "wrist_flex.pos": 61.0, "wrist_roll.pos": 0.0, "gripper.pos": 5.6},
]

DEMO_TURN_LEFT = [
    {"shoulder_pan.pos": -6.0, "shoulder_lift.pos": 28.0, "elbow_flex.pos": -10.0, "wrist_flex.pos": 61.0, "wrist_roll.pos": 0.0, "gripper.pos": 5.6},
    {"shoulder_pan.pos": -4.615384615384615, "shoulder_lift.pos": 29.23076923076923, "elbow_flex.pos": -4.131868131868132, "wrist_flex.pos": 57.18681318681319, "wrist_roll.pos": 88.7032967032967, "gripper.pos": 5.88235294117647},
    {"shoulder_pan.pos": -4.615384615384615, "shoulder_lift.pos": 29.91208791208791, "elbow_flex.pos": -4.923076923076923, "wrist_flex.pos": 72.21978021978022, "wrist_roll.pos": 89.23076923076923, "gripper.pos": 5.88235294117647},
    {"shoulder_pan.pos": -4.615384615384615, "shoulder_lift.pos": 43.91208791208791, "elbow_flex.pos": -4.923076923076923, "wrist_flex.pos": 72.21978021978022, "wrist_roll.pos": 89.23076923076923, "gripper.pos": 5.88235294117647},
    {"shoulder_pan.pos": 6.615384615384615, "shoulder_lift.pos": 43.91208791208791, "elbow_flex.pos": -5.8901098901098905, "wrist_flex.pos": 72.3076923076923, "wrist_roll.pos": 89.31868131868131, "gripper.pos": 5.88235294117647},
    {"shoulder_pan.pos": 16.615384615384615, "shoulder_lift.pos": 43.91208791208791, "elbow_flex.pos": -5.8901098901098905, "wrist_flex.pos": 72.3076923076923, "wrist_roll.pos": 89.31868131868131, "gripper.pos": 5.88235294117647},
    {"shoulder_pan.pos": 16.516483516483518, "shoulder_lift.pos": 23.956043956043956, "elbow_flex.pos": -7.1208791208791204, "wrist_flex.pos": 83.82417582417582, "wrist_roll.pos": 89.31868131868131, "gripper.pos": 5.743944636678201},
    {"shoulder_pan.pos": -6.0, "shoulder_lift.pos": 28.0, "elbow_flex.pos": -10.0, "wrist_flex.pos": 61.0, "wrist_roll.pos": 0.0, "gripper.pos": 5.6},
]

DEMO_TURN_RIGHT = [
    {"shoulder_pan.pos": -6.0, "shoulder_lift.pos": 28.0, "elbow_flex.pos": -10.0, "wrist_flex.pos": 61.0, "wrist_roll.pos": 0.0, "gripper.pos": 5.6},
    {"shoulder_pan.pos": -4.615384615384615, "shoulder_lift.pos": 29.23076923076923, "elbow_flex.pos": -4.131868131868132, "wrist_flex.pos": 57.18681318681319, "wrist_roll.pos": 88.7032967032967, "gripper.pos": 5.88235294117647},
    {"shoulder_pan.pos": -4.615384615384615, "shoulder_lift.pos": 29.91208791208791, "elbow_flex.pos": -4.923076923076923, "wrist_flex.pos": 72.21978021978022, "wrist_roll.pos": 89.23076923076923, "gripper.pos": 5.88235294117647},
    {"shoulder_pan.pos": -4.615384615384615, "shoulder_lift.pos": 43.91208791208791, "elbow_flex.pos": -4.923076923076923, "wrist_flex.pos": 72.21978021978022, "wrist_roll.pos": 89.23076923076923, "gripper.pos": 5.88235294117647},
    {"shoulder_pan.pos": -14.164835164835164, "shoulder_lift.pos": 43.91208791208791, "elbow_flex.pos": -5.8901098901098905, "wrist_flex.pos": 72.3076923076923, "wrist_roll.pos": 89.31868131868131, "gripper.pos": 5.88235294117647},
    {"shoulder_pan.pos": -24.164835164835164, "shoulder_lift.pos": 43.91208791208791, "elbow_flex.pos": -5.8901098901098905, "wrist_flex.pos": 72.3076923076923, "wrist_roll.pos": 89.31868131868131, "gripper.pos": 5.88235294117647},
    {"shoulder_pan.pos": -24.516483516483518, "shoulder_lift.pos": 23.956043956043956, "elbow_flex.pos": -7.1208791208791204, "wrist_flex.pos": 83.82417582417582, "wrist_roll.pos": 89.31868131868131, "gripper.pos": 5.743944636678201},
    {"shoulder_pan.pos": -6.0, "shoulder_lift.pos": 28.0, "elbow_flex.pos": -10.0, "wrist_flex.pos": 61.0, "wrist_roll.pos": 0.0, "gripper.pos": 5.6},
]

DEMO_TURN_LEFT_BIG = [
    {"shoulder_pan.pos": -6.0, "shoulder_lift.pos": 28.0, "elbow_flex.pos": -10.0, "wrist_flex.pos": 61.0, "wrist_roll.pos": 0.0, "gripper.pos": 5.6},
    {"shoulder_pan.pos": -4.615384615384615, "shoulder_lift.pos": 29.23076923076923, "elbow_flex.pos": -4.131868131868132, "wrist_flex.pos": 57.18681318681319, "wrist_roll.pos": 88.7032967032967, "gripper.pos": 5.88235294117647},
    {"shoulder_pan.pos": -4.615384615384615, "shoulder_lift.pos": 29.91208791208791, "elbow_flex.pos": -4.923076923076923, "wrist_flex.pos": 72.21978021978022, "wrist_roll.pos": 89.23076923076923, "gripper.pos": 5.88235294117647},
    {"shoulder_pan.pos": -4.615384615384615, "shoulder_lift.pos": 43.91208791208791, "elbow_flex.pos": -4.923076923076923, "wrist_flex.pos": 72.21978021978022, "wrist_roll.pos": 89.23076923076923, "gripper.pos": 5.88235294117647},
    {"shoulder_pan.pos": 16.615384615384615, "shoulder_lift.pos": 43.91208791208791, "elbow_flex.pos": -5.8901098901098905, "wrist_flex.pos": 72.3076923076923, "wrist_roll.pos": 89.31868131868131, "gripper.pos": 5.88235294117647},
    {"shoulder_pan.pos": 36.615384615384615, "shoulder_lift.pos": 43.91208791208791, "elbow_flex.pos": -5.8901098901098905, "wrist_flex.pos": 72.3076923076923, "wrist_roll.pos": 89.31868131868131, "gripper.pos": 5.88235294117647},
    {"shoulder_pan.pos": 36.516483516483518, "shoulder_lift.pos": 23.956043956043956, "elbow_flex.pos": -7.1208791208791204, "wrist_flex.pos": 83.82417582417582, "wrist_roll.pos": 89.31868131868131, "gripper.pos": 5.743944636678201},
    {"shoulder_pan.pos": -6.0, "shoulder_lift.pos": 28.0, "elbow_flex.pos": -10.0, "wrist_flex.pos": 61.0, "wrist_roll.pos": 0.0, "gripper.pos": 5.6},
]

DEMO_TURN_RIGHT_BIG = [
    {"shoulder_pan.pos": -6.0, "shoulder_lift.pos": 28.0, "elbow_flex.pos": -10.0, "wrist_flex.pos": 61.0, "wrist_roll.pos": 0.0, "gripper.pos": 5.6},
    {"shoulder_pan.pos": -4.615384615384615, "shoulder_lift.pos": 29.23076923076923, "elbow_flex.pos": -4.131868131868132, "wrist_flex.pos": 57.18681318681319, "wrist_roll.pos": 88.7032967032967, "gripper.pos": 5.88235294117647},
    {"shoulder_pan.pos": -4.615384615384615, "shoulder_lift.pos": 29.91208791208791, "elbow_flex.pos": -4.923076923076923, "wrist_flex.pos": 72.21978021978022, "wrist_roll.pos": 89.23076923076923, "gripper.pos": 5.88235294117647},
    {"shoulder_pan.pos": -4.615384615384615, "shoulder_lift.pos": 43.91208791208791, "elbow_flex.pos": -4.923076923076923, "wrist_flex.pos": 72.21978021978022, "wrist_roll.pos": 89.23076923076923, "gripper.pos": 5.88235294117647},
    {"shoulder_pan.pos": -24.164835164835164, "shoulder_lift.pos": 43.91208791208791, "elbow_flex.pos": -5.8901098901098905, "wrist_flex.pos": 72.3076923076923, "wrist_roll.pos": 89.31868131868131, "gripper.pos": 5.88235294117647},
    {"shoulder_pan.pos": -44.164835164835164, "shoulder_lift.pos": 43.91208791208791, "elbow_flex.pos": -5.8901098901098905, "wrist_flex.pos": 72.3076923076923, "wrist_roll.pos": 89.31868131868131, "gripper.pos": 5.88235294117647},
    {"shoulder_pan.pos": -44.516483516483518, "shoulder_lift.pos": 23.956043956043956, "elbow_flex.pos": -7.1208791208791204, "wrist_flex.pos": 83.82417582417582, "wrist_roll.pos": 89.31868131868131, "gripper.pos": 5.743944636678201},
    {"shoulder_pan.pos": -6.0, "shoulder_lift.pos": 28.0, "elbow_flex.pos": -10.0, "wrist_flex.pos": 61.0, "wrist_roll.pos": 0.0, "gripper.pos": 5.6},
]


def from_obs(obs_sequence):
    actions = []
    for action in obs_sequence:
        actions.append(
            np.array(
                [
                    action["shoulder_pan.pos"],
                    action["shoulder_lift.pos"],
                    action["elbow_flex.pos"],
                    action["wrist_flex.pos"],
                    action["wrist_roll.pos"],
                    action["gripper.pos"],
                ]
            )
        )
    return actions


class LerunbotEnv(Env):
    """
    LerunbotEnv

    For mysterious reasons, the qpos values are in degrees, yaw angle is in radians, step_ uses degrees, step uses radians.
    """

    ARM_DOF = 6
    DEG_TO_RAD = np.pi / 180
    RAD_TO_DEG = 180 / np.pi

    RESET_QPOS = np.array([-6.0, 28.0, -10.0, 61.0, 0.0, 5.6])
    RESET_QPOS_DEG = RESET_QPOS
    RESET_QPOS_RAD = RESET_QPOS_DEG * DEG_TO_RAD

    SAFETY_QPOS = np.array(
        [
            [-63.0, -104.0, -98.0, -101.0, -0.1, 5.5],
            [45.0, 103.0, 96.0, 101.0, 0.1, 5.7],
        ]
    )
    SAFETY_QPOS_DEG = SAFETY_QPOS
    SAFETY_QPOS_RAD = SAFETY_QPOS_DEG * DEG_TO_RAD

    def __init__(
        self,
        control_decimation: int = 10,
        reset_decimation: int = 50,
        control_frequency: Union[int, float] = 50.0,
        controller_addr: Optional[str] = None,
        controller_version: Optional[str] = None,
        position_server_addr: Optional[str] = None,
        translation_weight: float = 1.0,
        rotation_weight: float = 0.2,
        action_penalty_weight: float = 0.1,
        tracking_penalty_weight: float = 0.1,
        action_repeat: int = None,
        key_qpos: list = None,
        target_x_range=None,
        target_y_range=None,
        target_r_range=None,
        target_t_range=None,
        target_yaw_range=(0.0, 0.0),
        relative_yaw=False,
        observation_mode: str = "quat",
        pause_before_step: bool = False,
        skip_steps: int = 10,
    ):
        """
        Initialize the LerunbotEnv real-world environment.

        Args:
            control_decimation (int): Number of sub-steps used to interpolate actions during a normal control step.
            reset_decimation (int): Number of sub-steps used to interpolate actions during environment reset to ensure safe, slow movement.
            control_frequency (Union[int, float]): Frequency at which commands are sent to the controller.
            controller_addr (Optional[str]): ZMQ address of the hardware low-level controller (e.g., 'tcp://localhost:7101').
            controller_version (Optional[str]): Version string of the controller protocol.
            position_server_addr (Optional[str]): ZMQ address of the camera-based position server.
            translation_weight (float): Weight for the translation progress reward.
            rotation_weight (float): Weight for the rotation progress reward.
            action_penalty_weight (float): Penalty weight for abrupt changes in actions (smoothness penalty).
            tracking_penalty_weight (float): Penalty weight for the difference between expected actions and actual joint readings.
            action_repeat (int): Number of actions expected in a sequence for a single step.
            key_qpos (list): List of joint indices that are actively controlled.
            target_x_range (tuple): Range of target x coordinates (if sampling uniformly in Cartesian coordinates).
            target_y_range (tuple): Range of target y coordinates.
            target_r_range (tuple): Range of target polar radius (if sampling in polar coordinates).
            target_t_range (tuple): Range of target polar angle.
            target_yaw_range (tuple): Range of target yaw angles.
            relative_yaw (bool): Whether the yaw target should be relative to the position vector.
            observation_mode (str): Observation format, "quat" (quaternion) or "yaw".
            pause_before_step (bool): If True, waits for user input before executing each step. Useful for debugging real-world safety.
            skip_steps (int): Number of initial steps to skip without applying actions, to allow system initialization.
        """
        assert isinstance(control_decimation, int) and control_decimation > 0, "control_decimation must be a positive integer."
        assert control_frequency > 0, "control_frequency must be positive."
        if key_qpos is None:
            key_qpos = list(range(self.ARM_DOF))
        if action_repeat is None:
            print(Fore.YELLOW + "Warning: action repeat not specified, set to default value." + Style.RESET_ALL)
            action_repeat = 1
        if int(target_x_range is not None) + int(target_y_range is not None) == 1 or int(target_r_range is not None) + int(target_t_range is not None) == 1 or int(target_x_range is not None) + int(target_r_range is not None) != 1:
            print(Fore.YELLOW + "Either (target_x_range and target_y_range) or (target_r_range and target_t_range) must be provided, set to default values." + Style.RESET_ALL)
            target_x_range, target_y_range = (0, 0.2), (0, 0)
            target_r_range, target_t_range = None, None

        super().__init__()

        self.control_decimation = control_decimation
        self.reset_decimation = reset_decimation
        self.control_frequency = control_frequency
        self.sleep_time = 1.0 / control_frequency
        self.expected_qpos = self.RESET_QPOS.copy()
        self.controller_addr = controller_addr
        self.controller_version = controller_version
        self.position_server_addr = position_server_addr
        self.translation_weight = translation_weight
        self.rotation_weight = rotation_weight
        self.action_penalty_weight = action_penalty_weight
        self.tracking_penalty_weight = tracking_penalty_weight
        self.action_repeat = action_repeat
        self.target_x_range = target_x_range
        self.target_y_range = target_y_range
        self.target_r_range = target_r_range
        self.target_t_range = target_t_range
        self.target_yaw_range = target_yaw_range
        self.relative_yaw = relative_yaw
        self.pause_before_step = pause_before_step
        self.target_position = None
        self.total_steps = 0
        self.key_qpos = key_qpos
        self.skip_steps = skip_steps

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

        if observation_mode == "quat":
            observation_space_dict = {
                "target_trans": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                "target_rot": spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32),
            }
            self.observation_space = spaces.Dict(observation_space_dict)
        elif observation_mode == "yaw":
            observation_space_dict = {
                "target_trans": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                "target_rot": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            }
            self.observation_space = spaces.Dict(observation_space_dict)
        else:
            raise NotImplementedError(f"Observation mode '{observation_mode}' is not implemented.")

        if self.controller_addr is not None:
            assert self.controller_version is not None, "Controller version must be specified if controller address is provided."
            if not self.controller_addr.startswith("tcp://"):
                self.controller_addr = "tcp://" + self.controller_addr
            try:
                _ = int(self.controller_addr.split(":")[-1])
            except ValueError:
                self.controller_addr = self.controller_addr + ":7101"
            self.controller = zmq.Context().socket(zmq.PUSH)
            self.controller.set_hwm(1)
            self.controller.connect(self.controller_addr)
            print(f"Connected to controller at {self.controller_addr}:", self.controller)
            self.observation_server_addr = self.controller_addr
            port = int(self.observation_server_addr.split(":")[-1]) + 1
            self.observation_server_addr = ":".join(self.observation_server_addr.split(":")[:-1] + [str(port)])
            self.observation_server = zmq.Context().socket(zmq.DEALER)
            self.observation_server.setsockopt(zmq.SNDTIMEO, 5000)
            self.observation_server.setsockopt(zmq.RCVTIMEO, 5000)
            self.observation_server.connect(self.observation_server_addr)
        else:

            class DummyServer:
                def send(self, *args, **kwargs):
                    pass

                def send_string(self, *args, **kwargs):
                    pass

                def send_multipart(self, *args, **kwargs):
                    pass

                def recv(self, *args, **kwargs):
                    return b'{"version": "0.0.0", "timestamp": 0, "content": {}}'

                def recv_string(self, *args, **kwargs):
                    return "ERROR"

                def recv_multipart(self, *args, **kwargs):
                    return [b"ERROR"]

                def close(self):
                    pass

            self.controller = DummyServer()
            self.observation_server = DummyServer()
            print(Fore.YELLOW + "No controller address specified, running in standalone mode." + Style.RESET_ALL)

        if self.position_server_addr is not None:
            if not self.position_server_addr.startswith("tcp://"):
                self.position_server_addr = "tcp://" + self.position_server_addr
            try:
                _ = int(self.position_server_addr.split(":")[-1])
            except ValueError:
                self.position_server_addr = self.position_server_addr + ":7201"
            self.position_server = zmq.Context().socket(zmq.DEALER)
            self.position_server.setsockopt(zmq.SNDTIMEO, 5000)  # Set send timeout
            self.position_server.setsockopt(zmq.RCVTIMEO, 5000)  # Set receive timeout
            self.position_server.connect(self.position_server_addr)
            print(f"Connected to position server at {self.position_server_addr}:", self.position_server)
        else:

            class DummyPositionServer:
                def send_string(self, *args, **kwargs):
                    pass

                def recv_string(self, *args, **kwargs):
                    return "ERROR"

                def recv_multipart(self, *args, **kwargs):
                    return [b"ERROR"]

                def close(self):
                    pass

            self.position_server = DummyPositionServer()
            print(Fore.YELLOW + "No position server address specified, position requests will be ignored." + Style.RESET_ALL)

        self.last_timestamp = time.time()
        print("LerunbotEnv initialized.")

    @staticmethod
    def camera_to_world(pos: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert position from camera frame to world frame.
        :param pos: position dict with 'translation' and 'rotation' keys
        :return: converted position dict
        """
        R_cw = np.array(
            [
                [0, 0, 1],
                [1, 0, 0],
                [0, 1, 0],
            ]
        )
        rot_cw = R.from_matrix(R_cw)
        t_cam = np.asarray(pos["translation"])
        q_cam = np.asarray(pos["rotation"])  # (x, y, z, w)
        t_world = R_cw @ t_cam
        q_world = (rot_cw * R.from_quat(q_cam) * rot_cw.inv()).as_quat()
        return {
            "translation": t_world,
            "rotation": q_world,
        }

    @staticmethod
    def quat_to_yaw(quat: np.ndarray) -> float:
        """
        Extract yaw angle from quaternion.
        :param quat: quaternion as (x, y, z, w)
        :return: yaw angle in radians
        """
        return np.arctan2(2.0 * (quat[3] * quat[2]), 1.0 - 2.0 * (quat[2] ** 2))

    def get_real_qpos(self) -> np.ndarray:
        try:
            self.observation_server.send_string("OBS")
            msg = self.observation_server.recv_multipart()[-1]
            data = json.loads(msg.decode("utf-8") )
            qpos = np.array(
                [
                    data["observation"].get("shoulder_pan.pos", self.expected_qpos[0]),
                    data["observation"].get("shoulder_lift.pos", self.expected_qpos[1]),
                    data["observation"].get("elbow_flex.pos", self.expected_qpos[2]),
                    data["observation"].get("wrist_flex.pos", self.expected_qpos[3]),
                    data["observation"].get("wrist_roll.pos", self.expected_qpos[4]),
                    data["observation"].get("gripper.pos", self.expected_qpos[5]),
                ]
            )
            return qpos
        except zmq.Again:
            print(Fore.YELLOW + "Warning: No observation received from server." + Style.RESET_ALL)
            return self.expected_qpos
        except Exception as e:
            print(Fore.RED + f"Error receiving observation from server: {e}" + Style.RESET_ALL)
            return self.expected_qpos

    def apply_action(self, action: np.ndarray, control_decimation: Optional[int] = None):
        """
        Step the simulation forward based on the 6D joint action.
        """
        action = np.clip(action, self.SAFETY_QPOS[0], self.SAFETY_QPOS[1])
        # Temporary clipping
        action = np.clip(action, np.array([-63.0, 0, -98.0, -101.0, -0.1, 5.5]), np.array([45.0, 103.0, 96.0, 101.0, 0.1, 5.7]))
        control_decimation = control_decimation if control_decimation is not None else self.control_decimation
        for i in range(1, control_decimation + 1):
            qpos = action * (i / control_decimation) + self.expected_qpos * (1.0 - i / control_decimation)
            msg = {
                "version": self.controller_version,
                "timestamp": time.time(),
                "content": {
                    "shoulder_pan.pos": float(qpos[0]),
                    "shoulder_lift.pos": float(qpos[1]),
                    "elbow_flex.pos": float(qpos[2]),
                    "wrist_flex.pos": float(qpos[3]),
                    "wrist_roll.pos": float(qpos[4]),
                    "gripper.pos": float(qpos[5]),
                },
            }
            try:
                self.controller.send(json.dumps(msg).encode("utf-8"), zmq.NOBLOCK)
            except zmq.Again:
                print(Fore.YELLOW + "Warning: failed to send action to controller." + Style.RESET_ALL)
            except Exception as e:
                print(Fore.RED + f"Error sending action to controller: {e}" + Style.RESET_ALL)
            cur_timestamp = time.time()
            sleep_duration = self.sleep_time - (cur_timestamp - self.last_timestamp)
            if sleep_duration > 0:
                time.sleep(sleep_duration)
            self.last_timestamp = time.time()
        self.expected_qpos = action.copy()

    def position1(self):
        try:
            self.position_server.send_string("POS1")
            parts = self.position_server.recv_multipart()
            response = parts[-1].decode("utf-8")
            if response != "OK":
                print(Fore.RED + "Error: Position server returned error for POS1." + Style.RESET_ALL)
                return None
            return True
        except Exception as e:
            print(Fore.RED + f"Error requesting POS1 from position server: {e}" + Style.RESET_ALL)
            return None

    def position2(self):
        try:
            self.position_server.send_string("POS2")
            parts = self.position_server.recv_multipart()
            response = parts[-1].decode("utf-8")
            if response == "ERROR":
                print(Fore.RED + "Error: Position server returned error for POS2." + Style.RESET_ALL)
                return None
            data = json.loads(response)
            return data
        except Exception as e:
            print(Fore.RED + f"Error requesting POS2 from position server: {e}" + Style.RESET_ALL)
            return None

    def get_observation(self):
        observation = {
            "target_trans": np.array(self.target_position["translation"], dtype=np.float32),
            "target_rot": np.array(self.target_position["rotation"], dtype=np.float32),
        }
        return observation

    def reset(self, seed=None, options=None, control_decimation: Optional[int] = None, stop_skip: bool = False):
        if control_decimation is None:
            control_decimation = self.reset_decimation
        if stop_skip:
            self.skip_steps = -1
        super().reset(seed=seed, options=options)
        print("Resetting LerunbotEnv.")
        self.apply_action(self.RESET_QPOS, control_decimation=control_decimation)
        print("LerunbotEnv reset complete.")

        if self.target_x_range is not None:
            random_x = self.np_random.uniform(self.target_x_range[0], self.target_x_range[1])
            random_y = self.np_random.uniform(self.target_y_range[0], self.target_y_range[1])
        else:
            random_r = self.np_random.uniform(self.target_r_range[0], self.target_r_range[1])
            random_t = self.np_random.uniform(self.target_t_range[0], self.target_t_range[1])
            random_x = random_r * np.cos(random_t * self.DEG_TO_RAD)
            random_y = random_r * np.sin(random_t * self.DEG_TO_RAD)
        initial_z = 0
        random_trans = np.array([random_x, random_y, initial_z], dtype=np.float32)

        random_yaw = float(self.np_random.uniform(self.target_yaw_range[0], self.target_yaw_range[1]))
        if self.relative_yaw:
            random_yaw += np.arctan2(random_y, random_x)

        self.target_position = {
            "translation": random_trans,
            "rotation": random_yaw,
        }
        return self.get_observation(), {}

    def step_(self, actions: Union[list[np.ndarray], np.ndarray]):
        """
        step_: actions in degrees, no observation, reward, done, info returned
        """
        self.position1()
        expected_arms = [self.expected_qpos.copy()]
        true_arms = [self.get_real_qpos()]
        for action in actions:
            if self.total_steps > self.skip_steps:
                self.apply_action(action)
            expected_arms.append(self.expected_qpos.copy())
            true_arms.append(self.get_real_qpos())
        expected_arms.append(expected_arms[0])
        true_arms.append(true_arms[0])
        expected_arms = np.array(expected_arms)
        true_arms = np.array(true_arms)
        return expected_arms, true_arms

    def step(self, actions: Union[list[np.ndarray], np.ndarray]):
        """
        step: actions in radians if use_degrees=False, degrees if use_degrees=True
        """
        self.total_steps += 1
        if self.total_steps <= self.skip_steps:
            print(f"Step {self.total_steps}: skipped")
        actions = np.array(actions)
        assert actions.ndim <= 2
        if actions.ndim == 1:
            actions = actions.reshape(-1, len(self.key_qpos))
        if self.pause_before_step and self.total_steps > self.skip_steps:
            _ = input("Press Enter to continue to next step...")

        true_actions = np.tile(self.RESET_QPOS_DEG, (actions.shape[0], 1))
        true_actions[:, self.key_qpos] += actions * self.RAD_TO_DEG
        actions = true_actions

        expected_arms, true_arms = self.step_(actions)
        expected_arms *= self.DEG_TO_RAD
        true_arms *= self.DEG_TO_RAD

        observation = self.get_observation()
        ret, reward, reward_info = self.compute_reward(actions, expected_arms, true_arms)
        terminated = True
        truncated = ret
        info = {}
        info.update(reward_info)
        return observation, reward, terminated, truncated, info

    def compute_reward(self, actions, expected_arms=None, true_arms=None) -> tuple[bool, float]:
        """
        :return: failed (bool), reward (float)
        """
        if self.total_steps > self.skip_steps:
            time.sleep(1)
        pos = self.position2()
        if pos is not None and self.target_position is not None:
            success = pos["success"]
            pos = self.camera_to_world(pos)
            pos["rotation"] = self.quat_to_yaw(np.array(pos["rotation"]))

            rewards = {}

            print(f"true_translation: {pos['translation']}")
            print(f"targ_translation: {self.target_position['translation']}")
            distance = np.linalg.norm(np.array(pos["translation"]) - np.array(self.target_position["translation"]))
            original_distance = np.linalg.norm(np.array(self.target_position["translation"]))
            rewards["translation_reward"] = float(self.translation_weight) * (original_distance - distance)

            print(f"true_rotation: {pos['rotation']}")
            print(f"targ_rotation: {self.target_position['rotation']}")
            rotation_diff = abs((pos["rotation"] - self.target_position["rotation"] + np.pi) % (2 * np.pi) - np.pi)
            original_rotation_diff = abs(self.target_position["rotation"])
            rewards["rotation_reward"] = float(self.rotation_weight) * (original_rotation_diff - rotation_diff)

            if true_arms is not None:
                diffs = np.diff(true_arms, axis=0)
                rewards["action_penalty"] = -self.action_penalty_weight * np.sum(diffs**2)

            if true_arms is not None and expected_arms is not None:
                rewards["tracking_penalty"] = -self.tracking_penalty_weight * np.sum((true_arms - expected_arms)[1:] ** 2)



            reward = sum(rewards.values())
            if not success:
                print(Fore.YELLOW + "Warning: Position server reported failure to detect Lerunbot." + Style.RESET_ALL)
                reward = -1.0
            info = rewards
            print("Reward components:")
            for k, v in info.items():
                print(f"  {k}: {v}")
            print(f"Reward: {reward}")
            return False, float(reward), info

        print(Fore.YELLOW + "Warning: Position data not available; cannot compute reward." + Style.RESET_ALL)
        return True, 0.0, {}

    def close(self):
        self.controller.close()
        self.position_server.close()
        print("LerunbotEnv closed.")


def _test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", type=str, default="actions.npy", help="Path to demo actions file (numpy .npy format).")
    args = parser.parse_args()

    forward_actions = from_obs(DEMO_FORWARD)
    left_actions = from_obs(DEMO_TURN_LEFT)
    right_actions = from_obs(DEMO_TURN_RIGHT)
    big_left_actions = from_obs(DEMO_TURN_LEFT_BIG)
    big_right_actions = from_obs(DEMO_TURN_RIGHT_BIG)

    loaded_actions = []
    if os.path.isfile(args.demo):
        loaded_actions = np.load(args.demo, allow_pickle=True)
        for x in loaded_actions:
            x[0] -= 6.0  # Adjust shoulder_pan for demo compatibility
        print(f"Loaded actions from {args.demo}:", loaded_actions)
    else:
        print(Fore.YELLOW + f"Demo actions file '{args.demo}' not found. Demo option will be unavailable." + Style.RESET_ALL)

    env = LerunbotEnv(
        control_decimation=20,
        control_frequency=50,
        controller_addr="tcp://localhost:7101",
        controller_version="0.0.1",
        position_server_addr="tcp://localhost:7201",
        pause_before_step=False,
        skip_steps=-1,
    )
    env.target_position = {
        "translation": [0.0, 0.0, -1.0],
        "rotation": [0.0, 0.0, 0.0, 1.0],
    }
    env.reset()
    while True:
        cmd = input("Enter command (left/l, Left/L, right/r, Right/R, forward/f, quit/q, demo/d): ").strip().lower()
        demo_actions = []
        if cmd in ["quit", "q"]:
            break
        elif cmd in ["left", "l"]:
            demo_actions = left_actions
        elif cmd in ["Left", "L"]:
            demo_actions = big_left_actions
        elif cmd in ["right", "r"]:
            demo_actions = right_actions
        elif cmd in ["Right", "R"]:
            demo_actions = big_right_actions
        elif cmd in ["forward", "f"]:
            demo_actions = forward_actions
        elif cmd in ["demo", "d"]:
            demo_actions = loaded_actions
        env.step_(demo_actions)
        env.reset()
    env.close()


if __name__ == "__main__":
    _test()
