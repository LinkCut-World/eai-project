import json
import argparse
from pathlib import Path
from omegaconf import OmegaConf
from torch import Tensor
from torchrl.data import TensorSpec
from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type

from lerunbot_serl.agent import make_agent
from lerunbot_serl.env import make_single_environment


import zmq
from colorama import Fore, Style
import lerunbot.envs.lerunbot_env as lerunbot_env
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control_decimation", "-d", type=int, default=20, help="Control decimation for the LerunbotEnv environment.")
    parser.add_argument("--checkpoint", "-c", type=str, required=True, help="Path to the checkpoint directory.")
    parser.add_argument("--no_reset", action="store_true", help="Whether to reset the LerunbotEnv environment after each step.")
    return parser.parse_args()


args = parse_args()
checkpoint_path = Path(args.checkpoint)
cfg = OmegaConf.load(checkpoint_path / "config.yaml")


socket = zmq.Context().socket(zmq.REP)
socket.bind("tcp://*:7301")
env = lerunbot_env.LerunbotEnv(
    control_decimation=args.control_decimation,
    control_frequency=50,
    controller_addr="tcp://localhost:7101",
    controller_version="0.0.1",
    position_server_addr="tcp://localhost:7201",
    action_repeat=1,
    skip_steps=-1,
)
env.reset()

single_collect_env = make_single_environment(cfg)
agent = make_agent(
    cfg,
    cfg.experiment.rollout_device,
    observation_spec=single_collect_env.observation_spec,
    action_spec=single_collect_env.action_spec,
)
agent.loads(checkpoint_path / "agent")
exploration_type = ExplorationType.DETERMINISTIC

RESET_QPOS_DEG = env.RESET_QPOS


def get_action(obs):
    data_ = single_collect_env.reset(target=obs)
    key_actions = []
    while True:
        with set_exploration_type(exploration_type):
            data_ = agent.policy()(data_)
        key_actions.append(data_["action"].detach().cpu().numpy())
        data, data_ = single_collect_env.step_and_maybe_reset(data_)
        if data["next", "done"].any():
            break
    # data = TensorDict(obs, batch_size=[])
    # with set_exploration_type(exploration_type):
    #     data = agent.policy()(data)
    # key_action = data["action"].detach().cpu().numpy()
    assert len(key_actions) == cfg.env.max_steps
    key_action = np.concatenate(key_actions, axis=0)
    key_action = key_action * 180 / np.pi
    key_action = key_action.reshape(-1, len(cfg.env.gym[cfg.env.gym.name].args["key_qpos"]))
    action = np.zeros([key_action.shape[0], len(RESET_QPOS_DEG)])
    action[:, cfg.env.gym[cfg.env.gym.name].args["key_qpos"]] = key_action
    action += RESET_QPOS_DEG
    return action


def low_controller_loop():
    global socket
    while True:
        try:
            command = socket.recv_string()
        except zmq.Again:
            print(Fore.YELLOW + "Warning: No message received." + Style.RESET_ALL)
            continue
        print(f"Received command: {command}")
        if command in ["CAP", "c"]:
            env.position_server.send_string("CAP")
            image_data = env.position_server.recv_multipart()[-1].decode("utf-8")
            socket.send_string(image_data)
            print(f"Sent captured image data, length: {len(image_data)}.")
            continue
        if command == "RESET":
            env.reset()
            socket.send_string(f"OK: RESET")
            print("Forced environment reset.")
            continue
        try:
            x, y, z, yaw = json.loads(command)
        except:
            print(Fore.RED + "Error: Invalid command format." + Style.RESET_ALL)
            socket.send_string("ERROR: Invalid command format.")
            continue
        actions = get_action({"target_trans": np.array([x, y, z], dtype=np.float32), "target_rot": np.array([yaw], dtype=np.float32)})
        # obs, reward, terminated, done, info = env.step(actions)
        _, _ = env.step_(actions)
        print(f"Executed actions for command: {command}")
        if not args.no_reset:
            env.reset(control_decimation=args.control_decimation)
        print(f"Sending OK response for command: {command}")
        socket.send_string(f"OK: {command}")


if __name__ == "__main__":
    print("Running low-level controller...")
    low_controller_loop()
