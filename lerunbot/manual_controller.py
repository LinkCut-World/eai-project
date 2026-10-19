from colorama import Fore, Style
import lerunbot.envs.lerunbot_env as lerunbot_env
import numpy as np
import threading
from threading import Lock
import time
import argparse
import os

env = lerunbot_env.LerunbotEnv(
    control_decimation=20,
    control_frequency=50,
    controller_addr="tcp://localhost:7101",
    controller_version="0.0.1",
    position_server_addr="tcp://localhost:7201",
    skip_steps=-1,
)
env.reset()


capture = False
capture_lock = Lock()


def monitor():
    print("Started monitor thread.")
    while True:
        with capture_lock:
            if capture:
                try:
                    env.position_server.send_string("CAP")
                    image_data = env.position_server.recv_multipart()[-1].decode("utf-8")
                except Exception as e:
                    print(Fore.YELLOW + f"Error during image capture: {e}" + Style.RESET_ALL)
        time.sleep(1)


demo_actions = []


def manual_controller_loop():
    global capture
    while True:
        command = input("Enter command (CAP/c, FORWARD/f, TURN_LEFT/l, TURN_LEFT_BIG/L, TURN_RIGHT/r, TURN_RIGHT_BIG/R, DEMO/d, STOP/s): ").strip()
        print(f"Received command: {command}")
        if command in ["CAP", "c"]:
            with capture_lock:
                capture = not capture
            print(f"Capture mode set to: {capture}")
            # env.position_server.send_string("CAP")
            # image_data = env.position_server.recv_string()
            # print(f"Got captured image data, length: {len(image_data)}.")
            continue
        actions = None
        if command in ["FORWARD", "f"]:
            actions = lerunbot_env.from_obs(lerunbot_env.DEMO_FORWARD)
        elif command in ["TURN_LEFT", "l"]:
            actions = lerunbot_env.from_obs(lerunbot_env.DEMO_TURN_LEFT)
        elif command in ["TURN_LEFT_BIG", "L"]:
            actions = lerunbot_env.from_obs(lerunbot_env.DEMO_TURN_LEFT_BIG)
        elif command in ["TURN_RIGHT", "r"]:
            actions = lerunbot_env.from_obs(lerunbot_env.DEMO_TURN_RIGHT)
        elif command in ["TURN_RIGHT_BIG", "R"]:
            actions = lerunbot_env.from_obs(lerunbot_env.DEMO_TURN_RIGHT_BIG)
        elif command in ["DEMO", "d"]:
            if len(demo_actions) == 0:
                print(Fore.RED + "Error: No demo actions loaded." + Style.RESET_ALL)
                continue
            actions = demo_actions
        elif command in ["STOP", "s"]:
            actions = None
        else:
            print(Fore.RED + "Error: Unknown command." + Style.RESET_ALL)
            continue
        if actions is not None:
            # obs, reward, terminated, done, info = env.step(actions, use_degrees=True)
            expected_arms, true_arms = env.step_(actions)
            print(f"Executed actions for command: {command}")
            # print(f"Observation: {obs}, Reward: {reward}, Terminated: {terminated}, Done: {done}, Info: {info}")
            print(f"Expected Arms (degrees): {expected_arms} \nTrue Arms (degrees): {true_arms}")
        env.reset()
        print(f"OK: {command}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Manual controller for Lerunbot")
    parser.add_argument("--demo", type=str, default="envs/actions.npy", help="Path to demo actions file")
    args = parser.parse_args()
    if os.path.isfile(args.demo):
        demo_actions = np.load(args.demo, allow_pickle=True)
        for x in demo_actions:
            x[0] -= 6.0  # Adjust shoulder_pan for demo compatibility
        print(f"Loaded demo actions from {args.demo}, total steps: {len(demo_actions)}")
    print("Running manual controller...")
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    manual_controller_loop()
