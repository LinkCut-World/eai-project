controller_version = "0.0.1"
controller_versions = controller_version.split(".")
CHECK_VERSION = True

import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat

from lerobot.configs import parser
from lerobot.robots import (
    Robot,
    RobotConfig,
    make_robot_from_config,
)
from lerobot.robots.so101_follower import SO101FollowerConfig
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import init_logging

import json
from typing import Optional
import threading
from threading import Thread, Lock
import colorama
from colorama import Fore, Style
import zmq


@dataclass
class FollowerConfig:
    robot: RobotConfig
    fps: int = 100
    controller_addr: Optional[str] = None


robot_lock = Lock()
attached = False

param_lock = Lock()
current_obs: dict | None = None
current_action: dict | None = None
current_control_fps = 0
current_command_fps = 0

action_server_lock = Lock()
action_server: zmq.Socket | None = None
observation_server_lock = Lock()
observation_server: zmq.Socket | None = None


def action_server_daemon():
    global action_server, current_action, current_command_fps
    print(Fore.GREEN + "Action server daemon thread started." + Style.RESET_ALL)

    recv_time = time.perf_counter()
    while True:
        action = None
        with action_server_lock:
            if action_server is not None:
                try:
                    raw = action_server.recv(zmq.NOBLOCK)
                    # raw = action_server.recv()
                    if len(raw) > 0:
                        msg = json.loads(raw.decode("utf-8"))
                        if CHECK_VERSION:
                            version = msg.get("version", "inf")
                            versions = version.split(".")
                            assert len(versions) == len(controller_versions), f"Version length mismatch: leader {version} vs follower {controller_version}"
                            assert versions[0] == controller_versions[0], f"Version mismatch: leader {version} vs follower {controller_version}"
                            for v1, v2 in zip(versions[1:], controller_versions[1:]):
                                if int(v1) < int(v2):
                                    break
                                elif int(v1) > int(v2):
                                    raise ValueError(f"Leader version {version} is newer than follower version {controller_version}")
                        action = msg
                except zmq.Again:
                    action = None
                except Exception as e:
                    print(Fore.RED + f"Error receiving message: {e}" + Style.RESET_ALL)
                    action = None
                    action_server.close()
                    action_server = None
                    print(Fore.YELLOW + "Server socket closed due to error." + Style.RESET_ALL)
        if action is not None:
            time_elapsed = time.perf_counter() - recv_time
            recv_time = time.perf_counter()
            message_fps = 1 / max(time_elapsed, 1e-6)
            with param_lock:
                current_action = action
                current_command_fps = message_fps
        busy_wait(0.001)


def follower_daemon(
    robot: Robot,
    fps: int = 100,
):
    global current_obs, current_action, current_control_fps, attached, observation_server
    print(Fore.GREEN + "Follower daemon thread started." + Style.RESET_ALL)

    loop_start = time.perf_counter()
    current_timestamp = float("-inf")
    while True:
        obs = None
        action = None
        with robot_lock:
            if attached and robot.is_connected:
                try:
                    obs = robot.get_observation()
                except Exception as e:
                    print(Fore.RED + f"Error getting follower observation: {e}" + Style.RESET_ALL)
                    obs = None
                    attached = False
                    print(Fore.YELLOW + "Follower detached due to observation error." + Style.RESET_ALL)
        # print(f"Follower:", obs)

        if obs is not None:
            with param_lock:
                action = current_action
            with observation_server_lock:
                if observation_server is not None:
                    msg = {
                        "version": controller_version,
                        "timestamp": time.time(),
                        "observation": obs,
                    }
                    raw = json.dumps(msg).encode("utf-8")
                    while True:
                        try:
                            req = observation_server.recv_multipart(zmq.NOBLOCK)
                            identity = req[0]
                            assert req[-1].decode("utf-8") == "OBS"
                            observation_server.send_multipart([identity, b"", raw], zmq.NOBLOCK)
                        except zmq.Again:
                            break
                        except Exception as e:
                            print(Fore.RED + f"Error sending observation message: {e}" + Style.RESET_ALL)
                            observation_server.close()
                            observation_server = None
                            print(Fore.YELLOW + "Observation server socket closed due to error." + Style.RESET_ALL)

        if action is not None:
            timestamp = action.get("timestamp", float("-inf"))
            if timestamp > current_timestamp:
                current_timestamp = timestamp
                with robot_lock:
                    if attached:
                        try:
                            # print("[Follower] Sending action:", action)
                            robot.send_action(action['content'])
                        except Exception as e:
                            print(Fore.RED + f"Error sending action to robot: {e}" + Style.RESET_ALL)
                            attached = False
                            print(Fore.YELLOW + "Follower detached due to action sending error." + Style.RESET_ALL)

        dt_s = time.perf_counter() - loop_start
        busy_wait(1 / fps - dt_s)
        loop_s = time.perf_counter() - loop_start
        loop_start = time.perf_counter()
        with param_lock:
            current_obs = obs
            current_control_fps = 1 / max(loop_s, 1e-6)
        # print(f"\ntime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")


def follower_loop(
    robot: Robot,
    fps: int = 100,
    controller_addr: Optional[str] = None,
):
    global attached, action_server, observation_server, action_log

    follower_daemon_thread = Thread(
        target=follower_daemon,
        args=(
            robot,
            fps,
        ),
        daemon=True,
    )
    follower_daemon_thread.start()

    action_server_daemon_thread = Thread(
        target=action_server_daemon,
        daemon=True,
    )
    action_server_daemon_thread.start()

    def connect():
        with robot_lock:
            if robot.is_connected:
                print(Fore.YELLOW + "Follower is already connected." + Style.RESET_ALL)
                return
            robot.connect()
        print("Follower connected.")

    def disconnect():
        with robot_lock:
            if not robot.is_connected:
                print(Fore.YELLOW + "Follower is already disconnected." + Style.RESET_ALL)
                return
            robot.disconnect()
        print("Follower disconnected.")

    def attach():
        global attached
        with robot_lock:
            if attached:
                print(Fore.YELLOW + "Follower is already attached." + Style.RESET_ALL)
                return
            attached = True
        print("Follower attached.")

    def detach():
        global attached
        with robot_lock:
            if not attached:
                print(Fore.YELLOW + "Follower is already detached." + Style.RESET_ALL)
                return
            attached = False
        print("Follower detached.")

    def start_recv():
        global action_server
        if controller_addr is None:
            print(Fore.YELLOW + "No controller address specified." + Style.RESET_ALL)
            return
        addr_parts = controller_addr.split(":")
        port = 7101
        try:
            port = int(addr_parts[-1])
        except ValueError:
            port = 7101
        try:
            with action_server_lock:
                if action_server is not None:
                    print(Fore.YELLOW + "Server is already started." + Style.RESET_ALL)
                    return
                action_server = zmq.Context().socket(zmq.PULL)
                action_server.set_hwm(1)
                # action_server.setsockopt(zmq.RCVTIMEO, 1000)
                action_server.bind(f"tcp://*:{port}")
            print(f"Listening for leader actions on port {port}.")
        except Exception as e:
            print(Fore.RED + f"Error starting action server on port {port}: {e}" + Style.RESET_ALL)
            with action_server_lock:
                action_server = None

    def stop_recv():
        global action_server
        with action_server_lock:
            if action_server is None:
                print(Fore.YELLOW + "Server is not started." + Style.RESET_ALL)
                return
            action_server.close()
            action_server = None
        print("Stopped action server.")

    def start_broadcast():
        global observation_server
        if controller_addr is None:
            print(Fore.YELLOW + "No controller address specified." + Style.RESET_ALL)
            return
        addr_parts = controller_addr.split(":")
        port = 7101
        try:
            port = int(addr_parts[-1])
        except ValueError:
            port = 7101
        port = port + 1
        try:
            with observation_server_lock:
                if observation_server is not None:
                    print(Fore.YELLOW + "Observation server is already started." + Style.RESET_ALL)
                    return
                observation_server = zmq.Context().socket(zmq.ROUTER)
                # observation_server.set_hwm(1)
                observation_server.bind(f"tcp://*:{port}")
            print(f"Broadcasting observations on port {port}.")
        except Exception as e:
            print(Fore.RED + f"Error starting observation server on port {port}: {e}" + Style.RESET_ALL)
            with observation_server_lock:
                observation_server = None

    def stop_broadcast():
        global observation_server
        with observation_server_lock:
            if observation_server is None:
                print(Fore.YELLOW + "Observation server is not started." + Style.RESET_ALL)
                return
            observation_server.close()
            observation_server = None
        print("Stopped observation server.")

    def print_status():
        with param_lock:
            obs = current_obs
            control_fps = current_control_fps
            command_fps = current_command_fps
        with robot_lock:
            status_str = (Fore.GREEN + "CONNECTED" + Style.RESET_ALL) if robot.is_connected else (Fore.YELLOW + "disconnected" + Style.RESET_ALL)
            status_str += ", "
            status_str += (Fore.GREEN + "ATTACHED" + Style.RESET_ALL) if attached else (Fore.YELLOW + "detached" + Style.RESET_ALL)
        with action_server_lock:
            status_str += ", "
            status_str += (Fore.GREEN + "RECEIVING" + Style.RESET_ALL) if action_server is not None else (Fore.YELLOW + "isolated" + Style.RESET_ALL)
        with observation_server_lock:
            status_str += ", "
            status_str += (Fore.GREEN + "BROADCASTING" + Style.RESET_ALL) if observation_server is not None else (Fore.YELLOW + "silent" + Style.RESET_ALL)
        print(f"Status: {status_str}")
        print(f"Observation: {obs}")
        print(f"Action: {current_action}")
        print(f"Control FPS: {control_fps:.2f}")
        print(f"Command FPS: {command_fps:.2f}")
        print(f"Target FPS: {fps:.2f}")
        print(f"Controller Address: {controller_addr}")
        print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
        print(f"Version: {controller_version}")

    while True:
        cmd = input(">>> ").strip()
        if cmd == "exit" or cmd == "quit":
            stop_broadcast()
            stop_recv()
            detach()
            disconnect()
            time.sleep(1)
            print("Exiting leader loop.")
            break
        elif cmd == "start":
            connect()
            attach()
            start_recv()
            start_broadcast()
        elif cmd == "stop":
            stop_broadcast()
            stop_recv()
            detach()
            disconnect()
        elif cmd == "connect":
            connect()
        elif cmd == "disconnect":
            disconnect()
        elif cmd == "start_recv":
            start_recv()
        elif cmd == "stop_recv":
            stop_recv()
        elif cmd == "start_broadcast":
            start_broadcast()
        elif cmd == "stop_broadcast":
            stop_broadcast()
        elif cmd == "attach":
            attach()
        elif cmd == "detach":
            detach()
        elif cmd == "status":
            print_status()
        elif cmd == "watch":
            try:
                while True:
                    # Stop when press q or esc
                    print("\n===== Press Ctrl+C to stop watching =====")
                    print_status()
                    time.sleep(0.5)
            except KeyboardInterrupt:
                print("\nStopped watching status.")
        elif cmd == "help":
            print("Available commands:")
            print("  start           - Connect and attach to follower, receive leader actions")
            print("  stop            - Detach and disconnect from follower, stop receiving leader actions")
            print("  connect         - Connect the follower")
            print("  disconnect      - Disconnect the follower")
            print("  start_recv      - Start receiving leader actions")
            print("  stop_recv       - Stop receiving leader actions")
            print("  start_broadcast - Start broadcasting observations")
            print("  stop_broadcast  - Stop broadcasting observations")
            print("  attach          - Attach follower")
            print("  detach          - Detach follower")
            print("  status          - Show current connection status and FPS")
            print("  watch           - Continuously show status until 'q' or 'esc' is pressed")
            print("  help            - Show this help message")
            print("  exit/quit       - Exit the follower loop")
        elif cmd.strip() == "":
            continue
        else:
            print(Fore.YELLOW + f"Unknown command: {cmd}" + Style.RESET_ALL)


@parser.wrap()
def follower_main(cfg: FollowerConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    if isinstance(cfg.controller_addr, str):
        if cfg.controller_addr.startswith("tcp://"):
            cfg.controller_addr = cfg.controller_addr[6:]

    print("Using config:", cfg)

    robot = make_robot_from_config(cfg.robot)
    try:
        follower_loop(
            robot=robot,
            fps=cfg.fps,
            controller_addr=cfg.controller_addr,
        )
    except KeyboardInterrupt:
        pass


def main():
    follower_main()


if __name__ == "__main__":
    main()
