controller_version = "0.0.1"
controller_versions = controller_version.split(".")
CHECK_VERSION = True

import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat

from lerobot.configs import parser
from lerobot.teleoperators import (
    Teleoperator,
    TeleoperatorConfig,
    make_teleoperator_from_config,
)
from lerobot.teleoperators.so101_leader import SO101LeaderConfig
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
class LeaderConfig:
    teleop: TeleoperatorConfig
    fps: int = 100
    controller_addr: Optional[str] = None


teleop_lock = Lock()
attached = False

param_lock = Lock()
current_obs: dict | None = None
current_fps = 0

client_lock = Lock()
client: zmq.Socket | None = None
loss_rate = 0.0
LOSS_RATE_DECAY = 0.99


def leader_daemon(
    teleop: Teleoperator,
    fps: int = 100,
):
    global current_obs, current_fps, attached, client, loss_rate
    print(Fore.GREEN + "Leader daemon thread started." + Style.RESET_ALL)

    loop_start = time.perf_counter()
    while True:
        with teleop_lock:
            if attached and teleop.is_connected:
                try:
                    obs = teleop.get_action()
                except Exception as e:
                    print(Fore.RED + f"Error getting leader action: {e}" + Style.RESET_ALL)
                    obs = None
                    attached = False
                    print(Fore.YELLOW + "Leader detached due to observation error." + Style.RESET_ALL)
            else:
                obs = None
        # print(f"Leader:", obs)
        if obs is not None:
            with client_lock:
                if client is not None:
                    msg = {
                        "version": controller_version,
                        "timestamp": time.time(),
                        "content": obs,
                    }
                    try:
                        client.send(json.dumps(msg).encode("utf-8"), zmq.NOBLOCK)
                        loss_rate = loss_rate * LOSS_RATE_DECAY + 0.0 * (1 - LOSS_RATE_DECAY)
                    except zmq.Again:
                        loss_rate = loss_rate * LOSS_RATE_DECAY + 1.0 * (1 - LOSS_RATE_DECAY)
                    except Exception as e:
                        print(Fore.RED + f"Error sending leader action: {e}" + Style.RESET_ALL)
                        loss_rate = 0.0
                        client.close()
                        client = None
                        print(Fore.YELLOW + "Leader socket closed due to sending error." + Style.RESET_ALL)
        dt_s = time.perf_counter() - loop_start
        busy_wait(1 / fps - dt_s)
        loop_s = time.perf_counter() - loop_start
        loop_start = time.perf_counter()
        with param_lock:
            current_obs = obs
            current_fps = 1 / max(loop_s, 1e-6)
        # print(f"\ntime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")


def leader_loop(
    teleop: Teleoperator,
    fps: int = 100,
    controller_addr: Optional[str] = None,
):
    global attached, client, loss_rate
    daemon_thread = Thread(
        target=leader_daemon,
        args=(
            teleop,
            fps,
        ),
        daemon=True,
    )
    daemon_thread.start()

    def connect():
        with teleop_lock:
            if teleop.is_connected:
                print(Fore.YELLOW + "Leader is already connected." + Style.RESET_ALL)
                return
            teleop.connect()
        print("Leader connected.")

    def disconnect():
        with teleop_lock:
            if not teleop.is_connected:
                print(Fore.YELLOW + "Leader is already disconnected." + Style.RESET_ALL)
                return
            teleop.disconnect()
        print("Leader disconnected.")

    def attach():
        global attached
        with teleop_lock:
            if attached:
                print(Fore.YELLOW + "Leader is already attached." + Style.RESET_ALL)
                return
            attached = True
        print("Leader attached.")

    def detach():
        global attached
        with teleop_lock:
            if not attached:
                print(Fore.YELLOW + "Leader is already detached." + Style.RESET_ALL)
                return
            attached = False
        print("Leader detached.")

    def start_sending():
        global client
        if controller_addr is None:
            print(Fore.YELLOW + "No controller address specified." + Style.RESET_ALL)
            return
        addr_parts = controller_addr.split(":")
        host = addr_parts[0]
        port = 7101 if len(addr_parts) == 1 else int(addr_parts[1])
        try:
            with client_lock:
                if client is not None:
                    print(Fore.YELLOW + "Already connected to controller address." + Style.RESET_ALL)
                    return
                client = zmq.Context().socket(zmq.PUSH)
                client.set_hwm(1)
                client.connect(f"tcp://{host}:{port}")
            print(f"Connected to {controller_addr} for pushing leader actions.")
        except Exception as e:
            print(Fore.RED + f"Error connecting to {controller_addr}: {e}" + Style.RESET_ALL)
            with client_lock:
                client = None

    def stop_sending():
        global client, loss_rate
        with client_lock:
            if client is None:
                print(Fore.YELLOW + "Not connected to controller address." + Style.RESET_ALL)
                return
            client.close()
            client = None
            loss_rate = 0.0
        print("Disconnected from controller address.")

    while True:
        cmd = input(">>> ").strip()
        if cmd == "exit" or cmd == "quit":
            stop_sending()
            detach()
            disconnect()
            time.sleep(1)
            print("Exiting leader loop.")
            break
        elif cmd == "start":
            connect()
            attach()
            start_sending()
        elif cmd == "stop":
            stop_sending()
            detach()
            disconnect()
        elif cmd == "connect":
            connect()
        elif cmd == "disconnect":
            disconnect()
        elif cmd == "start_sending":
            start_sending()
        elif cmd == "stop_sending":
            stop_sending()
        elif cmd == "attach":
            attach()
        elif cmd == "detach":
            detach()
        elif cmd == "status":
            with param_lock:
                obs = current_obs
                cur_fps = current_fps
            with teleop_lock:
                status_str = (Fore.GREEN + "CONNECTED" + Style.RESET_ALL) if teleop.is_connected else (Fore.YELLOW + "disconnected" + Style.RESET_ALL)
                status_str += ", "
                status_str += (Fore.GREEN + "ATTACHED" + Style.RESET_ALL) if attached else (Fore.YELLOW + "detached" + Style.RESET_ALL)
            with client_lock:
                status_str += ", "
                status_str += (Fore.GREEN + "SENDING" + Style.RESET_ALL) if client is not None else (Fore.YELLOW + "isolated" + Style.RESET_ALL)
            print(f"Status: {status_str}")
            print(f"Observation: {obs}")
            print(f"Loss rate: {loss_rate * 100:.2f}%")
            print(f"FPS: {cur_fps:.2f}")
            print(f"Target FPS: {fps:.2f}")
            print(f"Controller Address: {controller_addr}")
        elif cmd == "help":
            print("Available commands:")
            print("  start         - Connect and attach to leader, send leader actions")
            print("  stop          - Detach and disconnect from leader, stop sending leader actions")
            print("  connect       - Connect the leader")
            print("  disconnect    - Disconnect the leader")
            print("  start_sending - Start sending leader actions")
            print("  stop_sending  - Stop sending leader actions")
            print("  attach        - Attach leader")
            print("  detach        - Detach leader")
            print("  status        - Show current connection status and FPS")
            print("  help          - Show this help message")
            print("  exit/quit     - Exit the leader loop")
        elif cmd.strip() == "":
            continue
        else:
            print(Fore.YELLOW + f"Unknown command: {cmd}" + Style.RESET_ALL)


@parser.wrap()
def leader_main(cfg: LeaderConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    if isinstance(cfg.controller_addr, str):
        if cfg.controller_addr.startswith("tcp://"):
            cfg.controller_addr = cfg.controller_addr[6:]

    print("Using config:", cfg)

    teleop = make_teleoperator_from_config(cfg.teleop)
    try:
        leader_loop(
            teleop=teleop,
            fps=cfg.fps,
            controller_addr=cfg.controller_addr,
        )
    except KeyboardInterrupt:
        pass


def main():
    leader_main()


if __name__ == "__main__":
    main()
