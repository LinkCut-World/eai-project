import zmq
from colorama import Fore, Style
import lerunbot.envs.lerunbot_env as lerunbot_env
import numpy as np

socket = zmq.Context().socket(zmq.REP)
socket.bind("tcp://*:7301")
env = lerunbot_env.LerunbotEnv(
    control_decimation=20,
    control_frequency=50,
    controller_addr="tcp://localhost:7101",
    controller_version="0.0.1",
    position_server_addr="tcp://localhost:7201",
    skip_steps=-1,
)
env.reset()


def low_controller_loop():
    global socket
    while True:
        try:
            command = socket.recv_string()
        except zmq.Again:
            print(Fore.YELLOW + "Warning: No message received." + Style.RESET_ALL)
            continue
        print(f"Received command: {command}")
        if command == "CAP":
            env.position_server.send_string("CAP")
            image_data = env.position_server.recv_multipart()[-1].decode("utf-8")
            socket.send_string(image_data)
            print(f"Sent captured image data, length: {len(image_data)}.")
            continue
        if command == "FORWARD":
            positions = lerunbot_env.DEMO_FORWARD
        elif command == "TURN_LEFT":
            positions = lerunbot_env.DEMO_TURN_LEFT
        elif command == "TURN_RIGHT":
            positions = lerunbot_env.DEMO_TURN_RIGHT
        elif command == "STOP":
            positions = None
        else:
            print(Fore.RED + "Error: Unknown command." + Style.RESET_ALL)
            socket.send_string("ERROR: Unknown command.")
            continue
        if positions is not None:
            actions = []
            for step in positions:
                action = np.array(
                    [
                        step["shoulder_pan.pos"],
                        step["shoulder_lift.pos"],
                        step["elbow_flex.pos"],
                        step["wrist_flex.pos"],
                        step["wrist_roll.pos"],
                        step["gripper.pos"],
                    ]
                )
                actions.append(action)
            # obs, reward, terminated, done, info = env.step(actions)
            expected_arms, true_arms = env.step_(actions)
            print(f"Executed actions for command: {command}")
        env.reset()
        print(f"Sending OK response for command: {command}")
        socket.send_string("OK")


if __name__ == "__main__":
    print("Running low-level controller...")
    low_controller_loop()
