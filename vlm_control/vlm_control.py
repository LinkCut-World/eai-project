import os
import dotenv
import copy
import time
import random
import zmq
from openai import OpenAI
import argparse
from colorama import Fore, Style


MAX_COMMAND_HISTORY = 50
MAX_IMAGE_HISTORY = 4
MAX_OPENAI_RETRIES = 10


openai_service = None
ctrl_socket = None
prompt = None
historical_commands = []
historical_images = []


def parse_args():
    parser = argparse.ArgumentParser(description="VLM Control Module")
    parser.add_argument("--low-controller-addr", type=str, default="tcp://localhost:7301", help="Address of the low-level controller.")
    parser.add_argument("--openai-api-key", type=str, default=None, help="OpenAI API Key.")
    parser.add_argument("--openai-api-base", type=str, default=None, help="OpenAI API Base URL.")
    parser.add_argument("--model-name", type=str, default="gpt-4v", help="Name of the OpenAI model to use.")
    parser.add_argument("--prompt-file", type=str, default="vlm_control/prompt.txt", help="Path to the prompt file.")
    parser.add_argument("--task-file", type=str, default=None, help="Path to the task file.")
    parser.add_argument("--max-command-history", type=int, default=MAX_COMMAND_HISTORY, help="Maximum number of historical commands to keep.")
    parser.add_argument("--max-image-history", type=int, default=MAX_IMAGE_HISTORY, help="Maximum number of historical images to keep.")
    parser.add_argument("--max-openai-retries", type=int, default=MAX_OPENAI_RETRIES, help="Maximum number of retries for OpenAI API requests.")
    parser.add_argument("--stepping-mode", action="store_true", help="Enable stepping mode.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose mode.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    return parser.parse_args()


def get_openai_response(messages, model_name):
    global openai_service
    for attempt in range(MAX_OPENAI_RETRIES):
        try:
            response = openai_service.chat.completions.create(
                model=model_name,
                messages=[{
                    "role": "user",
                    "content": messages
                }]
            )
            reply = response.choices[0].message.content
            if "CMD[" in reply:
                cmd = reply.split('CMD[')[-1].split(']')[0].strip()
            else:
                cmd = "FAILED_TO_PARSE_COMMAND"
            if "LOG[" in reply:
                log = reply.split('LOG[')[-1].split(']')[0].strip()
            else:
                log = ""
            return cmd, log, reply
        except Exception as e:
            if attempt == MAX_OPENAI_RETRIES - 1:
                raise
            delay = 2 ** attempt + random.random()
            print(Fore.YELLOW + f"Request failed: {e}. Retrying in {delay} seconds..." + Style.RESET_ALL)
            time.sleep(delay)
    raise RuntimeError("Failed to get response from OpenAI after multiple attempts.")


def main():
    global ctrl_socket, openai_service, prompt, historical_commands, historical_images
    global MAX_COMMAND_HISTORY, MAX_IMAGE_HISTORY, MAX_OPENAI_RETRIES
    args = parse_args()
    MAX_COMMAND_HISTORY = args.max_command_history
    MAX_IMAGE_HISTORY = args.max_image_history
    MAX_OPENAI_RETRIES = args.max_openai_retries
    dotenv.load_dotenv()

    openai_api_key = args.openai_api_key or os.getenv("OPENAI_API_KEY")
    openai_api_base = args.openai_api_base or os.getenv("OPENAI_API_BASE")

    openai_service = OpenAI(api_key=openai_api_key, base_url=openai_api_base)

    with open(args.prompt_file, "r") as f:
        prompt_text = f.read()
    if args.task_file is not None:
        with open(args.task_file, "r") as f:
            task_text = f.read()
        prompt_text = prompt_text.replace("{{task_description}}", task_text)
    prompt = [
        {"type": "text", "text": prompt_text.split('{{visual_input}}')[0]},
        {"type": "image_url", "image_url": None},
        {"type": "text", "text": prompt_text.split('{{visual_input}}')[1].split('{{historical_images}}')[0]},
        {"type": "text", "text": prompt_text.split('{{historical_images}}')[1]},
    ]

    low_controller_addr = args.low_controller_addr
    if not low_controller_addr.startswith("tcp://"):
        low_controller_addr = "tcp://" + low_controller_addr
    try:
        _ = int(low_controller_addr.split(":")[-1])
    except ValueError:
        low_controller_addr = low_controller_addr + ":7301"
    ctrl_socket = zmq.Context().socket(zmq.REQ)
    ctrl_socket.connect(low_controller_addr)
    print(f"Connected to low-level controller at {low_controller_addr}.")

    print("VLM Control Module is running...")
    paused = True
    while True:
        if args.stepping_mode or paused:
            _ = input("Press Enter to continue...")
        paused = False

        try:
            print("Requesting image from position server...")
            ctrl_socket.send_string("CAP")
            image_data = ctrl_socket.recv_string()
            if not image_data.startswith("data:image"):
                print(Fore.RED + "Error: Invalid image data received from position server." + Style.RESET_ALL)
                paused = True
                continue
            print("Received image data from position server.")
        except zmq.Again:
            print(Fore.YELLOW + "Warning: No response from position server." + Style.RESET_ALL)
            paused = True
            continue

        print("Generating command using VLM...")
        messages = copy.deepcopy(prompt)
        messages[1]["image_url"] = {"url": image_data}
        messages[0]["text"] = messages[0]["text"].replace("{{historical_commands}}", "\n".join(historical_commands))
        for i in range(1, len(historical_images) + 1):
            messages.insert(3, {"type": "text", "text": f"Captured image at {i} steps before:"})
            messages.insert(4, {"type": "image_url", "image_url": {"url": historical_images[-i]}})
        if args.verbose:
            output_messages = []
            for msg in messages:
                if msg["type"] == "text":
                    output_messages.append({'type': 'text', 'content': msg['text'][:30]})
                elif msg["type"] == "image_url":
                    output_messages.append({'type': 'image', 'content': msg['image_url']['url'][:30]})
            print("  Full messages:", Fore.LIGHTBLACK_EX + str(output_messages) + Style.RESET_ALL)
        cmd, log, reply = get_openai_response(messages, args.model_name)
        print(f"Generated Command: {cmd} | Log: {log}")
        if args.verbose or args.debug:
            print("  Full Response:", Fore.LIGHTBLACK_EX + reply + Style.RESET_ALL)
            if args.debug:
                _ = input("Press Enter to send command to low-level controller...")

        try:
            print("Sending command to low-level controller...")
            ctrl_socket.send_string(cmd)
            ctrl_response = ctrl_socket.recv_string()
            print(f"Sent command to low-level controller: {cmd}")
            print(f"Controller response: {ctrl_response}")
            historical_commands.append(
                f"Command: [{cmd}] | Log: [{log}] | Response: [{ctrl_response}]" if log else
                f"Command: [{cmd}] | Response: [{ctrl_response}]"
            )
            historical_commands = historical_commands[-MAX_COMMAND_HISTORY:]
            historical_images.append(image_data)
            historical_images = historical_images[-MAX_IMAGE_HISTORY:]
        except zmq.Again:
            print(Fore.YELLOW + "Warning: No response from low-level controller." + Style.RESET_ALL)
            paused = True
            continue

        if "STOP" in cmd:
            print("Received STOP command. Pausing execution.")
            paused = True


if __name__ == "__main__":
    main()
