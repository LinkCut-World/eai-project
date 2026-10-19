import cv2
import numpy as np
import match
import argparse
import zmq
import json
import base64
from colorama import Fore, Style

cap = None
calibration = None
socket = None


def parse_args():
    parser = argparse.ArgumentParser(description="Test relative position estimation between two images.")
    parser.add_argument("--position-server-addr", type=str, default=None, help="Address of the position server.")
    parser.add_argument("--device", type=str, default="/dev/video0", help="Camera device path.")
    parser.add_argument("--height", type=int, default=480, help="Camera frame height.")
    parser.add_argument("--width", type=int, default=640, help="Camera frame width.")
    parser.add_argument("--fps", type=int, default=10, help="Camera frames per second.")
    # parser.add_argument("--calibration-file", type=str, default=None, help="Path to camera calibration file.")
    parser.add_argument("--calibration-file", type=str, default="camera_calibration.npy", help="Path to camera calibration file.")
    parser.add_argument("--num-features", type=int, default=5000, help="Number of features to detect.")
    parser.add_argument("--ref-point", type=float, nargs=3, default=[0.10, 0.00, 0.16], help="Reference point in camera1 coordinate system (x y z).")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode to save match visualization.")
    return parser.parse_args()


def safe_send(identity, msg):
    global socket
    try:
        socket.send_multipart([identity, b"", msg.encode("utf-8")])
    except zmq.Again:
        print(Fore.YELLOW + "Warning: Failed to send message." + Style.RESET_ALL)


def main():
    global cap, calibration, socket
    args = parse_args()
    if args.calibration_file is not None:
        calibration = np.load(args.calibration_file, allow_pickle=True).item()

    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        print(Fore.RED + "Error: Could not open video." + Style.RESET_ALL)
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    print("Warming up the camera...")
    for _ in range(30):
        ret, frame = cap.read()  # Warm up the camera
    print("Camera is ready.")
    socket = zmq.Context().socket(zmq.ROUTER)

    port = 7201
    if args.position_server_addr is not None:
        try:
            port = int(args.position_server_addr.split(":")[-1])
        except ValueError:
            port = 7201
    socket.bind(f"tcp://*:{port}")
    print(f"Position server listening on port {port}.")

    frame1 = None
    frame2 = None
    while True:
        try:
            parts = socket.recv_multipart()
            identity = parts[0]
            message = parts[-1].decode("utf-8")
        except zmq.Again:
            print(Fore.YELLOW + "Warning: No message received." + Style.RESET_ALL)
            continue

        if message == "CAP":
            print("Received CAP request.")
            for _ in range(5):
                ret, frame = cap.read()
            if not ret:
                print(Fore.RED + "Error: Could not read frame." + Style.RESET_ALL)
                safe_send(identity, "ERROR")
                continue
            _, buffer = cv2.imencode(".jpg", frame)
            image_base64 = base64.b64encode(buffer).decode("utf-8")
            data_uri = f"data:image/jpeg;base64,{image_base64}"
            safe_send(identity, data_uri)
            # Write image to file if debug mode
            if args.debug:
                with open("debug_captured_image.jpg", "wb") as f:
                    f.write(buffer)
                print("Saved captured image to debug_captured_image.jpg.")
        elif message == "POS1":
            print("Received POS1 request.")
            # if frame1 is not None:
            #     print(Fore.YELLOW + "Warning: Previous frame1 not used before new POS1. Resetting all." + Style.RESET_ALL)
            #     frame1 = None
            #     frame2 = None
            for _ in range(5):
                ret, frame1 = cap.read()
            if not ret:
                print(Fore.RED + "Error: Could not read frame." + Style.RESET_ALL)
                safe_send(identity, "ERROR")
                frame1 = None
                # frame2 = None
                continue
            safe_send(identity, "OK")
        elif message == "POS2":
            print("Received POS2 request.")
            if frame1 is None:
                print(Fore.RED + "Error: No POS1 received before POS2." + Style.RESET_ALL)
                safe_send(identity, "ERROR")
                # frame1 = None
                # frame2 = None
                continue
            for _ in range(5):
                ret, frame2 = cap.read()
            if not ret:
                print(Fore.RED + "Error: Could not read frame." + Style.RESET_ALL)
                safe_send(identity, "ERROR")
                # frame1 = None
                frame2 = None
                continue
            try:
                suc, t_vec, quat = match.relative_position(
                    frame1,
                    frame2,
                    calibration,
                    num_features=args.num_features,
                    debug=args.debug,
                    ref_point=args.ref_point,
                )
                response = {
                    "success": suc,
                    "translation": t_vec.tolist(),
                    "rotation": quat.tolist(),
                }
                safe_send(identity, json.dumps(response))
                print("Sent position response:", response)
            except Exception as e:
                print(Fore.RED + f"Error computing relative position: {e}" + Style.RESET_ALL)
                safe_send(identity, "ERROR")
            # frame1 = None
            # frame2 = None


if __name__ == "__main__":
    main()
