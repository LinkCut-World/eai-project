from flask import Flask, render_template_string, jsonify, request
import base64
from io import BytesIO
from PIL import Image
import argparse
import zmq
from colorama import Fore, Style
import time
import math

args = None


def parse_args():
    parser = argparse.ArgumentParser(description="Flask Web Application with Joystick")
    parser.add_argument("--low-controller-addr", type=str, default="tcp://localhost:7301", help="Address of the low-level controller.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host for the Flask app.")
    parser.add_argument("--port", type=int, default=5000, help="Port for the Flask app.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    parser.add_argument("--angle-limit", type=float, default=30.0, help="Joystick angle limit (in degrees).")
    parser.add_argument("--xy-scale", type=float, default=0.2, help="Scale factor for X/Y joystick (in meters).")
    parser.add_argument("--yaw-scale", type=float, default=30.0, help="Scale factor for Yaw joystick (in degrees).")
    return parser.parse_args()


template_string = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WebUI Control</title>
    <style>
        body {
            display: flex;
            flex-direction: column;
            align-items: center;
            font-family: Arial, sans-serif;
            user-select: none;
        }
        #joystick-container {
            position: relative;
            width: 300px;
            height: 300px;
            background: conic-gradient(#f0f0f0 0deg {{ angle_limit }}deg, #ccc {{ angle_limit }}deg {{ 360 - angle_limit }}deg, #f0f0f0 {{ 360 - angle_limit }}deg 360deg);
            border-radius: 50%;
            border: 2px solid #ccc;
            touch-action: none;
        }
        #joystick-knob {
            position: absolute;
            width: 50px;
            height: 50px;
            background-color: #333;
            border-radius: 50%;
            top: 125px;
            left: 125px;
            cursor: pointer;
            pointer-events: none; /* Let events pass to container */
        }
        #controls {
            margin-top: 20px;
            text-align: center;
        }
        #yaw-joystick-container {
            position: relative;
            width: 300px;
            height: 50px;
            background-color: #f0f0f0;
            border-radius: 25px;
            border: 2px solid #ccc;
            touch-action: none;
        }
        #yaw-joystick-knob {
            position: absolute;
            width: 50px;
            height: 50px;
            background-color: #333;
            border-radius: 50%;
            top: 0px;
            left: 125px;
            cursor: pointer;
            pointer-events: none;
        }
    </style>
</head>
<body>
    <h1>LeRunBot</h1>
    
    <img id="image-box" src="{{image}}" width="640" height="480" />
    
    <div id="joysticks-wrapper" style="display: flex; flex-direction: row; gap: 100px; align-items: center; margin-top: 20px;">
        <div id="joystick-container">
            <div id="joystick-knob"></div>
        </div>
        
        <div id="yaw-joystick-container">
            <div id="yaw-joystick-knob"></div>
        </div>
    </div>
    
    <div id="controls">
        <p>Current Command: <span id="cmd-display">(0.00, 0.00)</span></p>
        <input type="text" id="manual-command" placeholder="Enter manual command">
        <button onclick="sendManual()">Send Manual</button>
        <button onclick="sendReset()">Reset</button>
        <p id="response-text">Response: [No response]</p>
    </div>
        
    <script>
        let joystickX = 0;
        let joystickY = 0;
        let joystickYaw = 0;
        const container = document.getElementById('joystick-container');
        const knob = document.getElementById('joystick-knob');
        const cmdDisplay = document.getElementById('cmd-display');
        
        const containerSize = 300;
        const knobSize = 50;
        const maxDist = (containerSize / 2) - (knobSize / 2);
        const centerX = containerSize / 2;
        const centerY = containerSize / 2;
        
        let isDragging = false;

        function updateJoystick(clientX, clientY) {
            const rect = container.getBoundingClientRect();
            let dx = clientX - rect.left - centerX;
            let dy = clientY - rect.top - centerY;
            
            let distance = Math.sqrt(dx*dx + dy*dy);
            
            // Calculate angle relative to UP (0 deg)
            // atan2(dx, -dy) gives angle from UP axis, clockwise positive, range -PI to PI
            let angle = Math.atan2(dx, -dy);
            const limit = {{ angle_limit }} * Math.PI / 180;
            
            // Clamp angle to [-{{ angle_limit }}, {{ angle_limit }}] degrees
            if (angle > limit) angle = limit;
            if (angle < -limit) angle = -limit;
            
            // Re-calculate dx, dy based on clamped angle
            // x = sin(angle) * dist, y = -cos(angle) * dist (since y is down)
            dx = Math.sin(angle) * distance;
            dy = -Math.cos(angle) * distance;

            if (distance > maxDist) {
                const ratio = maxDist / distance;
                dx *= ratio;
                dy *= ratio;
            }
            
            knob.style.transform = `translate(${dx}px, ${dy}px)`;
            
            // Normalize to -1..1
            // Y is inverted (up is negative in screen, positive in joystick)
            joystickX = dx / maxDist * {{ xy_scale }};
            joystickY = -dy / maxDist * {{ xy_scale }}; 
            
            updateDisplay();
        }

        function resetJoystick() {
            isDragging = false;
            joystickX = 0;
            joystickY = 0;
            knob.style.transform = `translate(0px, 0px)`;
            updateDisplay();
        }
        
        function updateDisplay() {
             cmdDisplay.innerText = `(${joystickY.toFixed(2)},${(-joystickX).toFixed(2)},${joystickYaw.toFixed(2)})`;
        }

        // Mouse Events
        container.addEventListener('mousedown', (e) => { 
            isDragging = true; 
            updateJoystick(e.clientX, e.clientY);
        });
        document.addEventListener('mousemove', (e) => { 
            if(isDragging) updateJoystick(e.clientX, e.clientY); 
        });
        document.addEventListener('mouseup', () => { 
            if(isDragging) resetJoystick(); 
        });

        // Touch Events
        let joystickTouchId = null;
        container.addEventListener('touchstart', (e) => { 
            e.preventDefault();
            if (joystickTouchId !== null) return;
            const touch = e.changedTouches[0];
            joystickTouchId = touch.identifier;
            isDragging = true; 
            updateJoystick(touch.clientX, touch.clientY);
        }, {passive: false});
        
        container.addEventListener('touchmove', (e) => { 
            e.preventDefault();
            if (!isDragging) return;
            for (let i = 0; i < e.changedTouches.length; i++) {
                if (e.changedTouches[i].identifier === joystickTouchId) {
                    updateJoystick(e.changedTouches[i].clientX, e.changedTouches[i].clientY);
                    break;
                }
            }
        }, {passive: false});
        
        const endJoystick = (e) => {
            e.preventDefault();
            for (let i = 0; i < e.changedTouches.length; i++) {
                if (e.changedTouches[i].identifier === joystickTouchId) {
                    resetJoystick();
                    joystickTouchId = null;
                    break;
                }
            }
        };
        container.addEventListener('touchend', endJoystick);
        container.addEventListener('touchcancel', endJoystick);

        const yawContainer = document.getElementById('yaw-joystick-container');
        const yawKnob = document.getElementById('yaw-joystick-knob');
        const yawMaxDist = 125; // (300 - 50) / 2
        const yawCenterX = 150; // 300 / 2
        let isDraggingYaw = false;

        function updateYawJoystick(clientX) {
            const rect = yawContainer.getBoundingClientRect();
            let dx = clientX - rect.left - yawCenterX;
            
            if (dx > yawMaxDist) dx = yawMaxDist;
            if (dx < -yawMaxDist) dx = -yawMaxDist;
            
            yawKnob.style.transform = `translateX(${dx}px)`;
            
            // Map to -1..1. 
            // Left (negative dx) is Positive Yaw (CCW), Right (positive dx) is Negative Yaw (CW).
            joystickYaw = -dx / yawMaxDist * {{ yaw_scale }}; 
            
            updateDisplay();
        }

        function resetYawJoystick() {
            isDraggingYaw = false;
            joystickYaw = 0;
            yawKnob.style.transform = `translateX(0px)`;
            updateDisplay();
        }

        // Yaw Mouse Events
        yawContainer.addEventListener('mousedown', (e) => { 
            isDraggingYaw = true; 
            updateYawJoystick(e.clientX);
        });
        document.addEventListener('mousemove', (e) => { 
            if(isDraggingYaw) updateYawJoystick(e.clientX); 
        });
        document.addEventListener('mouseup', () => { 
            if(isDraggingYaw) resetYawJoystick(); 
        });

        // Yaw Touch Events
        let yawTouchId = null;
        yawContainer.addEventListener('touchstart', (e) => { 
            e.preventDefault();
            if (yawTouchId !== null) return;
            const touch = e.changedTouches[0];
            yawTouchId = touch.identifier;
            isDraggingYaw = true; 
            updateYawJoystick(touch.clientX);
        }, {passive: false});
        
        yawContainer.addEventListener('touchmove', (e) => { 
            e.preventDefault();
            if (!isDraggingYaw) return;
            for (let i = 0; i < e.changedTouches.length; i++) {
                if (e.changedTouches[i].identifier === yawTouchId) {
                    updateYawJoystick(e.changedTouches[i].clientX);
                    break;
                }
            }
        }, {passive: false});
        
        const endYawJoystick = (e) => {
            e.preventDefault();
            for (let i = 0; i < e.changedTouches.length; i++) {
                if (e.changedTouches[i].identifier === yawTouchId) {
                    resetYawJoystick();
                    yawTouchId = null;
                    break;
                }
            }
        };
        yawContainer.addEventListener('touchend', endYawJoystick);
        yawContainer.addEventListener('touchcancel', endYawJoystick);

        
        let manualCmd = null;

        function sendCommand(cmd) {
            return fetch(`/control?command=${encodeURIComponent(cmd)}`, { method: 'GET' })
                .then(response => response.json())
                .then(data => {
                    document.getElementById('image-box').src = data.image;
                    document.getElementById('response-text').innerText = `Response: ${data.response}`;
                });
        }

        function sendManual() {
            const cmd = document.getElementById('manual-command').value;
            manualCmd = cmd;
            console.log("Manual command queued");
        }

        function sendReset() {
            manualCmd = "RESET";
            console.log("Reset command queued");
        }

        // Continuous Control Loop
        function controlLoop() {
            let cmd;
            if (manualCmd) {
                cmd = manualCmd;
                manualCmd = null;
            } else {
                cmd = `(${joystickY.toFixed(4)},${(-joystickX).toFixed(4)},${joystickYaw.toFixed(4)})`;
            }
            
            sendCommand(cmd)
            .catch(err => {
                console.error("Error in control loop:", err);
                // Add a small delay on error to avoid hammering
                return new Promise(resolve => setTimeout(resolve, 500));
            })
            .finally(() => {
                // Send next command immediately
                controlLoop();
            });
        }

        // Start the loop
        controlLoop();
    </script>
</body>
</html>
"""

ctrl_socket = None
app = Flask(__name__)

img = Image.new('RGB', (640, 480), color=(0, 255, 0))
buffered = BytesIO()
img.save(buffered, format="JPEG")
bad_image = "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode('utf-8')


def get_image():
    try:
        # print("Requesting image from position server...")
        ctrl_socket.send_string("CAP")
        image_data = ctrl_socket.recv_string()
        if not image_data.startswith("data:image"):
            print(Fore.RED + "Error: Invalid image data received from position server." + Style.RESET_ALL)
            return bad_image
        # print("Received image data from position server.")
    except zmq.Again:
        print(Fore.YELLOW + "Warning: No response from position server." + Style.RESET_ALL)
        return bad_image
    except Exception as e:
        print(Fore.RED + f"Error getting image from position server: {e}" + Style.RESET_ALL)
        return bad_image
    return image_data


@app.route('/')
def index():
    image_data = get_image()
    return render_template_string(template_string, image=image_data,
                                  angle_limit=args.angle_limit,
                                  xy_scale=args.xy_scale,
                                  yaw_scale=args.yaw_scale * math.pi / 180.0)


@app.route('/control')
def control():
    cmd: str = request.args.get('command', '(0,0,0)')

    should_send = True
    try:
        # Parse command to check norm
        # cmd format: "(x,y)"
        content = cmd.strip("()")
        parts = content.split(',')
        if len(parts) >= 2:
            x = float(parts[0])
            y = float(parts[1])
            yaw_offset = 0.0
            if len(parts) > 2:
                yaw_offset = float(parts[2])

            if math.sqrt(x * x + y * y) < 0.05 and abs(yaw_offset) < 0.05:
                should_send = False
            else:
                should_send = True
                yaw = math.atan2(y, x) + yaw_offset
                cmd = f"[{x:.4f},{y:.4f},0,{yaw:.4f}]"
    except ValueError:
        pass

    if should_send:
        try:
            print(f"Sending command to low-level controller: {cmd}")
            ctrl_socket.send_string(cmd)
            ctrl_response = ctrl_socket.recv_string()
            print(f"Controller response: {ctrl_response}")
        except zmq.Again:
            print(Fore.YELLOW + "Warning: No response from low-level controller." + Style.RESET_ALL)
            ctrl_response = "[No response]"
        except Exception as e:
            print(Fore.RED + f"Error communicating with low-level controller: {e}" + Style.RESET_ALL)
            ctrl_response = "[Error]"
    else:
        ctrl_response = "Ignored (deadzone)"

    img_str = get_image()
    return jsonify(image=img_str, response=ctrl_response)


def main():
    global template_string, ctrl_socket, args
    args = parse_args()
    low_controller_addr = args.low_controller_addr
    if not low_controller_addr.startswith("tcp://"):
        low_controller_addr = "tcp://" + low_controller_addr
    try:
        _ = int(low_controller_addr.split(":")[-1])
    except ValueError:
        low_controller_addr = low_controller_addr + ":7301"

    context = zmq.Context()
    ctrl_socket = context.socket(zmq.REQ)
    # Set timeouts to prevent hanging
    # ctrl_socket.setsockopt(zmq.RCVTIMEO, 2000)
    # ctrl_socket.setsockopt(zmq.SNDTIMEO, 2000)

    ctrl_socket.connect(low_controller_addr)
    print(f"Connected to low-level controller at {low_controller_addr}.")

    time.sleep(1)  # Reduced sleep

    print("Starting Flask web application...")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == '__main__':
    main()
