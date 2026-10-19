#!/bin/bash

# Start the position_server session
tmux new-session -d -s position_server  # Create the session
tmux send-keys -t position_server "cd lerunbot/camera" C-m
tmux send-keys -t position_server "conda activate lerobot" C-m
tmux send-keys -t position_server "python position_server.py --position-server-addr 'tcp://*:7201' --debug" C-m
echo "position_server session started."
sleep 3s

# Start the so101_controller session
tmux new-session -d -s so101_controller  # Create the session
tmux send-keys -t so101_controller "cd lerunbot/so101_controller" C-m
tmux send-keys -t so101_controller "conda activate lerobot" C-m
tmux send-keys -t so101_controller "python -m follower --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_awesome_follower_arm --controller_addr=localhost:7101 --robot.use_degrees=True" C-m
echo "so101_controller session started."
sleep 3s

# Start the low_controller session
tmux new-session -d -s low_controller  # Create the session
tmux send-keys -t low_controller "cd lerunbot" C-m
tmux send-keys -t low_controller "conda activate lerobot" C-m
tmux send-keys -t low_controller "python example_low_controller.py" C-m
echo "low_controller session started."
sleep 3s

# Start the vlm_control session
tmux new-session -d -s vlm_control  # Create the session
tmux send-keys -t vlm_control "cd vlm_control" C-m
tmux send-keys -t vlm_control "conda activate lerobot" C-m
tmux send-keys -t vlm_control "python vlm_control.py --model-name 'GLM-4.5V' --prompt-file ./example_prompt.txt --task-file ../tasks/move_to_the_bottle.txt --verbose" C-m
echo "vlm_control session started."
sleep 3s

echo "All tmux sessions started."

# sleep 3s
# tmux attach -t so101_controller #Attach to the so101_controller session
