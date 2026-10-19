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
tmux send-keys -t so101_controller "python -m follower --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_awesome_follower_arm --robot.use_degrees=True --controller_addr=localhost:7101" C-m
echo "so101_controller session started."
sleep 3s

# Start the manual_controller session
tmux new-session -d -s manual_controller  # Create the session
tmux send-keys -t manual_controller "cd lerunbot" C-m
tmux send-keys -t manual_controller "conda activate lerobot" C-m
tmux send-keys -t manual_controller "python manual_controller.py" C-m
echo "manual_controller session started."
sleep 3s

echo "All tmux sessions started."

# sleep 3s
# tmux attach -t so101_controller  # Attach to the so101_controller session
