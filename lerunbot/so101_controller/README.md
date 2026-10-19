Action ranges are scaled to [-100, 100].

Usage:

```bash
python -m leader --teleop.type=so101_leader --teleop.port=/dev/ttyACM0 --teleop.id=my_awesome_leader_arm --teleop.use_degrees True --controller_addr=183.172.56.145:7101
```

```bash
python -m follower --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_awesome_follower_arm  --robot.use_degrees True --controller_addr=localhost:7101
```
