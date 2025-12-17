---
sidebar_position: 1
---

# Chapter 01: ROS 2 Fundamentals and Installation

## Understanding the Robot Operating System

The Robot Operating System (ROS) is not actually an operating system, but rather a flexible framework for writing robot software. It provides services designed for a heterogeneous computer cluster such as hardware abstraction, device drivers, libraries, visualizers, message-passing, package management, and more.

ROS 2 is the next generation of this framework, designed to address the limitations of ROS 1 and to enable the development of production-level robotic applications.

### Why ROS 2 for Humanoid Robotics?

Humanoid robots require complex coordination between multiple subsystems including:
- Perception systems (cameras, LIDAR, IMU)
- Actuation systems (motors, servos)
- Control systems (balance, gait)
- AI systems (planning, reasoning)

ROS 2 provides the middleware architecture necessary to coordinate these systems effectively.

### Key Concepts in ROS 2

#### Nodes
A node is an executable that uses ROS 2 to communicate with other nodes. In a humanoid robot, you might have nodes for:
- Sensor data processing
- Motion planning
- Balance control
- Speech recognition

#### Topics
Topics are named buses over which nodes exchange messages. For example:
- `/joint_states` - current positions of robot joints
- `/cmd_vel` - velocity commands for navigation
- `/camera/image_raw` - raw image data from cameras

#### Services
Services provide a request/response communication pattern. For example:
- `/get_plan` - request a navigation plan
- `/set_parameters` - configure robot parameters

#### Actions
Actions are like services but support long-running tasks with feedback. For example:
- `/follow_joint_trajectory` - move robot joints to specific positions over time

### Installing ROS 2

For humanoid robotics, we recommend using the latest LTS (Long Term Support) version of ROS 2. At the time of writing, this is Humble Hawksbill (Ubuntu 22.04).

#### Ubuntu Installation

1. Set locale:
```bash
locale  # check for UTF-8
sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8
```

2. Add ROS 2 apt repository:
```bash
sudo apt update && sudo apt install curl gnupg lsb-release
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | sudo gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
```

3. Install ROS 2 packages:
```bash
sudo apt update
sudo apt install ros-humble-desktop
```

4. Install colcon build system:
```bash
sudo apt install python3-colcon-common-extensions
```

#### Alternative Installation Methods

For other platforms or development workflows, see the official ROS 2 installation guide. Docker containers are also available for consistent development environments.

### Setting Up Your Workspace

Create a workspace for your humanoid robot projects:

```bash
mkdir -p ~/humanoid_ws/src
cd ~/humanoid_ws
```

### Environment Setup

Add the following to your `~/.bashrc` or `~/.zshrc`:

```bash
source /opt/ros/humble/setup.bash
```

### First ROS 2 Node

Let's create a simple node to test our installation:

```python
# hello_robot.py
import rclpy
from rclpy.node import Node

class HelloRobotNode(Node):
    def __init__(self):
        super().__init__('hello_robot')
        self.get_logger().info('Hello from the Humanoid Robot!')

def main(args=None):
    rclpy.init(args=args)
    node = HelloRobotNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

Run this with:
```bash
python3 hello_robot.py
```

### Summary

In this chapter, we've covered:
- The fundamentals of ROS 2 and why it's essential for humanoid robotics
- Key concepts: nodes, topics, services, and actions
- Installation of ROS 2 Humble Hawksbill
- Setting up a development workspace
- Creating your first ROS 2 node

In the next chapter, we'll dive deeper into nodes, topics, and services, learning how to create more complex communication patterns between different parts of your humanoid robot.