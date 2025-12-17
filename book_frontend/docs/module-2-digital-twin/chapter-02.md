---
sidebar_position: 2
---

# Chapter 02: Gazebo Simulation Fundamentals

## Setting Up Your First Robot Simulation

In this chapter, we'll explore the fundamentals of Gazebo simulation and set up our first robot simulation environment. Gazebo (now part of Ignition Robotics) is a powerful physics simulator that's widely used in robotics research and development.

### Understanding Gazebo Architecture

Gazebo follows a client-server architecture:

- **Gazebo Server (gzserver)**: Runs the physics simulation, handles plugins, and manages the world state
- **Gazebo Client (gzclient)**: Provides the visual interface for users to view and interact with the simulation
- **Plugins**: Extend Gazebo functionality for sensors, controllers, and custom behaviors

### Installing Gazebo

For this module, we'll use Gazebo Garden (the latest version compatible with ROS 2 Humble):

```bash
# Add the osrfoundation repository
sudo apt update && sudo apt install wget
sudo sh -c 'echo "deb [arch=amd64] http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list'
wget http://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -

# Install Gazebo Garden
sudo apt update
sudo apt install gz-garden

# Install ROS 2 Gazebo integration
sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-gazebo-ros-control ros-humble-ros-gz-sim
```

### Launching Gazebo

You can launch Gazebo in different ways:

#### Standalone Gazebo
```bash
# Launch Gazebo with an empty world
gz sim

# Launch with a specific world file
gz sim -r empty.sdf
```

#### Through ROS 2
```bash
# Using ros_gz_sim package
ros2 launch ros_gz_sim gz_sim.launch.py
```

### Creating Your First World

Gazebo uses SDF (Simulation Description Format) to define worlds. Here's a simple world file:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="simple_world">
    <!-- Include a default ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Include a default sun -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Add a simple box obstacle -->
    <model name="box">
      <pose>2 2 0.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
          <material>
            <ambient>1 0 0 1</ambient>
            <diffuse>1 0 0 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>1.0</mass>
          <inertia>
            <ixx>0.166667</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.166667</iyy>
            <iyz>0</iyz>
            <izz>0.166667</izz>
          </inertia>
        </inertial>
      </link>
    </model>
  </world>
</sdf>
```

Save this as `simple_world.sdf` and launch it with:
```bash
gz sim -r simple_world.sdf
```

### Integrating with ROS 2

To use Gazebo with ROS 2, you'll typically create a launch file:

```python
# launch/simple_sim.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declare arguments
    declared_arguments = []
    declared_arguments.append(
        DeclareLaunchArgument(
            "world",
            default_value="empty.sdf",
            description="Choose one of the world files from `/gz_sim/worlds`",
        )
    )

    # Launch configuration variables
    world = LaunchConfiguration("world")

    # Nodes
    gz_sim = Node(
        package='ros_gz_sim',
        executable='gz_sim',
        arguments=['-r', '-v', '4', PathJoinSubstitution([FindPackageShare('my_robot_sim'), 'worlds', world])],
        output='screen',
    )

    # Bridge to connect Gazebo and ROS 2
    gz_ros_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        parameters=[{
            'services': [
                '/world/default/control',
                '/world/default/create',
                '/world/default/delete',
            ],
            'topics': [
                ['/model/robot_name/odometry', 'nav_msgs/msg/Odometry'],
                ['/model/robot_name/cmd_vel', 'geometry_msgs/msg/Twist'],
            ]
        }],
        output='screen'
    )

    nodes = [
        gz_sim,
        gz_ros_bridge,
    ]

    return LaunchDescription(declared_arguments + nodes)
```

### Working with Models in Gazebo

Gazebo models can be created in several ways:

#### 1. Using Built-in Models
Gazebo includes many built-in models like:
- ground_plane
- sun
- shapes (box, sphere, cylinder)
- simple robots (pr2, turtlebot3, etc.)

```xml
<include>
  <uri>model://ground_plane</uri>
</include>
```

#### 2. Custom URDF Models
You can spawn URDF models directly into Gazebo:

```bash
# Spawn a URDF model into Gazebo
ros2 run gazebo_ros spawn_entity.py -entity my_robot -file /path/to/robot.urdf -z 1.0
```

#### 3. SDF Models
Create complete SDF models with physics properties:

```xml
<sdf version="1.7">
  <model name="simple_robot">
    <link name="chassis">
      <pose>0 0 0.1 0 0 0</pose>
      <collision name="collision">
        <geometry>
          <box>
            <size>0.5 0.3 0.2</size>
          </box>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <box>
            <size>0.5 0.3 0.2</size>
          </box>
        </geometry>
        <material>
          <ambient>0.8 0.8 0.8 1</ambient>
          <diffuse>0.8 0.8 0.8 1</diffuse>
        </material>
      </visual>
      <inertial>
        <mass>5.0</mass>
        <inertia>
          <ixx>0.1</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>0.2</iyy>
          <iyz>0</iyz>
          <izz>0.2</izz>
        </inertia>
      </inertial>
    </link>
  </model>
</sdf>
```

### Gazebo Plugins

Plugins extend Gazebo's functionality. Common plugins for humanoid robots include:

#### Sensor Plugins
```xml
<sensor name="camera" type="camera">
  <camera name="head">
    <horizontal_fov>1.047</horizontal_fov>
    <image>
      <width>640</width>
      <height>480</height>
    </image>
    <clip>
      <near>0.1</near>
      <far>10</far>
    </clip>
  </camera>
  <always_on>1</always_on>
  <update_rate>30</update_rate>
  <visualize>true</visualize>
</sensor>
```

#### Joint Control Plugins
```xml
<plugin filename="gz-sim-joint-position-controller-system" name="gz::sim::systems::JointPositionController">
  <joint_name>joint1</joint_name>
</plugin>
```

### Running Your First Simulation

Let's create a complete example that combines everything:

1. Create a simple robot URDF file:

```xml
<?xml version="1.0"?>
<robot name="simple_wheeled_robot">
  <material name="blue">
    <color rgba="0.0 0.0 0.8 1.0"/>
  </material>

  <link name="base_link">
    <visual>
      <geometry>
        <cylinder length="0.2" radius="0.2"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.2" radius="0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <link name="wheel_left">
    <visual>
      <geometry>
        <cylinder length="0.05" radius="0.1"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.05" radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.2"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <link name="wheel_right">
    <visual>
      <geometry>
        <cylinder length="0.05" radius="0.1"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.05" radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.2"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="left_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_left"/>
    <origin xyz="0 0.15 -0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>

  <joint name="right_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_right"/>
    <origin xyz="0 -0.15 -0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>
</robot>
```

2. Create a world file with this robot:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="robot_world">
    <include>
      <uri>model://ground_plane</uri>
    </include>
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Simple box obstacle -->
    <model name="obstacle">
      <pose>2 0 0.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.4 0.1 1</ambient>
            <diffuse>0.8 0.4 0.1 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>10.0</mass>
          <inertia>
            <ixx>1</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>1</iyy>
            <iyz>0</iyz>
            <izz>1</izz>
          </inertia>
        </inertial>
      </link>
    </model>
  </world>
</sdf>
```

### Gazebo Command Line Tools

Gazebo provides several command-line tools for simulation control:

```bash
# List all topics
gz topic -l

# Echo a topic
gz topic -e -t /world/default/model/robot_name/pose

# Publish to a topic
gz topic -t /world/default/model/robot_name/cmd_vel -m ignition.msgs.Twist -p 'linear: {x: 1.0}, angular: {z: 0.5}'

# Run simulation for a specific time
gz sim -r -t 1000 simple_world.sdf  # Run for 1000ms
```

### Troubleshooting Common Issues

1. **Simulation is too slow**: Reduce physics complexity or use faster physics engine parameters
2. **Robot falls through the ground**: Check collision geometry and physics parameters
3. **Sensors not publishing**: Verify plugin configuration and topic names
4. **Controllers not working**: Check joint names and plugin configuration

### Summary

In this chapter, we've covered:
- Gazebo architecture and components
- Installation and basic usage
- Creating world files with SDF
- Integrating Gazebo with ROS 2
- Working with different types of models
- Using plugins for sensors and control
- Running your first complete simulation
- Command-line tools for simulation control

In the next chapter, we'll explore how to create humanoid robot models specifically for simulation, building on the URDF knowledge from Module 1.