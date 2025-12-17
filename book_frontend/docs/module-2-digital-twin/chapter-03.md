---
sidebar_position: 3
---

# Chapter 03: Creating Humanoid Robot Models for Simulation

## From URDF to Simulation-Ready Models

In this chapter, we'll transform the URDF humanoid model from Module 1 into a simulation-ready model for Gazebo. This involves adding physics properties, sensors, and controllers that enable realistic simulation of humanoid robots.

### Simulation-Specific Considerations

When preparing a URDF model for simulation, several additional elements are required beyond what's needed for visualization:

1. **Physics Properties**: Accurate mass, inertia, and collision geometry
2. **Sensors**: Simulated cameras, IMUs, force/torque sensors
3. **Actuators**: Joint controllers and motor models
4. **Gazebo Plugins**: Specialized plugins for simulation behavior

### Enhanced URDF for Simulation

Building on our basic humanoid model from Module 1, let's add simulation-specific elements:

```xml
<?xml version="1.0"?>
<robot name="sim_humanoid" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- Materials -->
  <material name="white">
    <color rgba="1.0 1.0 1.0 1.0"/>
  </material>
  <material name="black">
    <color rgba="0.0 0.0 0.0 1.0"/>
  </material>
  <material name="blue">
    <color rgba="0.0 0.0 0.8 1.0"/>
  </material>

  <!-- Gazebo-specific elements -->
  <gazebo>
    <plugin filename="gz-sim-model-plugin-system" name="gz::sim::systems::ModelPlugin">
      <filename>libgazebo_ros_init.so</filename>
      <ros>
        <namespace>/humanoid_robot</namespace>
      </ros>
    </plugin>
  </gazebo>

  <!-- Torso with enhanced properties -->
  <link name="torso">
    <visual>
      <geometry>
        <capsule length="0.6" radius="0.15"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <geometry>
        <capsule length="0.6" radius="0.15"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="15.0"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.8" ixy="0.0" ixz="0.0" iyy="0.8" iyz="0.0" izz="0.3"/>
    </inertial>
  </link>

  <gazebo reference="torso">
    <material>Gazebo/White</material>
    <mu1>0.2</mu1>
    <mu2>0.2</mu2>
    <self_collide>false</self_collide>
  </gazebo>

  <!-- Head with sensors -->
  <link name="head">
    <visual>
      <geometry>
        <sphere radius="0.15"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.15"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.02"/>
    </inertial>
  </link>

  <joint name="neck_joint" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0 0 0.45"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="10" velocity="1.0"/>
    <dynamics damping="0.5" friction="0.1"/>
  </joint>

  <gazebo reference="neck_joint">
    <implicitSpringDamper>1</implicitSpringDamper>
  </gazebo>

  <!-- Camera in head -->
  <joint name="camera_joint" type="fixed">
    <parent link="head"/>
    <child link="camera_frame"/>
    <origin xyz="0.1 0 0" rpy="0 0 0"/>
  </joint>

  <link name="camera_frame">
    <inertial>
      <mass value="0.01"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
  </link>

  <!-- Left Arm with actuators -->
  <link name="left_upper_arm">
    <visual>
      <geometry>
        <capsule length="0.3" radius="0.05"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <geometry>
        <capsule length="0.3" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 -0.15"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="left_shoulder_joint" type="revolute">
    <parent link="torso"/>
    <child link="left_upper_arm"/>
    <origin xyz="0.1 0.15 0.3"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.35" upper="1.57" effort="50" velocity="2.0"/>
    <dynamics damping="1.0" friction="0.2"/>
  </joint>

  <link name="left_lower_arm">
    <visual>
      <geometry>
        <capsule length="0.3" radius="0.04"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <geometry>
        <capsule length="0.3" radius="0.04"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.8"/>
      <origin xyz="0 0 -0.15"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="left_elbow_joint" type="revolute">
    <parent link="left_upper_arm"/>
    <child link="left_lower_arm"/>
    <origin xyz="0 0 -0.3"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="2.35" effort="30" velocity="2.0"/>
    <dynamics damping="0.8" friction="0.1"/>
  </joint>

  <!-- Left Hand -->
  <link name="left_hand">
    <visual>
      <geometry>
        <box size="0.1 0.08 0.05"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.1 0.08 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.3"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="left_wrist_joint" type="revolute">
    <parent link="left_lower_arm"/>
    <child link="left_hand"/>
    <origin xyz="0 0 -0.3"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1.57" upper="1.57" effort="10" velocity="1.0"/>
    <dynamics damping="0.3" friction="0.05"/>
  </joint>

  <!-- Right Arm (mirror of left) -->
  <link name="right_upper_arm">
    <visual>
      <geometry>
        <capsule length="0.3" radius="0.05"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <geometry>
        <capsule length="0.3" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 -0.15"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="right_shoulder_joint" type="revolute">
    <parent link="torso"/>
    <child link="right_upper_arm"/>
    <origin xyz="0.1 -0.15 0.3"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="2.35" effort="50" velocity="2.0"/>
    <dynamics damping="1.0" friction="0.2"/>
  </joint>

  <link name="right_lower_arm">
    <visual>
      <geometry>
        <capsule length="0.3" radius="0.04"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <geometry>
        <capsule length="0.3" radius="0.04"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.8"/>
      <origin xyz="0 0 -0.15"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="right_elbow_joint" type="revolute">
    <parent link="right_upper_arm"/>
    <child link="right_lower_arm"/>
    <origin xyz="0 0 -0.3"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="2.35" effort="30" velocity="2.0"/>
    <dynamics damping="0.8" friction="0.1"/>
  </joint>

  <link name="right_hand">
    <visual>
      <geometry>
        <box size="0.1 0.08 0.05"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.1 0.08 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.3"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="right_wrist_joint" type="revolute">
    <parent link="right_lower_arm"/>
    <child link="right_hand"/>
    <origin xyz="0 0 -0.3"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1.57" upper="1.57" effort="10" velocity="1.0"/>
    <dynamics damping="0.3" friction="0.05"/>
  </joint>

  <!-- Left Leg with enhanced properties -->
  <link name="left_upper_leg">
    <visual>
      <geometry>
        <capsule length="0.4" radius="0.06"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <geometry>
        <capsule length="0.4" radius="0.06"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.5"/>
      <origin xyz="0 0 -0.2"/>
      <inertia ixx="0.04" ixy="0.0" ixz="0.0" iyy="0.04" iyz="0.0" izz="0.002"/>
    </inertial>
  </link>

  <joint name="left_hip_joint" type="revolute">
    <parent link="torso"/>
    <child link="left_upper_leg"/>
    <origin xyz="-0.05 0.1 -0.1"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1.5"/>
    <dynamics damping="2.0" friction="0.5"/>
  </joint>

  <link name="left_lower_leg">
    <visual>
      <geometry>
        <capsule length="0.4" radius="0.05"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <geometry>
        <capsule length="0.4" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <origin xyz="0 0 -0.2"/>
      <inertia ixx="0.03" ixy="0.0" ixz="0.0" iyy="0.03" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="left_knee_joint" type="revolute">
    <parent link="left_upper_leg"/>
    <child link="left_lower_leg"/>
    <origin xyz="0 0 -0.4"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="2.35" effort="100" velocity="1.5"/>
    <dynamics damping="1.8" friction="0.4"/>
  </joint>

  <link name="left_foot">
    <visual>
      <geometry>
        <box size="0.2 0.1 0.05"/>
      </geometry>
      <material name="black"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.2 0.1 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.8"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.002"/>
    </inertial>
  </link>

  <joint name="left_ankle_joint" type="revolute">
    <parent link="left_lower_leg"/>
    <child link="left_foot"/>
    <origin xyz="0 0 -0.4"/>
    <axis xyz="1 0 0"/>
    <limit lower="-0.5" upper="0.5" effort="30" velocity="1.0"/>
    <dynamics damping="0.5" friction="0.1"/>
  </joint>

  <!-- Right Leg (mirror of left) -->
  <link name="right_upper_leg">
    <visual>
      <geometry>
        <capsule length="0.4" radius="0.06"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <geometry>
        <capsule length="0.4" radius="0.06"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.5"/>
      <origin xyz="0 0 -0.2"/>
      <inertia ixx="0.04" ixy="0.0" ixz="0.0" iyy="0.04" iyz="0.0" izz="0.002"/>
    </inertial>
  </link>

  <joint name="right_hip_joint" type="revolute">
    <parent link="torso"/>
    <child link="right_upper_leg"/>
    <origin xyz="-0.05 -0.1 -0.1"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1.5"/>
    <dynamics damping="2.0" friction="0.5"/>
  </joint>

  <link name="right_lower_leg">
    <visual>
      <geometry>
        <capsule length="0.4" radius="0.05"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <geometry>
        <capsule length="0.4" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <origin xyz="0 0 -0.2"/>
      <inertia ixx="0.03" ixy="0.0" ixz="0.0" iyy="0.03" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="right_knee_joint" type="revolute">
    <parent link="right_upper_leg"/>
    <child link="right_lower_leg"/>
    <origin xyz="0 0 -0.4"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="2.35" effort="100" velocity="1.5"/>
    <dynamics damping="1.8" friction="0.4"/>
  </joint>

  <link name="right_foot">
    <visual>
      <geometry>
        <box size="0.2 0.1 0.05"/>
      </geometry>
      <material name="black"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.2 0.1 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.8"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.002"/>
    </inertial>
  </link>

  <joint name="right_ankle_joint" type="revolute">
    <parent link="right_lower_leg"/>
    <child link="right_foot"/>
    <origin xyz="0 0 -0.4"/>
    <axis xyz="1 0 0"/>
    <limit lower="-0.5" upper="0.5" effort="30" velocity="1.0"/>
    <dynamics damping="0.5" friction="0.1"/>
  </joint>

  <!-- Add sensors to the robot -->

  <!-- IMU in torso -->
  <gazebo reference="torso">
    <sensor name="imu_sensor" type="imu">
      <always_on>true</always_on>
      <update_rate>100</update_rate>
      <visualize>false</visualize>
      <imu>
        <angular_velocity>
          <x>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>2e-4</stddev>
            </noise>
          </x>
          <y>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>2e-4</stddev>
            </noise>
          </y>
          <z>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>2e-4</stddev>
            </noise>
          </z>
        </angular_velocity>
        <linear_acceleration>
          <x>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>1.7e-2</stddev>
            </noise>
          </x>
          <y>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>1.7e-2</stddev>
            </noise>
          </y>
          <z>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>1.7e-2</stddev>
            </noise>
          </z>
        </linear_acceleration>
      </imu>
    </sensor>
  </gazebo>

  <!-- Camera in head -->
  <gazebo reference="camera_frame">
    <sensor name="camera" type="camera">
      <always_on>true</always_on>
      <update_rate>30</update_rate>
      <camera name="head">
        <horizontal_fov>1.3962634</horizontal_fov> <!-- 80 degrees -->
        <image>
          <width>640</width>
          <height>480</height>
          <format>R8G8B8</format>
        </image>
        <clip>
          <near>0.1</near>
          <far>10</far>
        </clip>
        <noise>
          <type>gaussian</type>
          <mean>0.0</mean>
          <stddev>0.007</stddev>
        </noise>
      </camera>
      <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Camera">
        <topic_name>/humanoid_robot/camera/image_raw</topic_name>
        <camera_name>camera</camera_name>
        <frame_name>camera_frame</frame_name>
        <update_rate>30</update_rate>
      </plugin>
    </sensor>
  </gazebo>

  <!-- Force/Torque sensors in feet -->
  <gazebo reference="left_foot">
    <sensor name="left_foot_force_torque" type="force_torque">
      <always_on>true</always_on>
      <update_rate>100</update_rate>
      <force_torque>
        <frame>child</frame>
        <measure_direction>child_to_parent</measure_direction>
      </force_torque>
    </sensor>
  </gazebo>

  <gazebo reference="right_foot">
    <sensor name="right_foot_force_torque" type="force_torque">
      <always_on>true</always_on>
      <update_rate>100</update_rate>
      <force_torque>
        <frame>child</frame>
        <measure_direction>child_to_parent</measure_direction>
      </force_torque>
    </sensor>
  </gazebo>

  <!-- Joint state publisher plugin -->
  <gazebo>
    <plugin filename="gz-sim-joint-state-publisher-system" name="gz::sim::systems::JointStatePublisher">
      <ros>
        <namespace>/humanoid_robot</namespace>
        <remapping>~/out:=joint_states</remapping>
      </ros>
      <update_rate>50</update_rate>
    </plugin>
  </gazebo>

  <!-- ROS2 Control interface -->
  <ros2_control name="GazeboSystem" type="system">
    <hardware>
      <plugin>gazebo_ros2_control/GazeboSystem</plugin>
    </hardware>
    <joint name="left_shoulder_joint">
      <command_interface name="position">
        <param name="min">-2.35</param>
        <param name="max">1.57</param>
      </command_interface>
      <state_interface name="position"/>
      <state_interface name="velocity"/>
    </joint>
    <joint name="left_elbow_joint">
      <command_interface name="position">
        <param name="min">0</param>
        <param name="max">2.35</param>
      </command_interface>
      <state_interface name="position"/>
      <state_interface name="velocity"/>
    </joint>
    <joint name="right_shoulder_joint">
      <command_interface name="position">
        <param name="min">-1.57</param>
        <param name="max">2.35</param>
      </command_interface>
      <state_interface name="position"/>
      <state_interface name="velocity"/>
    </joint>
    <joint name="right_elbow_joint">
      <command_interface name="position">
        <param name="min">0</param>
        <param name="max">2.35</param>
      </command_interface>
      <state_interface name="position"/>
      <state_interface name="velocity"/>
    </joint>
    <joint name="left_hip_joint">
      <command_interface name="position">
        <param name="min">-1.57</param>
        <param name="max">1.57</param>
      </command_interface>
      <state_interface name="position"/>
      <state_interface name="velocity"/>
    </joint>
    <joint name="left_knee_joint">
      <command_interface name="position">
        <param name="min">0</param>
        <param name="max">2.35</param>
      </command_interface>
      <state_interface name="position"/>
      <state_interface name="velocity"/>
    </joint>
    <joint name="right_hip_joint">
      <command_interface name="position">
        <param name="min">-1.57</param>
        <param name="max">1.57</param>
      </command_interface>
      <state_interface name="position"/>
      <state_interface name="velocity"/>
    </joint>
    <joint name="right_knee_joint">
      <command_interface name="position">
        <param name="min">0</param>
        <param name="max">2.35</param>
      </command_interface>
      <state_interface name="position"/>
      <state_interface name="velocity"/>
    </joint>
  </ros2_control>

</robot>
```

### Adding ROS 2 Control for Simulation

To enable proper joint control in simulation, we need to set up ros2_control. Create a configuration file:

```yaml
# config/humanoid_control.yaml
controller_manager:
  ros__parameters:
    update_rate: 100  # Hz

    left_arm_controller:
      type: joint_trajectory_controller/JointTrajectoryController

    right_arm_controller:
      type: joint_trajectory_controller/JointTrajectoryController

    leg_controller:
      type: joint_trajectory_controller/JointTrajectoryController

    joint_state_broadcaster:
      type: joint_state_broadcaster/JointStateBroadcaster

left_arm_controller:
  ros__parameters:
    joints:
      - left_shoulder_joint
      - left_elbow_joint
    command_interfaces:
      - position
    state_interfaces:
      - position
      - velocity

right_arm_controller:
  ros__parameters:
    joints:
      - right_shoulder_joint
      - right_elbow_joint
    command_interfaces:
      - position
    state_interfaces:
      - position
      - velocity

leg_controller:
  ros__parameters:
    joints:
      - left_hip_joint
      - left_knee_joint
      - right_hip_joint
      - right_knee_joint
    command_interfaces:
      - position
    state_interfaces:
      - position
      - velocity

joint_state_broadcaster:
  ros__parameters:
    type: joint_state_broadcaster/JointStateBroadcaster
```

### Physics Parameter Tuning

Realistic physics simulation requires careful tuning of parameters:

#### Friction and Damping
```xml
<gazebo reference="left_foot">
  <mu1>0.8</mu1>  <!-- Static friction coefficient -->
  <mu2>0.8</mu2>  <!-- Dynamic friction coefficient -->
  <fdir1>1 0 0</fdir1>  <!-- Friction direction -->
  <kp>1000000.0</kp>    <!-- Contact stiffness -->
  <kd>100.0</kd>        <!-- Contact damping -->
  <max_vel>100.0</max_vel>      <!-- Maximum contact correction velocity -->
  <min_depth>0.001</min_depth>  <!-- Depth of interpenetration -->
</gazebo>
```

#### Inertia Considerations
For humanoid robots, accurate inertial properties are crucial for realistic simulation:

- Mass values should reflect real robot components
- Center of mass should be correctly positioned
- Inertia tensors should be physically plausible

### Sensor Integration

Humanoid robots typically use multiple sensor types:

#### Vision Sensors
```xml
<sensor name="stereo_camera" type="multicamera">
  <always_on>true</always_on>
  <update_rate>30</update_rate>
  <camera name="left">
    <horizontal_fov>1.3962634</horizontal_fov>
    <image>
      <width>640</width>
      <height>480</height>
    </image>
    <clip>
      <near>0.1</near>
      <far>10</far>
    </clip>
  </camera>
  <camera name="right">
    <pose>0.06 0 0 0 0 0</pose>  <!-- 6cm baseline -->
    <horizontal_fov>1.3962634</horizontal_fov>
    <image>
      <width>640</width>
      <height>480</height>
    </image>
    <clip>
      <near>0.1</near>
      <far>10</far>
    </clip>
  </camera>
</sensor>
```

#### Range Sensors (LiDAR/Depth)
```xml
<sensor name="depth_camera" type="depth_camera">
  <always_on>true</always_on>
  <update_rate>15</update_rate>
  <camera>
    <horizontal_fov>1.047</horizontal_fov>  <!-- 60 degrees -->
    <image>
      <width>320</width>
      <height>240</height>
    </image>
    <clip>
      <near>0.1</near>
      <far>5</far>
    </clip>
  </camera>
</sensor>
```

### Validation and Testing

Before using your simulation model, validate it:

1. **Visual Check**: Load in RViz2 to verify joint ranges and visual appearance
2. **Physics Check**: Load in Gazebo to ensure robot doesn't fall apart
3. **Control Check**: Verify all joints can be controlled properly
4. **Sensor Check**: Confirm sensors publish data on expected topics

### Loading and Testing Your Model

To load your simulation-ready humanoid model:

```bash
# Method 1: Spawn URDF directly
ros2 launch gazebo_ros spawn_entity.py -entity humanoid_robot -file path/to/your/sim_humanoid.urdf -x 0 -y 0 -z 1.0

# Method 2: Use a launch file
# launch/humanoid_sim.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    pkg_gazebo_ros = FindPackageShare('gazebo_ros')
    pkg_humanoid_description = FindPackageShare('humanoid_description')

    world = LaunchConfiguration('world')

    world_arg = DeclareLaunchArgument(
        'world',
        default_value='empty.sdf',
        description='Choose one of the world files from `/gazebo_ros/worlds`'
    )

    # Launch Gazebo with empty world
    gzserver_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_gazebo_ros, 'launch', 'gzserver.launch.py'])
        ),
        launch_arguments={'world': world}.items()
    )

    gzclient_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_gazebo_ros, 'launch', 'gzclient.launch.py'])
        )
    )

    # Spawn robot in gazebo
    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'humanoid_robot',
            '-file', PathJoinSubstitution([pkg_humanoid_description, 'urdf', 'sim_humanoid.urdf']),
            '-x', '0', '-y', '0', '-z', '1.0'
        ],
        output='screen'
    )

    return LaunchDescription([
        world_arg,
        gzserver_launch,
        gzclient_launch,
        spawn_robot,
    ])
```

### Common Issues and Solutions

1. **Robot falls apart**: Check joint limits and physics properties
2. **Joints don't move**: Verify ROS 2 control configuration
3. **Sensors not publishing**: Check Gazebo plugins and namespaces
4. **Simulation is unstable**: Adjust physics parameters and solver settings

### Summary

In this chapter, we've covered:
- Enhancing URDF models with simulation-specific elements
- Adding sensors (IMU, cameras, force/torque) to the robot
- Configuring ROS 2 Control for joint actuation
- Tuning physics parameters for realistic behavior
- Validating simulation-ready models
- Loading and testing models in Gazebo

With your simulation-ready humanoid model, you're ready to move on to Chapter 4 where we'll explore physics-based environment design for your humanoid robots.