---
sidebar_position: 5
---

# Chapter 05: Sensor Simulation and Integration

## Emulating Real-World Perception in Simulation

In this final chapter of Module 2, we'll explore sensor simulation for humanoid robots. Accurate sensor simulation is crucial for developing perception algorithms that can transfer from simulation to reality. We'll cover the simulation of various sensor types including cameras, LiDAR, IMU, and force/torque sensors.

### Understanding Sensor Simulation

Sensor simulation in Gazebo involves creating virtual sensors that produce data similar to their real-world counterparts. This requires:
- Accurate geometric and physical models of sensors
- Realistic noise models
- Proper integration with ROS 2 topics
- Calibration parameters matching real sensors

### Camera Simulation

Cameras are essential for humanoid robots for navigation, object recognition, and interaction. Let's explore camera simulation in detail:

#### Basic Camera Setup
```xml
<sensor name="rgb_camera" type="camera">
  <always_on>true</always_on>
  <update_rate>30</update_rate>
  <camera name="head_camera">
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
  <visualize>true</visualize>
</sensor>
```

#### Advanced Camera Configuration with Plugins
```xml
<gazebo reference="camera_frame">
  <sensor name="camera" type="camera">
    <always_on>true</always_on>
    <update_rate>30</update_rate>
    <camera name="head">
      <horizontal_fov>1.3962634</horizontal_fov>
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
```

#### Stereo Camera Setup
```xml
<sensor name="stereo_camera" type="multicamera">
  <always_on>true</always_on>
  <update_rate>30</update_rate>
  <camera name="left">
    <horizontal_fov>1.047</horizontal_fov> <!-- 60 degrees -->
    <image>
      <width>640</width>
      <height>480</height>
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
  <camera name="right">
    <pose>0.06 0 0 0 0 0</pose> <!-- 6cm baseline -->
    <horizontal_fov>1.047</horizontal_fov>
    <image>
      <width>640</width>
      <height>480</height>
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
  <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::MultiCamera">
    <left_topic_name>/humanoid_robot/stereo_camera/left/image_raw</left_topic_name>
    <right_topic_name>/humanoid_robot/stereo_camera/right/image_raw</right_topic_name>
    <frame_name>stereo_camera_left</frame_name>
  </plugin>
</sensor>
```

#### Depth Camera Simulation
```xml
<sensor name="depth_camera" type="depth_camera">
  <always_on>true</always_on>
  <update_rate>15</update_rate>
  <camera name="depth">
    <horizontal_fov>1.047</horizontal_fov> <!-- 60 degrees -->
    <image>
      <width>320</width>
      <height>240</height>
    </image>
    <clip>
      <near>0.1</near>
      <far>5</far>
    </clip>
  </camera>
  <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::DepthCamera">
    <topic_name>/humanoid_robot/depth_camera/image_raw</topic_name>
    <frame_name>depth_camera_frame</frame_name>
  </plugin>
</sensor>
```

### LiDAR and Range Sensor Simulation

LiDAR sensors are crucial for navigation and mapping:

#### 2D LiDAR (Planar Scanner)
```xml
<sensor name="laser_2d" type="gpu_lidar" xmlns:gz="http://gazebosim.org/schemas/sdf/1.6">
  <gz:plugin name="lidar_2d" filename="gz-sim-sensors-system">
    <gz:topic>/humanoid_robot/laser_scan</gz:topic>
  </gz:plugin>
  <always_on>true</always_on>
  <update_rate>10</update_rate>
  <visualize>false</visualize>
  <topic>/laser_scan</topic>
  <ray>
    <scan>
      <horizontal>
        <samples>720</samples>
        <resolution>1</resolution>
        <min_angle>-1.570796</min_angle> <!-- -90 degrees -->
        <max_angle>1.570796</max_angle>   <!-- 90 degrees -->
      </horizontal>
    </scan>
    <range>
      <min>0.1</min>
      <max>30.0</max>
      <resolution>0.01</resolution>
    </range>
  </ray>
</sensor>
```

#### 3D LiDAR (Velodyne-style)
```xml
<sensor name="velodyne_vlp16" type="gpu_lidar" xmlns:gz="http://gazebosim.org/schemas/sdf/1.6">
  <gz:plugin name="velodyne_vlp16" filename="gz-sim-sensors-system">
    <gz:topic>/humanoid_robot/velodyne_points</gz:topic>
  </gz:plugin>
  <always_on>true</always_on>
  <update_rate>10</update_rate>
  <visualize>false</visualize>
  <topic>/velodyne_points</topic>
  <ray>
    <scan>
      <horizontal>
        <samples>1800</samples>
        <resolution>1</resolution>
        <min_angle>-3.14159</min_angle> <!-- -180 degrees -->
        <max_angle>3.14159</max_angle>   <!-- 180 degrees -->
      </horizontal>
      <vertical>
        <samples>16</samples>
        <resolution>1</resolution>
        <min_angle>-0.261799</min_angle> <!-- -15 degrees -->
        <max_angle>0.261799</max_angle>   <!-- 15 degrees -->
      </vertical>
    </scan>
    <range>
      <min>0.1</min>
      <max>100.0</max>
      <resolution>0.01</resolution>
    </range>
  </ray>
</sensor>
```

### IMU Simulation

IMUs provide crucial orientation and acceleration data for humanoid robots:

```xml
<sensor name="imu_sensor" type="imu">
  <always_on>true</always_on>
  <update_rate>100</update_rate>
  <visualize>false</visualize>
  <topic>/humanoid_robot/imu</topic>
  <imu>
    <angular_velocity>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
          <bias_mean>0.0000075</bias_mean>
          <bias_stddev>0.0000008</bias_stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
          <bias_mean>0.0000075</bias_mean>
          <bias_stddev>0.0000008</bias_stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
          <bias_mean>0.0000075</bias_mean>
          <bias_stddev>0.0000008</bias_stddev>
        </noise>
      </z>
    </angular_velocity>
    <linear_acceleration>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
          <bias_mean>0.1</bias_mean>
          <bias_stddev>0.001</bias_stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
          <bias_mean>0.1</bias_mean>
          <bias_stddev>0.001</bias_stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
          <bias_mean>0.1</bias_mean>
          <bias_stddev>0.001</bias_stddev>
        </noise>
      </z>
    </linear_acceleration>
  </imu>
  <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Imu">
    <topic_name>/humanoid_robot/imu</topic_name>
    <frame_id>imu_link</frame_id>
  </plugin>
</sensor>
```

### Force/Torque Sensor Simulation

Force/torque sensors are essential for manipulation tasks:

```xml
<sensor name="left_foot_force_torque" type="force_torque">
  <always_on>true</always_on>
  <update_rate>100</update_rate>
  <force_torque>
    <frame>child</frame>
    <measure_direction>child_to_parent</measure_direction>
  </force_torque>
  <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::ForceTorque">
    <topic_name>/humanoid_robot/left_foot/ft_sensor</topic_name>
  </plugin>
</sensor>
```

### Multi-Sensor Integration Node

To process data from multiple sensors, you'll need a ROS 2 node that subscribes to all sensor topics:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, Imu, PointCloud2, JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from tf2_ros import TransformBroadcaster
from cv_bridge import CvBridge
import numpy as np
import cv2

class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion_node')

        # Create transform broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Initialize CV bridge
        self.cv_bridge = CvBridge()

        # Sensor data storage
        self.camera_data = None
        self.laser_data = None
        self.imu_data = None
        self.joint_states = None

        # Create subscribers for all sensors
        self.camera_sub = self.create_subscription(
            Image, '/humanoid_robot/camera/image_raw',
            self.camera_callback, 10)

        self.laser_sub = self.create_subscription(
            LaserScan, '/humanoid_robot/laser_scan',
            self.laser_callback, 10)

        self.imu_sub = self.create_subscription(
            Imu, '/humanoid_robot/imu',
            self.imu_callback, 10)

        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states',
            self.joint_state_callback, 10)

        # Create publishers for processed data
        self.processed_data_pub = self.create_publisher(
            String, '/processed_sensor_data', 10)

        # Timer for fusion processing
        self.fusion_timer = self.create_timer(0.05, self.process_sensor_fusion)

    def camera_callback(self, msg):
        """Process camera data"""
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
            # Process image (e.g., object detection, feature extraction)
            self.camera_data = cv_image
        except Exception as e:
            self.get_logger().error(f'Camera callback error: {e}')

    def laser_callback(self, msg):
        """Process laser scan data"""
        # Process LiDAR data (e.g., obstacle detection, mapping)
        self.laser_data = {
            'ranges': msg.ranges,
            'angle_min': msg.angle_min,
            'angle_max': msg.angle_max,
            'angle_increment': msg.angle_increment
        }

    def imu_callback(self, msg):
        """Process IMU data"""
        # Process IMU data (e.g., orientation estimation, balance control)
        self.imu_data = {
            'orientation': msg.orientation,
            'angular_velocity': msg.angular_velocity,
            'linear_acceleration': msg.linear_acceleration
        }

    def joint_state_callback(self, msg):
        """Process joint state data"""
        self.joint_states = {
            'position': dict(zip(msg.name, msg.position)),
            'velocity': dict(zip(msg.name, msg.velocity)),
            'effort': dict(zip(msg.name, msg.effort))
        }

    def process_sensor_fusion(self):
        """Process and fuse data from multiple sensors"""
        if all(data is not None for data in
               [self.camera_data, self.laser_data, self.imu_data, self.joint_states]):

            # Example fusion: detect obstacles using both camera and LiDAR
            camera_obstacles = self.detect_obstacles_camera(self.camera_data)
            lidar_obstacles = self.detect_obstacles_lidar(self.laser_data)

            # Combine sensor data for more robust perception
            fused_data = self.fuse_obstacle_data(camera_obstacles, lidar_obstacles)

            # Publish fused results
            result_msg = String()
            result_msg.data = str(fused_data)
            self.processed_data_pub.publish(result_msg)

    def detect_obstacles_camera(self, image):
        """Detect obstacles in camera image"""
        # Simple example: detect bright regions as potential obstacles
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # Find contours of bright regions
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        obstacles = []
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Filter small regions
                x, y, w, h = cv2.boundingRect(contour)
                obstacles.append({
                    'x': x, 'y': y, 'width': w, 'height': h,
                    'area': cv2.contourArea(contour)
                })

        return obstacles

    def detect_obstacles_lidar(self, laser_data):
        """Detect obstacles in LiDAR data"""
        obstacles = []
        for i, range_val in enumerate(laser_data['ranges']):
            if 0.1 < range_val < 2.0:  # Obstacle within 2m
                angle = laser_data['angle_min'] + i * laser_data['angle_increment']
                obstacles.append({
                    'angle': angle,
                    'range': range_val,
                    'x': range_val * np.cos(angle),
                    'y': range_val * np.sin(angle)
                })

        return obstacles

    def fuse_obstacle_data(self, camera_obstacles, lidar_obstacles):
        """Fuse obstacle data from camera and LiDAR"""
        # This is a simplified example - in practice, you'd use more sophisticated
        # sensor fusion algorithms like Kalman filters or particle filters
        return {
            'camera_obstacles': len(camera_obstacles),
            'lidar_obstacles': len(lidar_obstacles),
            'timestamp': self.get_clock().now().to_msg()
        }

def main(args=None):
    rclpy.init(args=args)
    sensor_fusion_node = SensorFusionNode()

    try:
        rclpy.spin(sensor_fusion_node)
    except KeyboardInterrupt:
        pass
    finally:
        sensor_fusion_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Sensor Calibration and Validation

Proper calibration ensures that simulated sensors match real-world counterparts:

#### Camera Calibration
```yaml
# camera_calibration.yaml
image_width: 640
image_height: 480
camera_name: head_camera
camera_matrix:
  rows: 3
  cols: 3
  data: [525.0, 0.0, 319.5, 0.0, 525.0, 239.5, 0.0, 0.0, 1.0]
distortion_model: plumb_bob
distortion_coefficients:
  rows: 1
  cols: 5
  data: [0.0, 0.0, 0.0, 0.0, 0.0]
rectification_matrix:
  rows: 3
  cols: 3
  data: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
projection_matrix:
  rows: 3
  cols: 4
  data: [525.0, 0.0, 319.5, 0.0, 0.0, 525.0, 239.5, 0.0, 0.0, 0.0, 1.0, 0.0]
```

#### Sensor Validation Node
```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, Imu
from std_msgs.msg import Float64
import numpy as np

class SensorValidatorNode(Node):
    def __init__(self):
        super().__init__('sensor_validator')

        # Sensor validation publishers
        self.camera_validity_pub = self.create_publisher(Float64, '/camera_validity', 10)
        self.laser_validity_pub = self.create_publisher(Float64, '/laser_validity', 10)
        self.imu_validity_pub = self.create_publisher(Float64, '/imu_validity', 10)

        # Sensor subscribers
        self.camera_sub = self.create_subscription(
            Image, '/humanoid_robot/camera/image_raw',
            self.validate_camera, 10)

        self.laser_sub = self.create_subscription(
            LaserScan, '/humanoid_robot/laser_scan',
            self.validate_laser, 10)

        self.imu_sub = self.create_subscription(
            Imu, '/humanoid_robot/imu',
            self.validate_imu, 10)

    def validate_camera(self, msg):
        """Validate camera sensor data"""
        # Check if image has valid data
        validity_score = 1.0  # Start with perfect validity

        # Check for common issues
        if msg.width <= 0 or msg.height <= 0:
            validity_score = 0.0
        elif msg.encoding not in ['rgb8', 'bgr8', 'mono8']:
            validity_score = 0.5
        else:
            # Additional checks could include:
            # - Checking for all-black or all-white images
            # - Checking for excessive noise
            # - Checking for consistent frame rates
            pass

        validity_msg = Float64()
        validity_msg.data = validity_score
        self.camera_validity_pub.publish(validity_msg)

    def validate_laser(self, msg):
        """Validate LiDAR sensor data"""
        validity_score = 1.0

        # Check for valid ranges
        valid_ranges = [r for r in msg.ranges if not (np.isnan(r) or np.isinf(r))]
        if len(valid_ranges) == 0:
            validity_score = 0.0
        elif len(valid_ranges) < len(msg.ranges) * 0.5:  # Less than 50% valid
            validity_score = 0.3
        else:
            # Check for reasonable range values
            if max(valid_ranges) > msg.range_max or min(valid_ranges) < msg.range_min:
                validity_score = 0.7

        validity_msg = Float64()
        validity_msg.data = validity_score
        self.laser_validity_pub.publish(validity_msg)

    def validate_imu(self, msg):
        """Validate IMU sensor data"""
        validity_score = 1.0

        # Check if orientation is normalized
        norm = np.sqrt(
            msg.orientation.x**2 +
            msg.orientation.y**2 +
            msg.orientation.z**2 +
            msg.orientation.w**2
        )

        if abs(norm - 1.0) > 0.1:  # Not normalized
            validity_score = 0.5

        # Check for reasonable acceleration values (should be around 9.8 m/s^2 when still)
        linear_acc = np.sqrt(
            msg.linear_acceleration.x**2 +
            msg.linear_acceleration.y**2 +
            msg.linear_acceleration.z**2
        )

        if linear_acc < 5.0 or linear_acc > 15.0:  # Should be around 9.8
            validity_score *= 0.8

        validity_msg = Float64()
        validity_msg.data = validity_score
        self.imu_validity_pub.publish(validity_msg)

def main(args=None):
    rclpy.init(args=args)
    validator = SensorValidatorNode()

    try:
        rclpy.spin(validator)
    except KeyboardInterrupt:
        pass
    finally:
        validator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Sensor Noise Modeling

Realistic noise modeling is crucial for sim-to-real transfer:

#### Camera Noise Models
```xml
<sensor name="noisy_camera" type="camera">
  <camera name="head">
    <horizontal_fov>1.3962634</horizontal_fov>
    <image>
      <width>640</width>
      <height>480</height>
      <format>R8G8B8</format>
    </image>
    <clip>
      <near>0.1</near>
      <far>10</far>
    </clip>
    <!-- Add realistic noise -->
    <noise>
      <type>gaussian</type>
      <mean>0.0</mean>
      <stddev>0.01</stddev>  <!-- 1% noise for realistic camera -->
    </noise>
  </camera>
</sensor>
```

#### LiDAR Noise Models
```xml
<sensor name="noisy_lidar" type="gpu_lidar">
  <ray>
    <scan>
      <horizontal>
        <samples>720</samples>
        <resolution>1</resolution>
        <min_angle>-1.570796</min_angle>
        <max_angle>1.570796</max_angle>
      </horizontal>
    </scan>
    <range>
      <min>0.1</min>
      <max>30.0</max>
      <resolution>0.01</resolution>
      <!-- Add realistic noise -->
      <noise>
        <type>gaussian</type>
        <mean>0.0</mean>
        <stddev>0.02</stddev>  <!-- 2cm noise for typical LiDAR -->
      </noise>
    </range>
  </ray>
</sensor>
```

### Sensor Simulation Best Practices

1. **Match Real Sensor Specifications**: Use parameters that match your real sensors
2. **Include Realistic Noise**: Add appropriate noise models based on real sensor characteristics
3. **Validate Sensor Data**: Implement validation nodes to ensure sensor quality
4. **Consider Computational Cost**: Balance realism with simulation performance
5. **Calibrate Regularly**: Ensure simulated sensors remain aligned with real counterparts

### Testing Sensor Integration

Create comprehensive tests for your sensor setup:

```bash
# Test that all sensors are publishing data
ros2 topic list | grep sensor

# Echo specific sensor topics to verify data
ros2 topic echo /humanoid_robot/camera/image_raw --field data -c

# Visualize sensors in RViz2
ros2 run rviz2 rviz2
```

### Troubleshooting Common Sensor Issues

1. **No Data Publishing**: Check Gazebo plugins and ROS 2 bridge configuration
2. **Incorrect Data**: Verify sensor parameters and coordinate frames
3. **Performance Issues**: Reduce update rates or simplify sensor models
4. **Noise Too High/Low**: Adjust noise parameters to match real sensors

### Summary

In this chapter, we've covered:
- Camera simulation with various configurations (RGB, stereo, depth)
- LiDAR and range sensor simulation for navigation and mapping
- IMU simulation with realistic noise models
- Force/torque sensor simulation for manipulation
- Multi-sensor integration and fusion techniques
- Sensor calibration and validation approaches
- Realistic noise modeling for sim-to-real transfer
- Best practices for sensor simulation

With this comprehensive understanding of sensor simulation, you now have the tools to create realistic digital twins for humanoid robots. In Module 3, we'll explore the AI-Robot Brain using NVIDIA Isaac tools for perception and navigation intelligence.