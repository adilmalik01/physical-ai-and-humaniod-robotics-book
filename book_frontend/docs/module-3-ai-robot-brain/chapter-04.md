---
sidebar_position: 4
---

# Chapter 04: Navigation with Nav2 for Humanoids

## Autonomous Navigation and Path Planning

In this chapter, we'll explore how to implement autonomous navigation for humanoid robots using Navigation2 (Nav2) in conjunction with Isaac tools. Humanoid navigation presents unique challenges compared to wheeled robots, requiring specialized approaches for path planning, obstacle avoidance, and locomotion control.

### Understanding Humanoid Navigation Challenges

Humanoid robots face distinct navigation challenges compared to traditional mobile robots:

#### Physical Constraints
- **Stability**: Maintaining balance while moving
- **Step Height**: Limited ability to step over obstacles
- **Foot Placement**: Need for stable foot placement locations
- **Turning Radius**: Different turning dynamics than wheeled robots
- **Energy Efficiency**: Walking patterns affect battery life

#### Environmental Requirements
- **Walkable Surfaces**: Not all ground is traversable
- **Stair Navigation**: Ability to navigate stairs (if equipped)
- **Narrow Spaces**: Need to navigate through tight spaces
- **Dynamic Obstacles**: Interaction with humans and moving objects

### Nav2 Architecture Overview

Navigation2 provides a flexible and extensible navigation framework:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Perception    │ -> │   Planning      │ -> │   Control       │
│   (SLAM,       │    │   (Global &     │    │   (Local        │
│   Localization) │    │   Local)        │    │   Planning)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Nav2 Components for Humanoids

#### Global Planner
- **A* or Dijkstra**: Pathfinding algorithms
- **Grid-based**: Consider humanoid-specific constraints
- **Smooth paths**: Generate paths suitable for bipedal locomotion

#### Local Planner
- **Trajectory Rollout**: Evaluate possible trajectories
- **Dynamic Window Approach**: Consider humanoid dynamics
- **Obstacle Avoidance**: Real-time obstacle avoidance

#### Controller
- **Footstep Planner**: For bipedal robots
- **Balance Control**: Maintain stability during movement
- **Gait Generation**: Generate appropriate walking patterns

### Setting Up Nav2 for Humanoids

To configure Nav2 for humanoid robots, you'll need to customize several parameters:

#### Nav2 Launch File for Humanoids
```xml
<!-- launch/humanoid_navigation.launch.xml -->
<launch>
  <!-- Map server -->
  <node pkg="nav2_map_server" exec="map_server" name="map_server">
    <param name="yaml_filename" value="/path/to/map.yaml"/>
    <param name="topic" value="map"/>
    <param name="frame_id" value="map"/>
    <param name="set_initial_pose" value="true"/>
  </node>

  <!-- Local costmap -->
  <node pkg="nav2_costmap_2d" exec="nav2_costmap_2d" name="local_costmap">
    <param name="use_sim_time" value="True"/>
    <param name="global_frame" value="odom"/>
    <param name="robot_base_frame" value="base_link"/>
    <param name="update_frequency" value="5.0"/>
    <param name="publish_frequency" value="2.0"/>
    <param name="width" value="10.0"/>
    <param name="height" value="10.0"/>
    <param name="resolution" value="0.05"/>
    <param name="footprint" value="[[0.3, 0.3], [0.3, -0.3], [-0.3, -0.3], [-0.3, 0.3]]"/>
    <param name="plugins" value="[nav2_costmap_2d::StaticLayer, nav2_costmap_2d::ObstacleLayer, nav2_costmap_2d::InflationLayer]"/>
  </node>

  <!-- Global costmap -->
  <node pkg="nav2_costmap_2d" exec="nav2_costmap_2d" name="global_costmap">
    <param name="use_sim_time" value="True"/>
    <param name="global_frame" value="map"/>
    <param name="robot_base_frame" value="base_link"/>
    <param name="update_frequency" value="1.0"/>
    <param name="static_map" value="true"/>
    <param name="width" value="100.0"/>
    <param name="height" value="100.0"/>
    <param name="resolution" value="0.05"/>
    <param name="footprint" value="[[0.3, 0.3], [0.3, -0.3], [-0.3, -0.3], [-0.3, 0.3]]"/>
    <param name="plugins" value="[nav2_costmap_2d::StaticLayer, nav2_costmap_2d::ObstacleLayer, nav2_costmap_2d::InflationLayer]"/>
  </node>

  <!-- Planner Server -->
  <node pkg="nav2_planner" exec="planner_server" name="planner_server">
    <param name="use_sim_time" value="True"/>
    <param name="planner_plugins" value="GridBased"/>
    <param name="GridBased.type" value="nav2_navfn_planner/NavfnPlanner"/>
  </node>

  <!-- Controller Server -->
  <node pkg="nav2_controller" exec="controller_server" name="controller_server">
    <param name="use_sim_time" value="True"/>
    <param name="controller_frequency" value="20.0"/>
    <param name="min_x_velocity_threshold" value="0.001"/>
    <param name="min_y_velocity_threshold" value="0.5"/>
    <param name="min_theta_velocity_threshold" value="0.001"/>
    <param name="progress_checker_plugin" value="progress_checker"/>
    <param name="goal_checker_plugin" value="goal_checker"/>
    <param name="controller_plugins" value="FollowPath"/>
    <param name="FollowPath.type" value="nav2_mppi_controller/MPPITrajectoryController"/>
  </node>

  <!-- Behavior Tree Server -->
  <node pkg="nav2_bt_navigator" exec="bt_navigator" name="bt_navigator">
    <param name="use_sim_time" value="True"/>
    <param name="global_frame" value="map"/>
    <param name="robot_base_frame" value="base_link"/>
    <param name="odom_topic" value="odom"/>
    <param name="default_bt_xml_filename" value="package://nav2_bt_navigator/behavior_trees/navigate_w_replanning_and_recovery.xml"/>
  </node>

  <!-- Lifecycle Manager -->
  <node pkg="nav2_lifecycle_manager" exec="lifecycle_manager" name="lifecycle_manager">
    <param name="use_sim_time" value="True"/>
    <param name="autostart" value="True"/>
    <param name="node_names" value="[map_server, local_costmap, global_costmap, planner_server, controller_server, bt_navigator]"/>
  </node>
</launch>
```

#### Humanoid-Specific Costmap Configuration
```yaml
# config/humanoid_costmap_params.yaml
local_costmap:
  global_frame: odom
  robot_base_frame: base_link
  update_frequency: 5.0
  publish_frequency: 2.0
  width: 10.0
  height: 10.0
  resolution: 0.05
  origin_x: 0.0
  origin_y: 0.0
  footprint: [[0.3, 0.3], [0.3, -0.3], [-0.3, -0.3], [-0.3, 0.3]]
  footprint_padding: 0.01
  plugins:
    - {name: static_layer, type: "nav2_costmap_2d::StaticLayer"}
    - {name: obstacle_layer, type: "nav2_costmap_2d::ObstacleLayer"}
    - {name: inflation_layer, type: "nav2_costmap_2d::InflationLayer"}
  obstacle_layer:
    enabled: true
    observation_sources: scan
    scan:
      topic: /scan
      max_obstacle_height: 2.0  # Humanoid can step over low obstacles
      clearing: true
      marking: true
      data_type: "LaserScan"
      raytrace_max_range: 3.0
      raytrace_min_range: 0.0
      obstacle_max_range: 2.5
      obstacle_min_range: 0.0
  inflation_layer:
    enabled: true
    cost_scaling_factor: 3.0
    inflation_radius: 0.55
    inflate_unknown: false

global_costmap:
  global_frame: map
  robot_base_frame: base_link
  update_frequency: 1.0
  static_map: true
  width: 100.0
  height: 100.0
  resolution: 0.05
  origin_x: 0.0
  origin_y: 0.0
  footprint: [[0.3, 0.3], [0.3, -0.3], [-0.3, -0.3], [-0.3, 0.3]]
  plugins:
    - {name: static_layer, type: "nav2_costmap_2d::StaticLayer"}
    - {name: obstacle_layer, type: "nav2_costmap_2d::ObstacleLayer"}
    - {name: inflation_layer, type: "nav2_costmap_2d::InflationLayer"}
```

### Isaac Integration with Nav2

Integrating Isaac tools with Nav2 enhances perception and navigation capabilities:

#### Isaac Visual SLAM Integration
```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid
from tf2_ros import TransformListener, Buffer
import numpy as np

class IsaacSLAMToNav2Bridge(Node):
    def __init__(self):
        super().__init__('isaac_slam_to_nav2_bridge')

        # Initialize TF listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subscribe to Isaac Visual SLAM pose
        self.slam_pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/isaac_visual_slam/pose',
            self.slam_pose_callback,
            10
        )

        # Subscribe to Isaac SLAM map
        self.slam_map_sub = self.create_subscription(
            OccupancyGrid,
            '/isaac_visual_slam/map',
            self.slam_map_callback,
            10
        )

        # Publishers for Nav2
        self.initial_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            '/initialpose',
            10
        )

        # Store localization state
        self.localization_initialized = False

        self.get_logger().info('Isaac SLAM to Nav2 Bridge initialized')

    def slam_pose_callback(self, msg):
        """Handle Isaac SLAM pose updates"""
        if not self.localization_initialized:
            # Initialize Nav2 with SLAM pose
            self.initialize_nav2_localization(msg)
            self.localization_initialized = True

        # Transform SLAM pose to Nav2 coordinate frame
        try:
            transform = self.tf_buffer.lookup_transform(
                'map',  # Nav2 map frame
                msg.header.frame_id,  # SLAM frame
                rclpy.time.Time()
            )

            # Apply transform to pose
            transformed_pose = self.transform_pose(msg.pose, transform)

            # Update Nav2 localization
            self.update_nav2_pose(transformed_pose, msg.header)
        except Exception as e:
            self.get_logger().warn(f'Transform lookup failed: {e}')

    def slam_map_callback(self, msg):
        """Handle Isaac SLAM map updates"""
        # Publish map to Nav2
        map_pub = self.create_publisher(OccupancyGrid, '/map', 10)
        map_pub.publish(msg)

    def initialize_nav2_localization(self, slam_pose):
        """Initialize Nav2 localization with SLAM pose"""
        # Create initial pose message for Nav2
        initial_pose = PoseWithCovarianceStamped()
        initial_pose.header.frame_id = 'map'
        initial_pose.header.stamp = self.get_clock().now().to_msg()
        initial_pose.pose = slam_pose.pose

        # Publish to initialize Nav2 localization
        self.initial_pose_pub.publish(initial_pose)

        self.get_logger().info('Nav2 localization initialized from Isaac SLAM')

    def transform_pose(self, pose, transform):
        """Apply transform to pose"""
        # Implementation of pose transformation
        # This would convert the pose from SLAM frame to Nav2 frame
        return pose  # Placeholder
```

#### Isaac Perception Integration
```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, PointCloud2
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import MarkerArray
import numpy as np

class IsaacPerceptionToNav2(Node):
    def __init__(self):
        super().__init__('isaac_perception_to_nav2')

        # Subscribe to Isaac perception outputs
        self.detection_sub = self.create_subscription(
            Detection2DArray,
            '/isaac_dnn_inference/detections',
            self.detection_callback,
            10
        )

        self.pointcloud_sub = self.create_subscription(
            PointCloud2,
            '/depth_camera/points',
            self.pointcloud_callback,
            10
        )

        # Publishers for Nav2 costmaps
        self.obstacle_pub = self.create_publisher(
            LaserScan,
            '/scan',
            10
        )

        # Store detection data
        self.detections = []
        self.pointcloud = None

        self.get_logger().info('Isaac Perception to Nav2 node initialized')

    def detection_callback(self, msg):
        """Process Isaac perception detections"""
        # Convert DNN detections to obstacle representation
        obstacles = self.process_detections(msg)

        # Publish to Nav2 obstacle layer
        obstacle_scan = self.create_obstacle_scan(obstacles)
        self.obstacle_pub.publish(obstacle_scan)

    def pointcloud_callback(self, msg):
        """Process Isaac point cloud data"""
        # Convert point cloud to 2D obstacles for navigation
        obstacles_2d = self.process_pointcloud(msg)

        # Update costmap with 3D-aware obstacles
        self.update_costmap_with_3d_obstacles(obstacles_2d)

    def process_detections(self, detections_msg):
        """Process detections for navigation"""
        obstacles = []

        for detection in detections_msg.detections:
            # Calculate obstacle position from detection
            center_x = detection.bbox.center.x
            center_y = detection.bbox.center.y
            width = detection.bbox.size_x
            height = detection.bbox.size_y

            # Determine obstacle height from class
            class_id = detection.results[0].hypothesis.class_id
            if class_id in ['person', 'human']:
                height_m = 1.7  # Human height
            elif class_id in ['chair', 'table']:
                height_m = 0.8  # Furniture height
            else:
                height_m = 0.5  # Default height

            obstacle = {
                'center_x': center_x,
                'center_y': center_y,
                'width': width,
                'height': height_m,
                'class': class_id
            }
            obstacles.append(obstacle)

        return obstacles

    def create_obstacle_scan(self, obstacles):
        """Create LaserScan message from obstacles"""
        scan = LaserScan()
        scan.header.stamp = self.get_clock().now().to_msg()
        scan.header.frame_id = 'base_scan'
        scan.angle_min = -np.pi / 2
        scan.angle_max = np.pi / 2
        scan.angle_increment = 0.01
        scan.time_increment = 0.0
        scan.scan_time = 0.1
        scan.range_min = 0.1
        scan.range_max = 10.0

        # Convert obstacles to ranges
        num_ranges = int((scan.angle_max - scan.angle_min) / scan.angle_increment)
        ranges = [scan.range_max] * num_ranges

        for obstacle in obstacles:
            # Calculate angle and distance to obstacle
            distance = np.sqrt(obstacle['center_x']**2 + obstacle['center_y']**2)
            angle = np.arctan2(obstacle['center_y'], obstacle['center_x'])

            # Find corresponding range index
            angle_index = int((angle - scan.angle_min) / scan.angle_increment)
            if 0 <= angle_index < len(ranges):
                ranges[angle_index] = min(ranges[angle_index], distance)

        scan.ranges = ranges
        return scan
```

### Humanoid Path Planning

For humanoid robots, path planning needs to consider bipedal locomotion:

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Path
from visualization_msgs.msg import MarkerArray, Marker
import numpy as np

class HumanoidPathPlanner(Node):
    def __init__(self):
        super().__init__('humanoid_path_planner')

        # Subscribe to Nav2 global path
        self.global_path_sub = self.create_subscription(
            Path,
            '/plan',
            self.global_path_callback,
            10
        )

        # Publisher for humanoid-specific path
        self.humanoid_path_pub = self.create_publisher(
            Path,
            '/humanoid_plan',
            10
        )

        # Publisher for footstep plan
        self.footstep_pub = self.create_publisher(
            MarkerArray,
            '/footsteps',
            10
        )

        # Parameters for humanoid navigation
        self.step_length = 0.3  # meters
        self.step_width = 0.2   # meters (lateral step)
        self.max_step_height = 0.15  # maximum obstacle height to step over

        self.get_logger().info('Humanoid Path Planner initialized')

    def global_path_callback(self, msg):
        """Process global path for humanoid-specific requirements"""
        # Convert Nav2 path to humanoid-appropriate path
        humanoid_path = self.adapt_path_for_humanoid(msg)

        # Generate footstep plan
        footsteps = self.generate_footsteps(humanoid_path)

        # Publish both paths
        self.humanoid_path_pub.publish(humanoid_path)
        self.publish_footsteps(footsteps, msg.header)

    def adapt_path_for_humanoid(self, nav2_path):
        """Adapt Nav2 path for humanoid constraints"""
        # Create new path with humanoid-specific constraints
        humanoid_path = Path()
        humanoid_path.header = nav2_path.header

        # Smooth the path to account for humanoid turning radius
        smoothed_poses = self.smooth_path(nav2_path.poses)

        # Filter points to ensure they're on walkable surfaces
        walkable_poses = self.filter_walkable_surfaces(smoothed_poses)

        humanoid_path.poses = walkable_poses
        return humanoid_path

    def smooth_path(self, poses):
        """Smooth path for humanoid locomotion"""
        if len(poses) < 3:
            return poses

        smoothed = [poses[0]]  # Keep first pose

        for i in range(1, len(poses) - 1):
            # Calculate smoothed point using adjacent points
            prev_pos = np.array([
                poses[i-1].pose.position.x,
                poses[i-1].pose.position.y
            ])
            curr_pos = np.array([
                poses[i].pose.position.x,
                poses[i].pose.position.y
            ])
            next_pos = np.array([
                poses[i+1].pose.position.x,
                poses[i+1].pose.position.y
            ])

            # Smooth using weighted average
            smoothed_pos = 0.25 * prev_pos + 0.5 * curr_pos + 0.25 * next_pos

            # Create new pose
            new_pose = PoseStamped()
            new_pose.header = poses[i].header
            new_pose.pose.position.x = smoothed_pos[0]
            new_pose.pose.position.y = smoothed_pos[1]
            new_pose.pose.position.z = poses[i].pose.position.z  # Keep original Z
            smoothed.append(new_pose)

        if len(poses) > 1:
            smoothed.append(poses[-1])  # Keep last pose

        return smoothed

    def filter_walkable_surfaces(self, poses):
        """Filter path to ensure walkable surfaces"""
        # In practice, this would check with local map for walkable areas
        # For now, return all poses
        return poses

    def generate_footsteps(self, path):
        """Generate footstep plan from path"""
        footsteps = []

        if len(path.poses) < 2:
            return footsteps

        # Start with current position
        current_pos = np.array([
            path.poses[0].pose.position.x,
            path.poses[0].pose.position.y
        ])

        # Determine initial step (left or right foot)
        step_side = 'left'  # Start with left foot

        for i in range(1, len(path.poses)):
            target_pos = np.array([
                path.poses[i].pose.position.x,
                path.poses[i].pose.position.y
            ])

            # Calculate direction
            direction = target_pos - current_pos
            distance = np.linalg.norm(direction)

            if distance > self.step_length * 0.5:  # Need to take a step
                # Normalize direction
                direction_unit = direction / distance

                # Calculate foot position
                foot_offset = self.step_width / 2 if step_side == 'left' else -self.step_width / 2
                foot_pos = current_pos + self.step_length * direction_unit
                foot_pos[1] += foot_offset if step_side == 'left' else -foot_offset

                footsteps.append({
                    'position': foot_pos,
                    'side': step_side,
                    'timestamp': path.poses[i].header.stamp
                })

                # Move to this position
                current_pos = foot_pos
                step_side = 'right' if step_side == 'left' else 'left'

        return footsteps

    def publish_footsteps(self, footsteps, header):
        """Publish footsteps as visualization markers"""
        marker_array = MarkerArray()

        for i, footstep in enumerate(footsteps):
            marker = Marker()
            marker.header = header
            marker.ns = "footsteps"
            marker.id = i
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD

            marker.pose.position.x = footstep['position'][0]
            marker.pose.position.y = footstep['position'][1]
            marker.pose.position.z = 0.01  # Slightly above ground
            marker.pose.orientation.w = 1.0

            marker.scale.x = 0.1  # Foot size
            marker.scale.y = 0.05
            marker.scale.z = 0.01

            marker.color.a = 0.8
            if footstep['side'] == 'left':
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
            else:
                marker.color.r = 0.0
                marker.color.g = 0.0
                marker.color.b = 1.0

            marker_array.markers.append(marker)

        self.footstep_pub.publish(marker_array)
```

### Isaac Sim Navigation Testing

Test navigation in Isaac Sim before real-world deployment:

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
import time

class IsaacNavigationTester(Node):
    def __init__(self):
        super().__init__('isaac_navigation_tester')

        # Navigation action client
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Robot velocity publisher
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Wait for navigation server
        self.nav_client.wait_for_server()

        # Test waypoints
        self.waypoints = [
            self.create_pose(2.0, 0.0, 0.0),    # Move forward
            self.create_pose(2.0, 2.0, 1.57),   # Turn and move right
            self.create_pose(0.0, 2.0, 3.14),   # Turn and move back
            self.create_pose(0.0, 0.0, -1.57)   # Return to start
        ]

        # Start testing
        self.timer = self.create_timer(5.0, self.execute_navigation_test)

        self.current_waypoint = 0
        self.test_active = False

        self.get_logger().info('Isaac Navigation Tester initialized')

    def create_pose(self, x, y, theta):
        """Create a navigation goal pose"""
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.position.z = 0.0

        # Convert theta to quaternion
        import math
        goal.pose.orientation.z = math.sin(theta / 2.0)
        goal.pose.orientation.w = math.cos(theta / 2.0)

        return goal

    def execute_navigation_test(self):
        """Execute navigation test sequence"""
        if self.test_active:
            return  # Wait for current navigation to complete

        if self.current_waypoint >= len(self.waypoints):
            self.get_logger().info('Navigation test completed')
            self.timer.cancel()
            return

        # Send navigation goal
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = self.waypoints[self.current_waypoint]
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        self.get_logger().info(f'Navigating to waypoint {self.current_waypoint + 1}')

        # Send goal
        self.nav_client.send_goal_async(
            goal_msg,
            feedback_callback=self.navigation_feedback
        ).add_done_callback(self.navigation_result)

        self.test_active = True
        self.current_waypoint += 1

    def navigation_feedback(self, feedback_msg):
        """Handle navigation feedback"""
        feedback = feedback_msg.feedback
        self.get_logger().debug(f'Navigation progress: {feedback.distance_remaining:.2f}m remaining')

    def navigation_result(self, future):
        """Handle navigation result"""
        goal_result = future.result().result
        result_status = future.result().status

        if result_status == 4:  # SUCCEEDED
            self.get_logger().info('Navigation goal reached successfully')
        else:
            self.get_logger().warn(f'Navigation failed with status: {result_status}')

        self.test_active = False

def main(args=None):
    rclpy.init(args=args)
    tester = IsaacNavigationTester()

    try:
        rclpy.spin(tester)
    except KeyboardInterrupt:
        pass
    finally:
        tester.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Safety and Recovery Behaviors

Implement safety measures for humanoid navigation:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
import numpy as np

class HumanoidNavigationSafety(Node):
    def __init__(self):
        super().__init__('humanoid_navigation_safety')

        # Subscribe to sensors
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )

        # Subscribe to navigation status
        self.nav_status_sub = self.create_subscription(
            Bool,
            '/navigation_active',
            self.nav_status_callback,
            10
        )

        # Publisher for emergency stop
        self.emergency_stop_pub = self.create_publisher(
            Bool,
            '/emergency_stop',
            10
        )

        # Publisher for safe velocity commands
        self.safe_cmd_pub = self.create_publisher(
            Twist,
            '/safe_cmd_vel',
            10
        )

        # Safety parameters
        self.safety_distance = 0.5  # meters
        self.balance_threshold = 0.2  # radian tilt threshold
        self.navigation_active = False

        # Current state
        self.current_scan = None
        self.current_imu = None

        self.get_logger().info('Humanoid Navigation Safety initialized')

    def scan_callback(self, msg):
        """Process laser scan for obstacle detection"""
        self.current_scan = msg

        if self.navigation_active:
            # Check for imminent collision
            min_distance = min([r for r in msg.ranges if not np.isnan(r) and r > 0], default=float('inf'))

            if min_distance < self.safety_distance:
                self.trigger_emergency_stop(f"Obstacle detected at {min_distance:.2f}m")

    def imu_callback(self, msg):
        """Process IMU data for balance monitoring"""
        self.current_imu = msg

        if self.navigation_active:
            # Check for balance loss
            roll, pitch = self.quaternion_to_rpy(
                msg.orientation.w,
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z
            )

            if abs(roll) > self.balance_threshold or abs(pitch) > self.balance_threshold:
                self.trigger_emergency_stop(f"Balance threshold exceeded: roll={roll:.2f}, pitch={pitch:.2f}")

    def nav_status_callback(self, msg):
        """Update navigation status"""
        self.navigation_active = msg.data

    def trigger_emergency_stop(self, reason):
        """Trigger emergency stop for safety"""
        self.get_logger().error(f'EMERGENCY STOP: {reason}')

        # Publish emergency stop signal
        stop_msg = Bool()
        stop_msg.data = True
        self.emergency_stop_pub.publish(stop_msg)

        # Publish zero velocity
        stop_twist = Twist()
        self.safe_cmd_pub.publish(stop_twist)

        # Disable navigation
        self.navigation_active = False

    def quaternion_to_rpy(self, w, x, y, z):
        """Convert quaternion to roll-pitch-yaw"""
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if np.abs(sinp) >= 1:
            pitch = np.sign(sinp) * np.pi / 2  # Use 90 degrees if out of range
        else:
            pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch

def main(args=None):
    rclpy.init(args=args)
    safety_node = HumanoidNavigationSafety()

    try:
        rclpy.spin(safety_node)
    except KeyboardInterrupt:
        pass
    finally:
        safety_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Best Practices for Humanoid Navigation

1. **Step Planning**: Plan footsteps explicitly for bipedal locomotion
2. **Balance Monitoring**: Continuously monitor robot balance during navigation
3. **Obstacle Classification**: Distinguish between navigable and non-navigable obstacles
4. **Dynamic Adjustment**: Adapt navigation parameters based on terrain
5. **Safety First**: Implement comprehensive safety checks and emergency procedures

### Troubleshooting Navigation Issues

Common issues and solutions:

1. **Path Oscillation**: Increase smoothing parameters
2. **Collision**: Adjust costmap inflation and safety margins
3. **Localization Drift**: Improve sensor fusion and calibration
4. **Performance**: Optimize costmap resolution and update rates

### Summary

In this chapter, we've covered:
- Humanoid-specific navigation challenges and requirements
- Nav2 architecture and configuration for humanoids
- Isaac integration with Nav2 for enhanced perception
- Humanoid path planning with footstep generation
- Safety measures and emergency procedures
- Testing navigation in simulation

In the next chapter, we'll explore how to close the perception-to-action loop, integrating all the perception and navigation components into a cohesive AI-Robot Brain system.