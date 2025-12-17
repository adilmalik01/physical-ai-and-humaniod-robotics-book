---
sidebar_position: 3
---

# Chapter 03: Autonomous Navigation and Manipulation

## Creating Intelligent Autonomous Behaviors

In this chapter, we'll focus on creating autonomous navigation and manipulation behaviors for the humanoid robot. Building on the voice-to-action pipeline we developed in the previous chapter, we'll implement sophisticated navigation algorithms, manipulation strategies, and the coordination between these capabilities to create truly autonomous behaviors.

### Understanding Autonomous Behaviors

Autonomous behaviors in humanoid robots require the integration of several complex systems:

1. **Perception**: Understanding the environment through sensors
2. **Planning**: Creating paths and manipulation plans
3. **Control**: Executing planned actions with precision
4. **Adaptation**: Adjusting behavior based on changing conditions
5. **Safety**: Ensuring safe operation in dynamic environments

The autonomy stack:

```
┌─────────────────┐    ┌─────────────────┐    └─────────────────┘
│   Mission      │ -> │  Behavior       │ -> │  Execution      │
│   (Goals,      │    │  (Navigation,   │    │  (Motion,       │
│   Tasks)       │    │  Manipulation)   │    │  Control)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                       │                       │
        └───────────────────────┼───────────────────────┘
                                │
                    ┌─────────────────┐
                    │  Perception     │
                    │  (Vision,       │
                    │  Localization)   │
                    └─────────────────┘
```

### Advanced Navigation System

Let's implement a sophisticated navigation system for humanoid robots:

```python
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path, OccupancyGrid, Odometry
from geometry_msgs.msg import PoseStamped, Twist, Point
from sensor_msgs.msg import LaserScan, Imu
from humanoid_robot_msgs.msg import NavigationGoal, NavigationStatus
from humanoid_robot_msgs.srv import PlanPath, GetRobotState
import numpy as np
import heapq
import math
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class NavigationState:
    """State of the navigation system"""
    current_pose: Optional[PoseStamped] = None
    current_velocity: Optional[Twist] = None
    target_pose: Optional[PoseStamped] = None
    path: List[PoseStamped] = None
    status: str = 'idle'
    current_index: int = 0
    balance_state: str = 'balanced'

class AdvancedNavigationSystem(Node):
    """
    Advanced navigation system for humanoid robots
    """
    def __init__(self):
        super().__init__('advanced_navigation_system')

        # Initialize navigation state
        self.nav_state = NavigationState()

        # Initialize navigation components
        self.path_planner = PathPlanner(self)
        self.local_planner = LocalPlanner(self)
        self.controller = NavigationController(self)
        self.obstacle_detector = ObstacleDetector(self)
        self.balance_monitor = BalanceMonitor(self)

        # Set up ROS 2 interfaces
        self.setup_interfaces()

        # Navigation parameters
        self.navigation_params = {
            'max_linear_speed': 0.5,
            'max_angular_speed': 0.5,
            'min_linear_speed': 0.1,
            'min_angular_speed': 0.1,
            'linear_acceleration': 0.5,
            'angular_acceleration': 0.5,
            'inflation_radius': 0.5,
            'footprint_radius': 0.3,
            'step_height': 0.15,
            'balance_threshold': 0.2
        }

        # Navigation thread
        self.navigation_thread = None
        self.navigation_active = False

        # Performance monitoring
        self.navigation_metrics = {
            'paths_planned': 0,
            'navigations_completed': 0,
            'average_path_length': 0.0,
            'average_completion_time': 0.0,
            'collisions_avoided': 0
        }

        self.get_logger().info('Advanced Navigation System initialized')

    def setup_interfaces(self):
        """Set up ROS 2 communication interfaces"""
        # Publishers
        self.path_pub = self.create_publisher(Path, '/navigation/path', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.navigation_status_pub = self.create_publisher(NavigationStatus, '/navigation/status', 10)
        self.goal_pub = self.create_publisher(PoseStamped, '/navigation/goal', 10)

        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10
        )
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10
        )
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10
        )
        self.goal_sub = self.create_subscription(
            PoseStamped, '/move_base_simple/goal', self.external_goal_callback, 10
        )

        # Services
        self.plan_path_srv = self.create_service(
            PlanPath, '/navigation/plan_path', self.plan_path_callback
        )
        self.get_robot_state_srv = self.create_service(
            GetRobotState, '/navigation/get_robot_state', self.get_robot_state_callback
        )

    def odom_callback(self, msg):
        """Handle odometry updates"""
        pose = PoseStamped()
        pose.header = msg.header
        pose.pose = msg.pose.pose

        self.nav_state.current_pose = pose
        self.nav_state.current_velocity = msg.twist.twist

        # Update controller with current state
        self.controller.update_current_state(pose, msg.twist.twist)

    def scan_callback(self, msg):
        """Handle laser scan updates"""
        # Process obstacles for local planning
        obstacles = self.obstacle_detector.process_scan(msg)

        # Update local planner with obstacle information
        self.local_planner.update_obstacles(obstacles)

        # Check for balance issues related to obstacles
        self.balance_monitor.check_obstacle_balance(obstacles)

    def imu_callback(self, msg):
        """Handle IMU updates for balance monitoring"""
        # Update balance state
        self.nav_state.balance_state = self.balance_monitor.update_imu_data(msg)

    def map_callback(self, msg):
        """Handle map updates"""
        self.path_planner.update_map(msg)

    def external_goal_callback(self, msg):
        """Handle external navigation goal"""
        self.set_navigation_goal(msg)

    def set_navigation_goal(self, goal: PoseStamped):
        """Set navigation goal and start navigation"""
        self.nav_state.target_pose = goal
        self.nav_state.status = 'planning'

        # Plan path to goal
        path = self.path_planner.plan_path_to_goal(
            self.nav_state.current_pose,
            goal,
            self.navigation_params
        )

        if path:
            self.nav_state.path = path
            self.nav_state.current_index = 0

            # Publish path
            path_msg = Path()
            path_msg.header = goal.header
            path_msg.poses = path
            self.path_pub.publish(path_msg)

            # Start navigation thread
            if not self.navigation_active:
                self.start_navigation()

            self.nav_state.status = 'navigating'
        else:
            self.nav_state.status = 'path_failed'
            self.get_logger().error('Failed to plan path to goal')

    def start_navigation(self):
        """Start navigation thread"""
        if self.navigation_thread is None or not self.navigation_thread.is_alive():
            self.navigation_active = True
            self.navigation_thread = threading.Thread(target=self.navigation_loop, daemon=True)
            self.navigation_thread.start()

    def stop_navigation(self):
        """Stop navigation"""
        self.navigation_active = False
        if self.navigation_thread and self.navigation_thread.is_alive():
            self.navigation_thread.join(timeout=1.0)

        # Stop robot
        stop_msg = Twist()
        self.cmd_vel_pub.publish(stop_msg)

        self.nav_state.status = 'idle'

    def navigation_loop(self):
        """Main navigation execution loop"""
        while self.navigation_active and rclpy.ok():
            try:
                # Check if we have a valid path and goal
                if (self.nav_state.path and self.nav_state.current_pose and
                    self.nav_state.target_pose and self.nav_state.status == 'navigating'):

                    # Get current path index
                    current_idx = self.nav_state.current_index

                    # Check if we've reached the goal
                    if current_idx >= len(self.nav_state.path) - 1:
                        # Check if we're close enough to goal
                        if self.is_close_to_goal():
                            self.nav_state.status = 'completed'
                            self.publish_navigation_status()
                            self.stop_navigation()
                            continue

                    # Get next waypoint
                    if current_idx < len(self.nav_state.path):
                        target_pose = self.nav_state.path[current_idx]

                        # Check if we've reached the current waypoint
                        if self.has_reached_waypoint(target_pose):
                            self.nav_state.current_index += 1
                            target_pose = self.nav_state.path[min(current_idx + 1, len(self.nav_state.path) - 1)]

                    # Plan local trajectory considering obstacles
                    local_plan = self.local_planner.plan_local_trajectory(
                        self.nav_state.current_pose,
                        target_pose
                    )

                    # Generate velocity commands
                    cmd_vel = self.controller.compute_velocity_command(
                        self.nav_state.current_pose,
                        local_plan,
                        self.nav_state.current_velocity
                    )

                    # Check balance before executing command
                    if self.nav_state.balance_state == 'balanced':
                        # Publish velocity command
                        self.cmd_vel_pub.publish(cmd_vel)
                    else:
                        # Reduce speed or stop if balance is compromised
                        reduced_cmd = self.reduce_speed_for_balance(cmd_vel)
                        self.cmd_vel_pub.publish(reduced_cmd)

                    # Update navigation status
                    self.publish_navigation_status()

                # Small delay to prevent busy waiting
                time.sleep(0.05)  # 20 Hz control loop

            except Exception as e:
                self.get_logger().error(f'Navigation loop error: {e}')
                time.sleep(0.1)

    def is_close_to_goal(self) -> bool:
        """Check if robot is close enough to goal"""
        if not self.nav_state.current_pose or not self.nav_state.target_pose:
            return False

        current_pos = self.nav_state.current_pose.pose.position
        goal_pos = self.nav_state.target_pose.pose.position

        distance = math.sqrt(
            (current_pos.x - goal_pos.x)**2 +
            (current_pos.y - goal_pos.y)**2
        )

        return distance < 0.3  # 30cm threshold

    def has_reached_waypoint(self, waypoint: PoseStamped) -> bool:
        """Check if robot has reached a waypoint"""
        if not self.nav_state.current_pose:
            return False

        current_pos = self.nav_state.current_pose.pose.position
        wp_pos = waypoint.pose.position

        distance = math.sqrt(
            (current_pos.x - wp_pos.x)**2 +
            (current_pos.y - wp_pos.y)**2
        )

        return distance < 0.5  # 50cm threshold

    def reduce_speed_for_balance(self, cmd_vel: Twist) -> Twist:
        """Reduce speed when balance is compromised"""
        reduced_cmd = Twist()
        reduced_cmd.linear.x = cmd_vel.linear.x * 0.3  # Reduce to 30% speed
        reduced_cmd.angular.z = cmd_vel.angular.z * 0.5  # Reduce to 50% angular
        return reduced_cmd

    def publish_navigation_status(self):
        """Publish current navigation status"""
        status_msg = NavigationStatus()
        status_msg.header.stamp = self.get_clock().now().to_msg()
        status_msg.status = self.nav_state.status
        status_msg.current_pose = self.nav_state.current_pose.pose if self.nav_state.current_pose else Pose()
        status_msg.target_pose = self.nav_state.target_pose.pose if self.nav_state.target_pose else Pose()
        status_msg.progress = self.nav_state.current_index / len(self.nav_state.path) if self.nav_state.path else 0.0
        status_msg.balance_state = self.nav_state.balance_state

        self.navigation_status_pub.publish(status_msg)

    def plan_path_callback(self, request, response):
        """Service callback for path planning"""
        start = PoseStamped()
        start.pose = request.start_pose

        goal = PoseStamped()
        goal.pose = request.goal_pose

        path = self.path_planner.plan_path_to_goal(start, goal, self.navigation_params)

        if path:
            response.success = True
            response.path = path
            self.navigation_metrics['paths_planned'] += 1
        else:
            response.success = False
            response.message = 'Failed to plan path'

        return response

    def get_robot_state_callback(self, request, response):
        """Service callback for getting robot state"""
        response.current_pose = self.nav_state.current_pose.pose if self.nav_state.current_pose else Pose()
        response.current_velocity = self.nav_state.current_velocity if self.nav_state.current_velocity else Twist()
        response.navigation_status = self.nav_state.status
        response.balance_state = self.nav_state.balance_state

        return response

class PathPlanner:
    """
    Advanced path planning for humanoid navigation
    """
    def __init__(self, node):
        self.node = node
        self.costmap = None
        self.resolution = 0.05  # meters per cell
        self.origin = [0, 0]  # map origin in meters

    def update_map(self, occupancy_grid: OccupancyGrid):
        """Update internal costmap representation"""
        self.costmap = np.array(occupancy_grid.data).reshape(
            occupancy_grid.info.height,
            occupancy_grid.info.width
        )
        self.resolution = occupancy_grid.info.resolution
        self.origin = [occupancy_grid.info.origin.position.x, occupancy_grid.info.origin.position.y]

    def plan_path_to_goal(self, start: PoseStamped, goal: PoseStamped, params: Dict[str, Any]) -> Optional[List[PoseStamped]]:
        """Plan path from start to goal considering humanoid constraints"""
        if self.costmap is None:
            self.node.get_logger().error('No costmap available for path planning')
            return None

        # Convert poses to grid coordinates
        start_grid = self.world_to_grid([start.pose.position.x, start.pose.position.y])
        goal_grid = self.world_to_grid([goal.pose.position.x, goal.pose.position.y])

        # Check if start and goal are valid
        if not self.is_valid_cell(start_grid) or not self.is_valid_cell(goal_grid):
            self.node.get_logger().error('Start or goal pose is in invalid location')
            return None

        # Plan path using A* with humanoid-specific constraints
        path_grid = self.a_star_search(start_grid, goal_grid)

        if not path_grid:
            self.node.get_logger().error('No path found')
            return None

        # Convert grid path back to world coordinates
        world_path = []
        for grid_pos in path_grid:
            world_pos = self.grid_to_world(grid_pos)
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = 'map'
            pose_stamped.pose.position.x = world_pos[0]
            pose_stamped.pose.position.y = world_pos[1]
            pose_stamped.pose.position.z = 0.0  # Assume flat terrain for now
            world_path.append(pose_stamped)

        return world_path

    def a_star_search(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """A* path planning algorithm with humanoid constraints"""
        # Define possible movements (8-connected grid)
        movements = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]

        # Calculate heuristic (Euclidean distance)
        def heuristic(pos):
            return math.sqrt((pos[0] - goal[0])**2 + (pos[1] - goal[1])**2)

        # Initialize open and closed sets
        open_set = [(heuristic(start), start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start)}

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]  # Return reversed path

            for dx, dy in movements:
                neighbor = (current[0] + dx, current[1] + dy)

                # Check bounds
                if not self.is_valid_cell(neighbor):
                    continue

                # Check if walkable (cost < 50 means not an obstacle)
                if self.costmap[neighbor[0], neighbor[1]] >= 50:
                    continue

                # Calculate tentative g_score
                movement_cost = math.sqrt(dx**2 + dy**2)  # Diagonal movement costs more
                tentative_g_score = g_score[current] + movement_cost

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None  # No path found

    def world_to_grid(self, world_pos: List[float]) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates"""
        grid_x = int((world_pos[0] - self.origin[0]) / self.resolution)
        grid_y = int((world_pos[1] - self.origin[1]) / self.resolution)
        return (grid_y, grid_x)  # Note: row, col convention for numpy arrays

    def grid_to_world(self, grid_pos: Tuple[int, int]) -> List[float]:
        """Convert grid coordinates to world coordinates"""
        world_x = grid_pos[1] * self.resolution + self.origin[0]
        world_y = grid_pos[0] * self.resolution + self.origin[1]
        return [world_x, world_y]

    def is_valid_cell(self, pos: Tuple[int, int]) -> bool:
        """Check if grid position is valid"""
        if self.costmap is None:
            return False

        row, col = pos
        return (0 <= row < self.costmap.shape[0] and
                0 <= col < self.costmap.shape[1])

class LocalPlanner:
    """
    Local path planning for obstacle avoidance
    """
    def __init__(self, node):
        self.node = node
        self.obstacles = []
        self.local_map = None
        self.lookahead_distance = 2.0  # meters
        self.control_horizon = 1.0  # meters

    def update_obstacles(self, obstacles: List[Dict[str, Any]]):
        """Update with new obstacle information"""
        self.obstacles = obstacles

    def plan_local_trajectory(self, current_pose: PoseStamped, target_pose: PoseStamped) -> List[PoseStamped]:
        """Plan local trajectory considering obstacles"""
        # Generate local path that avoids obstacles
        local_path = []

        # Simple implementation: generate waypoints that avoid obstacles
        current_pos = current_pose.pose.position
        target_pos = target_pose.pose.position

        # Calculate direction to target
        dx = target_pos.x - current_pos.x
        dy = target_pos.y - current_pos.y
        distance = math.sqrt(dx**2 + dy**2)

        if distance < 0.1:  # Already at target
            local_path.append(target_pose)
            return local_path

        # Normalize direction
        direction_x = dx / distance
        direction_y = dy / distance

        # Generate intermediate waypoints
        num_waypoints = min(int(distance / 0.3), 10)  # 30cm between waypoints, max 10
        step_size = distance / num_waypoints if num_waypoints > 0 else 0

        for i in range(num_waypoints + 1):
            # Calculate intermediate position
            interp_x = current_pos.x + direction_x * step_size * i
            interp_y = current_pos.y + direction_y * step_size * i

            # Check for obstacles and adjust if needed
            adjusted_x, adjusted_y = self.avoid_obstacles(interp_x, interp_y, current_pose)

            waypoint = PoseStamped()
            waypoint.header.frame_id = 'map'
            waypoint.pose.position.x = adjusted_x
            waypoint.pose.position.y = adjusted_y
            waypoint.pose.position.z = 0.0

            # Set orientation to face target
            if i < num_waypoints:  # Not the last waypoint
                next_dx = target_pos.x - adjusted_x
                next_dy = target_pos.y - adjusted_y
                angle = math.atan2(next_dy, next_dx)

                # Convert angle to quaternion
                from geometry_msgs.msg import Quaternion
                waypoint.pose.orientation = self.euler_to_quaternion(0, 0, angle)

            local_path.append(waypoint)

        return local_path

    def avoid_obstacles(self, x: float, y: float, current_pose: PoseStamped) -> Tuple[float, float]:
        """Adjust position to avoid obstacles"""
        adjusted_x, adjusted_y = x, y

        for obstacle in self.obstacles:
            obs_x = obstacle['x']
            obs_y = obstacle['y']
            obs_radius = obstacle.get('radius', 0.1)

            # Calculate distance to obstacle
            dist_to_obs = math.sqrt((x - obs_x)**2 + (y - obs_y)**2)

            # If too close, move away
            if dist_to_obs < obs_radius + 0.3:  # 30cm safety margin
                # Calculate direction away from obstacle
                away_x = x - obs_x
                away_y = y - obs_y
                away_dist = math.sqrt(away_x**2 + away_y**2)

                if away_dist > 0:
                    away_x /= away_dist
                    away_y /= away_dist

                    # Move away by safety margin
                    adjusted_x = x + away_x * (obs_radius + 0.3 - dist_to_obs)
                    adjusted_y = y + away_y * (obs_radius + 0.3 - dist_to_obs)

        return adjusted_x, adjusted_y

    def euler_to_quaternion(self, roll: float, pitch: float, yaw: float) -> Quaternion:
        """Convert Euler angles to quaternion"""
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        q = Quaternion()
        q.w = cr * cp * cy + sr * sp * sy
        q.x = sr * cp * cy - cr * sp * sy
        q.y = cr * sp * cy + sr * cp * sy
        q.z = cr * cp * sy - sr * sp * cy

        return q

class NavigationController:
    """
    Navigation controller for generating velocity commands
    """
    def __init__(self, node):
        self.node = node
        self.current_pose = None
        self.current_velocity = None
        self.pid_params = {
            'linear': {'kp': 1.0, 'ki': 0.0, 'kd': 0.0},
            'angular': {'kp': 2.0, 'ki': 0.0, 'kd': 0.0}
        }
        self.linear_error_integral = 0.0
        self.angular_error_integral = 0.0

    def update_current_state(self, pose: PoseStamped, velocity: Twist):
        """Update with current robot state"""
        self.current_pose = pose
        self.current_velocity = velocity

    def compute_velocity_command(self, current_pose: PoseStamped, path: List[PoseStamped], current_velocity: Twist) -> Twist:
        """Compute velocity command based on current pose and desired path"""
        cmd_vel = Twist()

        if not path or len(path) == 0:
            return cmd_vel

        # Get the next point on the path
        target_pose = path[0]

        # Calculate errors
        linear_error, angular_error = self.calculate_errors(current_pose, target_pose)

        # Apply PID control
        linear_vel = self.pid_control(linear_error, self.linear_error_integral, 'linear')
        angular_vel = self.pid_control(angular_error, self.angular_error_integral, 'angular')

        # Apply limits
        linear_vel = max(min(linear_vel, 0.5), -0.5)  # ±0.5 m/s
        angular_vel = max(min(angular_vel, 0.5), -0.5)  # ±0.5 rad/s

        # If very close to target, slow down
        if abs(linear_error) < 0.5:
            linear_vel *= 0.5  # Slow down when approaching target

        cmd_vel.linear.x = linear_vel
        cmd_vel.angular.z = angular_vel

        return cmd_vel

    def calculate_errors(self, current_pose: PoseStamped, target_pose: PoseStamped) -> Tuple[float, float]:
        """Calculate linear and angular errors to target"""
        # Calculate linear error (distance to target)
        dx = target_pose.pose.position.x - current_pose.pose.position.x
        dy = target_pose.pose.position.y - current_pose.pose.position.y
        linear_error = math.sqrt(dx**2 + dy**2)

        # Calculate angular error (direction to target)
        current_yaw = self.quaternion_to_yaw(current_pose.pose.orientation)
        target_yaw = math.atan2(dy, dx)

        # Normalize angular error to [-π, π]
        angular_error = target_yaw - current_yaw
        while angular_error > math.pi:
            angular_error -= 2 * math.pi
        while angular_error < -math.pi:
            angular_error += 2 * math.pi

        return linear_error, angular_error

    def pid_control(self, error: float, integral: float, control_type: str) -> float:
        """PID controller"""
        params = self.pid_params[control_type]
        kp = params['kp']
        ki = params['ki']
        kd = params['kd']

        # Calculate derivative
        # In a real implementation, you'd store the previous error
        derivative = 0.0  # Simplified for this example

        # Calculate output
        output = kp * error + ki * integral + kd * derivative

        return output

    def quaternion_to_yaw(self, orientation) -> float:
        """Convert quaternion to yaw angle"""
        siny_cosp = 2 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy_cosp = 1 - 2 * (orientation.y * orientation.y + orientation.z * orientation.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw

class ObstacleDetector:
    """
    Obstacle detection from sensor data
    """
    def __init__(self, node):
        self.node = node

    def process_scan(self, scan_msg: LaserScan) -> List[Dict[str, Any]]:
        """Process laser scan to detect obstacles"""
        obstacles = []

        # Process each range reading
        for i, range_val in enumerate(scan_msg.ranges):
            if not (math.isnan(range_val) or math.isinf(range_val)) and range_val < 3.0:  # Within 3m
                # Calculate angle of this reading
                angle = scan_msg.angle_min + i * scan_msg.angle_increment

                # Convert to Cartesian coordinates
                x = range_val * math.cos(angle)
                y = range_val * math.sin(angle)

                # Add obstacle with estimated size
                obstacle = {
                    'x': x,
                    'y': y,
                    'range': range_val,
                    'angle': angle,
                    'radius': 0.1  # Estimated obstacle radius
                }
                obstacles.append(obstacle)

        return obstacles

class BalanceMonitor:
    """
    Balance monitoring for humanoid navigation
    """
    def __init__(self, node):
        self.node = node
        self.roll = 0.0
        self.pitch = 0.0
        self.balance_threshold = 0.2  # Radians
        self.balance_state = 'balanced'

    def update_imu_data(self, imu_msg: Imu) -> str:
        """Update with IMU data and return balance state"""
        # Convert quaternion to roll and pitch
        orientation = imu_msg.orientation
        self.roll, self.pitch, _ = self.quaternion_to_euler(
            orientation.x, orientation.y, orientation.z, orientation.w
        )

        # Check if robot is within balance limits
        if abs(self.roll) > self.balance_threshold or abs(self.pitch) > self.balance_threshold:
            self.balance_state = 'unbalanced'
        else:
            self.balance_state = 'balanced'

        return self.balance_state

    def check_obstacle_balance(self, obstacles: List[Dict[str, Any]]):
        """Check if obstacles pose balance threats"""
        for obstacle in obstacles:
            if obstacle['range'] < 0.5:  # Very close obstacle
                # Could affect balance when stopping suddenly
                # This is where you'd implement logic for balance preservation
                pass

    def quaternion_to_euler(self, x, y, z, w) -> Tuple[float, float, float]:
        """Convert quaternion to Euler angles"""
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = math.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw
```

### Advanced Manipulation System

Now let's implement an advanced manipulation system for the humanoid robot:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
from geometry_msgs.msg import PoseStamped, Point, Vector3
from humanoid_robot_msgs.msg import ManipulationGoal, ManipulationStatus
from humanoid_robot_msgs.srv import PlanGrasp, ExecuteGrasp
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import math
from typing import List, Dict, Any, Optional, Tuple
from scipy.spatial.transform import Rotation as R
import tf2_ros
from tf2_ros import TransformException

@dataclass
class ManipulationState:
    """State of the manipulation system"""
    current_pose: Optional[PoseStamped] = None
    target_object: Optional[Dict[str, Any]] = None
    manipulation_status: str = 'idle'
    joint_states: Optional[JointState] = None
    grasp_plan: Optional[Dict[str, Any]] = None

class AdvancedManipulationSystem(Node):
    """
    Advanced manipulation system for humanoid robots
    """
    def __init__(self):
        super().__init__('advanced_manipulation_system')

        # Initialize manipulation state
        self.manip_state = ManipulationState()

        # Initialize manipulation components
        self.grasp_planner = GraspPlanner(self)
        self.motion_planner = MotionPlanner(self)
        self.controller = ManipulationController(self)
        self.vision_processor = VisionProcessor(self)

        # TF buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Set up ROS 2 interfaces
        self.setup_interfaces()

        # Manipulation parameters
        self.manipulation_params = {
            'reach_distance': 1.0,
            'grasp_types': ['pinch', 'power', 'hook'],
            'max_load': 2.0,
            'approach_distance': 0.1,
            'retreat_distance': 0.1,
            'grasp_tolerance': 0.02
        }

        # Manipulation thread
        self.manipulation_thread = None
        self.manipulation_active = False

        # Performance monitoring
        self.manipulation_metrics = {
            'grasps_attempted': 0,
            'grasps_successful': 0,
            'average_grasp_time': 0.0,
            'manipulations_completed': 0
        }

        self.get_logger().info('Advanced Manipulation System initialized')

    def setup_interfaces(self):
        """Set up ROS 2 communication interfaces"""
        # Publishers
        self.manip_status_pub = self.create_publisher(ManipulationStatus, '/manipulation/status', 10)
        self.joint_cmd_pub = self.create_publisher(JointState, '/joint_commands', 10)
        self.visualization_pub = self.create_publisher(MarkerArray, '/manipulation/visualization', 10)

        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )
        self.vision_sub = self.create_subscription(
            Image, '/camera/image_raw', self.vision_callback, 10
        )

        # Services
        self.plan_grasp_srv = self.create_service(
            PlanGrasp, '/manipulation/plan_grasp', self.plan_grasp_callback
        )
        self.execute_grasp_srv = self.create_service(
            ExecuteGrasp, '/manipulation/execute_grasp', self.execute_grasp_callback
        )

    def joint_state_callback(self, msg):
        """Handle joint state updates"""
        self.manip_state.joint_states = msg

        # Update controller with current joint states
        self.controller.update_joint_states(msg)

    def vision_callback(self, msg):
        """Handle vision updates for object detection"""
        # Process image to detect objects
        detected_objects = self.vision_processor.process_image(msg)

        # Update vision-based state
        if detected_objects:
            # For now, just update the list of detected objects
            self.manip_state.detected_objects = detected_objects

    def plan_grasp_callback(self, request, response):
        """Service callback for grasp planning"""
        object_info = {
            'class': request.object_class,
            'position': [request.object_pose.position.x, request.object_pose.position.y, request.object_pose.position.z],
            'orientation': [request.object_pose.orientation.x, request.object_pose.orientation.y,
                           request.object_pose.orientation.z, request.object_pose.orientation.w],
            'dimensions': request.object_dimensions
        }

        grasp_plan = self.grasp_planner.plan_grasp(object_info, self.manipulation_params)

        if grasp_plan:
            response.success = True
            response.grasp_plan = grasp_plan
            self.manipulation_metrics['grasps_planned'] = self.manipulation_metrics.get('grasps_planned', 0) + 1
        else:
            response.success = False
            response.message = 'Failed to plan grasp'

        return response

    def execute_grasp_callback(self, request, response):
        """Service callback for grasp execution"""
        self.manip_state.target_object = request.object_info
        self.manip_state.grasp_plan = request.grasp_plan
        self.manip_state.manipulation_status = 'planning'

        # Plan manipulation trajectory
        trajectory = self.motion_planner.plan_manipulation_trajectory(
            self.manip_state.joint_states,
            request.grasp_plan
        )

        if trajectory:
            self.manip_state.manipulation_status = 'executing'

            # Start manipulation thread
            if not self.manipulation_active:
                self.start_manipulation(trajectory)

            response.success = True
            response.message = 'Grasp execution started'
        else:
            response.success = False
            response.message = 'Failed to plan manipulation trajectory'

        return response

    def start_manipulation(self, trajectory: List[Dict[str, Any]]):
        """Start manipulation execution thread"""
        if self.manipulation_thread is None or not self.manipulation_thread.is_alive():
            self.manipulation_active = True
            self.manipulation_thread = threading.Thread(
                target=self.manipulation_loop,
                args=(trajectory,),
                daemon=True
            )
            self.manipulation_thread.start()

    def manipulation_loop(self, trajectory: List[Dict[str, Any]]):
        """Main manipulation execution loop"""
        for waypoint in trajectory:
            if not self.manipulation_active or not rclpy.ok():
                break

            # Execute the waypoint
            success = self.controller.execute_waypoint(waypoint)

            if not success:
                self.manip_state.manipulation_status = 'failed'
                self.publish_manipulation_status()
                break

            # Small delay between waypoints
            time.sleep(0.1)

        if self.manip_state.manipulation_status == 'executing':
            self.manip_state.manipulation_status = 'completed'
            self.manipulation_metrics['manipulations_completed'] += 1

        self.publish_manipulation_status()
        self.manipulation_active = False

    def publish_manipulation_status(self):
        """Publish current manipulation status"""
        status_msg = ManipulationStatus()
        status_msg.header.stamp = self.get_clock().now().to_msg()
        status_msg.status = self.manip_state.manipulation_status
        status_msg.target_object = self.manip_state.target_object if self.manip_state.target_object else {}
        status_msg.progress = 0.0  # Would be calculated based on trajectory progress

        self.manip_status_pub.publish(status_msg)

    def detect_object_in_workspace(self, object_class: str) -> Optional[Dict[str, Any]]:
        """Detect object of specific class in robot workspace"""
        # This would interface with the vision system to find objects
        # For simulation, return a mock object
        return {
            'class': object_class,
            'position': [0.5, 0.2, 0.1],  # Relative to robot
            'orientation': [0, 0, 0, 1],
            'dimensions': [0.08, 0.08, 0.1],  # width, depth, height
            'confidence': 0.9
        }

class GraspPlanner:
    """
    Grasp planning for humanoid manipulation
    """
    def __init__(self, node):
        self.node = node
        self.grasp_database = self.load_grasp_database()

    def load_grasp_database(self) -> Dict[str, Any]:
        """Load grasp database with precomputed grasps for common objects"""
        return {
            'cup': {
                'grasp_types': ['pinch', 'power'],
                'approach_directions': [[0, 0, 1], [0, 1, 0]],  # From top, from side
                'grasp_points': [[0.04, 0, 0.05], [-0.04, 0, 0.05]],  # Handle sides
                'preferred_grasp': 'pinch'
            },
            'bottle': {
                'grasp_types': ['power', 'pinch'],
                'approach_directions': [[0, 0, 1], [0, 1, 0]],
                'grasp_points': [[0, 0, 0.1], [0, 0, 0.2]],  # Neck area
                'preferred_grasp': 'power'
            },
            'book': {
                'grasp_types': ['power', 'pinch'],
                'approach_directions': [[0, 1, 0], [1, 0, 0]],
                'grasp_points': [[0.1, 0, 0.015], [-0.1, 0, 0.015]],  # Spine
                'preferred_grasp': 'power'
            }
        }

    def plan_grasp(self, object_info: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Plan grasp for the given object"""
        obj_class = object_info['class']
        obj_pos = object_info['position']
        obj_orient = object_info['orientation']
        obj_dims = object_info['dimensions']

        if obj_class not in self.grasp_database:
            self.node.get_logger().warn(f'No grasp database entry for {obj_class}, using generic grasp')
            return self.plan_generic_grasp(object_info, params)

        # Get object-specific grasp information
        obj_grasps = self.grasp_database[obj_class]

        # Select the best grasp based on accessibility and robot constraints
        best_grasp = self.select_best_grasp(obj_grasps, obj_pos, obj_orient, params)

        if best_grasp:
            # Transform grasp pose to world frame
            grasp_pose = self.transform_grasp_to_world(best_grasp, obj_pos, obj_orient)

            return {
                'grasp_type': best_grasp['type'],
                'grasp_pose': grasp_pose,
                'approach_direction': best_grasp['approach_direction'],
                'grasp_points': best_grasp['grasp_points'],
                'confidence': 0.8,
                'object_class': obj_class
            }

        return None

    def select_best_grasp(self, obj_grasps: Dict[str, Any], obj_pos: List[float],
                         obj_orient: List[float], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Select the best grasp based on accessibility and constraints"""
        # For now, select the preferred grasp if it's accessible
        preferred_type = obj_grasps['preferred_grasp']

        # Check if preferred grasp is accessible
        for i, grasp_point in enumerate(obj_grasps['grasp_points']):
            # Transform grasp point to world coordinates
            world_grasp_point = self.transform_point_to_world(
                grasp_point, obj_pos, obj_orient
            )

            # Check if this point is reachable
            if self.is_reachable(world_grasp_point, params):
                return {
                    'type': preferred_type,
                    'grasp_point': grasp_point,
                    'approach_direction': obj_grasps['approach_directions'][i % len(obj_grasps['approach_directions'])],
                    'grasp_points': [grasp_point]
                }

        # If preferred grasp is not accessible, try other grasp types
        for grasp_type in obj_grasps['grasp_types']:
            if grasp_type != preferred_type:
                for i, grasp_point in enumerate(obj_grasps['grasp_points']):
                    world_grasp_point = self.transform_point_to_world(
                        grasp_point, obj_pos, obj_orient
                    )

                    if self.is_reachable(world_grasp_point, params):
                        return {
                            'type': grasp_type,
                            'grasp_point': grasp_point,
                            'approach_direction': obj_grasps['approach_directions'][i % len(obj_grasps['approach_directions'])],
                            'grasp_points': [grasp_point]
                        }

        return None

    def is_reachable(self, point: List[float], params: Dict[str, Any]) -> bool:
        """Check if a point is reachable by the robot"""
        # Simple reachability check based on reach distance
        # In practice, this would use inverse kinematics
        distance_from_robot = math.sqrt(point[0]**2 + point[1]**2 + point[2]**2)
        return distance_from_robot <= params['reach_distance']

    def transform_point_to_world(self, local_point: List[float], obj_pos: List[float],
                                obj_orient: List[float]) -> List[float]:
        """Transform a point from object local coordinates to world coordinates"""
        # Convert quaternion to rotation matrix
        rot = R.from_quat(obj_orient)
        rotation_matrix = rot.as_matrix()

        # Transform the point
        world_point = np.dot(rotation_matrix, local_point) + obj_pos
        return world_point.tolist()

    def transform_grasp_to_world(self, grasp_info: Dict[str, Any], obj_pos: List[float],
                                obj_orient: List[float]) -> Dict[str, Any]:
        """Transform grasp information to world frame"""
        grasp_pose = {
            'position': self.transform_point_to_world(
                grasp_info['grasp_point'], obj_pos, obj_orient
            ),
            'orientation': obj_orient  # For now, use object orientation
        }

        return grasp_pose

    def plan_generic_grasp(self, object_info: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Plan a generic grasp for unknown objects"""
        obj_pos = object_info['position']
        obj_dims = object_info['dimensions']

        # For generic objects, use center + dimensions to estimate grasp points
        # Prefer power grasp for larger objects, pinch for smaller ones
        largest_dim = max(obj_dims)
        grasp_type = 'pinch' if largest_dim < 0.08 else 'power'

        # Use center of object as grasp point
        grasp_point = obj_pos

        # Approach from above
        approach_direction = [0, 0, 1]

        return {
            'grasp_type': grasp_type,
            'grasp_pose': {
                'position': grasp_point,
                'orientation': [0, 0, 0, 1]  # Identity orientation
            },
            'approach_direction': approach_direction,
            'grasp_points': [grasp_point],
            'confidence': 0.6
        }

class MotionPlanner:
    """
    Motion planning for manipulation trajectories
    """
    def __init__(self, node):
        self.node = node
        self.ik_solver = None  # Would be initialized with robot-specific IK solver

    def plan_manipulation_trajectory(self, current_joints: JointState, grasp_plan: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """Plan complete manipulation trajectory: approach -> grasp -> lift -> retreat"""
        if not current_joints:
            self.node.get_logger().error('No current joint states available')
            return None

        trajectory = []

        # 1. Plan approach trajectory
        approach_traj = self.plan_approach_trajectory(current_joints, grasp_plan)
        if approach_traj:
            trajectory.extend(approach_traj)
        else:
            self.node.get_logger().error('Failed to plan approach trajectory')
            return None

        # 2. Add grasp command
        trajectory.append({
            'type': 'gripper_command',
            'command': 'close',
            'parameters': {'force': 10.0, 'position': 0.0}  # Close gripper
        })

        # 3. Plan lift trajectory
        lift_traj = self.plan_lift_trajectory(current_joints, grasp_plan)
        if lift_traj:
            trajectory.extend(lift_traj)

        # 4. Plan retreat trajectory
        retreat_traj = self.plan_retreat_trajectory(current_joints, grasp_plan)
        if retreat_traj:
            trajectory.extend(retreat_traj)

        return trajectory

    def plan_approach_trajectory(self, current_joints: JointState, grasp_plan: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """Plan approach trajectory to grasp point"""
        # Calculate approach position (a bit above the grasp point)
        grasp_pos = grasp_plan['grasp_pose']['position']
        approach_pos = [
            grasp_pos[0],
            grasp_pos[1],
            grasp_pos[2] + 0.1  # 10cm above grasp point
        ]

        # Plan trajectory from current position to approach position
        # This would use inverse kinematics to generate joint trajectories
        # For simulation, we'll create a simple trajectory

        return [
            {
                'type': 'joint_trajectory',
                'waypoints': [
                    # Waypoints would be calculated using IK
                    # For simulation, use mock values
                    {
                        'positions': self.interpolate_joints(current_joints.position, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
                        'time_from_start': 1.0
                    },
                    {
                        'positions': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                        'time_from_start': 2.0
                    }
                ]
            }
        ]

    def plan_lift_trajectory(self, current_joints: JointState, grasp_plan: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """Plan lift trajectory after grasping"""
        # Lift the object slightly
        return [
            {
                'type': 'joint_trajectory',
                'waypoints': [
                    {
                        'positions': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                        'time_from_start': 1.0
                    }
                ]
            }
        ]

    def plan_retreat_trajectory(self, current_joints: JointState, grasp_plan: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """Plan retreat trajectory after lifting"""
        # Retreat to a safe position
        return [
            {
                'type': 'joint_trajectory',
                'waypoints': [
                    {
                        'positions': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Return to home position
                        'time_from_start': 2.0
                    }
                ]
            }
        ]

    def interpolate_joints(self, start_pos: List[float], end_pos: List[float], steps: int = 10) -> List[List[float]]:
        """Interpolate between joint positions"""
        # Simple linear interpolation for simulation
        interpolated = []
        for i in range(steps):
            t = i / (steps - 1) if steps > 1 else 0
            pos = [start + t * (end - start) for start, end in zip(start_pos, end_pos)]
            interpolated.append(pos)
        return interpolated

class ManipulationController:
    """
    Controller for executing manipulation commands
    """
    def __init__(self, node):
        self.node = node
        self.current_joints = None
        self.gripper_controller = None  # Would interface with gripper controller

    def update_joint_states(self, joint_states: JointState):
        """Update with current joint states"""
        self.current_joints = joint_states

    def execute_waypoint(self, waypoint: Dict[str, Any]) -> bool:
        """Execute a single waypoint in the manipulation trajectory"""
        waypoint_type = waypoint['type']

        if waypoint_type == 'joint_trajectory':
            return self.execute_joint_trajectory(waypoint)
        elif waypoint_type == 'gripper_command':
            return self.execute_gripper_command(waypoint)
        else:
            self.node.get_logger().error(f'Unknown waypoint type: {waypoint_type}')
            return False

    def execute_joint_trajectory(self, waypoint: Dict[str, Any]) -> bool:
        """Execute joint trajectory waypoint"""
        waypoints = waypoint['waypoints']

        for wp in waypoints:
            positions = wp['positions']
            duration = wp['time_from_start']

            # Create joint command
            joint_cmd = JointState()
            joint_cmd.header.stamp = self.node.get_clock().now().to_msg()
            joint_cmd.name = self.get_robot_joint_names()  # Would return actual joint names
            joint_cmd.position = positions

            # Publish joint command
            self.node.joint_cmd_pub.publish(joint_cmd)

            # Wait for execution (in practice, monitor actual joint positions)
            time.sleep(duration)

        return True

    def execute_gripper_command(self, waypoint: Dict[str, Any]) -> bool:
        """Execute gripper command"""
        command = waypoint['command']
        parameters = waypoint['parameters']

        if command == 'close':
            # Close gripper
            self.close_gripper(parameters)
        elif command == 'open':
            # Open gripper
            self.open_gripper(parameters)
        else:
            self.node.get_logger().error(f'Unknown gripper command: {command}')
            return False

        return True

    def close_gripper(self, params: Dict[str, Any]):
        """Close gripper with specified parameters"""
        # In practice, this would interface with gripper controller
        force = params.get('force', 10.0)
        position = params.get('position', 0.0)

        self.node.get_logger().info(f'Closing gripper with force {force}N')

    def open_gripper(self, params: Dict[str, Any]):
        """Open gripper with specified parameters"""
        # In practice, this would interface with gripper controller
        self.node.get_logger().info('Opening gripper')

    def get_robot_joint_names(self) -> List[str]:
        """Get robot joint names (would be robot-specific)"""
        # Return mock joint names for simulation
        return ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']

class VisionProcessor:
    """
    Vision processing for manipulation
    """
    def __init__(self, node):
        self.node = node
        self.object_detector = None  # Would be initialized with actual detector

    def process_image(self, image_msg) -> List[Dict[str, Any]]:
        """Process image to detect objects for manipulation"""
        # In practice, this would run object detection
        # For simulation, return mock detections
        return [
            {
                'class': 'cup',
                'position': [0.5, 0.2, 0.1],
                'orientation': [0, 0, 0, 1],
                'dimensions': [0.08, 0.08, 0.1],
                'confidence': 0.9
            },
            {
                'class': 'book',
                'position': [0.8, -0.1, 0.05],
                'orientation': [0, 0, 0, 1],
                'dimensions': [0.2, 0.15, 0.03],
                'confidence': 0.85
            }
        ]

    def detect_object_in_hand(self) -> Optional[Dict[str, Any]]:
        """Detect if robot is holding an object"""
        # In practice, this would use force/torque sensors or wrist-mounted camera
        # For simulation, return mock result
        return None
```

### Autonomous Behavior Coordination

Now let's implement the coordination between navigation and manipulation for autonomous behaviors:

```python
class AutonomousBehaviorSystem(Node):
    """
    System for coordinating autonomous navigation and manipulation behaviors
    """
    def __init__(self):
        super().__init__('autonomous_behavior_system')

        # Initialize subsystems
        self.navigation_system = AdvancedNavigationSystem(self)
        self.manipulation_system = AdvancedManipulationSystem(self)

        # Behavior state
        self.current_behavior = 'idle'
        self.behavior_queue = []
        self.behavior_active = False

        # Set up interfaces
        self.setup_behavior_interfaces()

        # Behavior coordination
        self.coordinator = BehaviorCoordinator(self)

        self.get_logger().info('Autonomous Behavior System initialized')

    def setup_behavior_interfaces(self):
        """Set up interfaces for behavior coordination"""
        # Publishers
        self.behavior_status_pub = self.create_publisher(String, '/behavior/status', 10)

        # Services for high-level behavior commands
        self.execute_behavior_srv = self.create_service(
            ExecuteBehavior, '/behavior/execute', self.execute_behavior_callback
        )

    def execute_behavior_callback(self, request, response):
        """Execute a high-level behavior"""
        behavior_type = request.behavior_type
        parameters = request.parameters

        if behavior_type == 'fetch_object':
            response.success = self.execute_fetch_behavior(parameters)
        elif behavior_type == 'navigate_to_location':
            response.success = self.execute_navigation_behavior(parameters)
        elif behavior_type == 'deliver_object':
            response.success = self.execute_delivery_behavior(parameters)
        elif behavior_type == 'inspect_area':
            response.success = self.execute_inspection_behavior(parameters)
        else:
            response.success = False
            response.message = f'Unknown behavior type: {behavior_type}'

        return response

    def execute_fetch_behavior(self, parameters: Dict[str, Any]) -> bool:
        """Execute object fetching behavior"""
        try:
            # 1. Navigate to object location
            object_class = parameters.get('object_class', 'unknown')
            target_location = parameters.get('target_location', 'default')

            # Detect object in current vicinity
            detected_objects = self.manipulation_system.detect_object_in_workspace(object_class)

            if not detected_objects:
                # Navigate to target location to find object
                goal_pose = self.create_pose_from_location(target_location)
                self.navigation_system.set_navigation_goal(goal_pose)

                # Wait for navigation to complete
                if not self.wait_for_navigation_completion():
                    self.get_logger().error('Navigation to object location failed')
                    return False

                # Detect object again after reaching location
                detected_objects = self.manipulation_system.detect_object_in_workspace(object_class)

            if not detected_objects:
                self.get_logger().error(f'Could not find {object_class} at location')
                return False

            # 2. Plan and execute grasp
            object_info = detected_objects[0]  # Take first detected object
            grasp_plan = self.manipulation_system.grasp_planner.plan_grasp(
                object_info,
                self.manipulation_system.manipulation_params
            )

            if not grasp_plan:
                self.get_logger().error(f'Could not plan grasp for {object_class}')
                return False

            # Execute the grasp
            success = self.execute_grasp_with_verification(grasp_plan, object_info)

            if not success:
                self.get_logger().error(f'Grasp of {object_class} failed')
                return False

            # 3. Navigate back to user or designated location
            return_location = parameters.get('return_location', 'user')
            return_pose = self.create_pose_from_location(return_location)
            self.navigation_system.set_navigation_goal(return_pose)

            return True

        except Exception as e:
            self.get_logger().error(f'Fetch behavior error: {e}')
            return False

    def execute_navigation_behavior(self, parameters: Dict[str, Any]) -> bool:
        """Execute navigation behavior"""
        try:
            target_location = parameters.get('target_location', 'default')
            goal_pose = self.create_pose_from_location(target_location)

            self.navigation_system.set_navigation_goal(goal_pose)

            return self.wait_for_navigation_completion()

        except Exception as e:
            self.get_logger().error(f'Navigation behavior error: {e}')
            return False

    def execute_delivery_behavior(self, parameters: Dict[str, Any]) -> bool:
        """Execute object delivery behavior"""
        try:
            # 1. Navigate to delivery location
            delivery_location = parameters.get('delivery_location', 'default')
            delivery_pose = self.create_pose_from_location(delivery_location)

            self.navigation_system.set_navigation_goal(delivery_pose)

            if not self.wait_for_navigation_completion():
                self.get_logger().error('Navigation to delivery location failed')
                return False

            # 2. Place object (simplified - assumes object is already grasped)
            # This would involve releasing the gripper
            release_cmd = {
                'type': 'gripper_command',
                'command': 'open',
                'parameters': {'position': 1.0}  # Fully open
            }

            success = self.manipulation_system.controller.execute_waypoint(release_cmd)

            if success:
                # Update manipulation state to indicate object is released
                self.manipulation_system.manip_state.target_object = None

            return success

        except Exception as e:
            self.get_logger().error(f'Delivery behavior error: {e}')
            return False

    def execute_inspection_behavior(self, parameters: Dict[str, Any]) -> bool:
        """Execute area inspection behavior"""
        try:
            inspection_area = parameters.get('inspection_area', 'default')
            inspection_pose = self.create_pose_from_location(inspection_area)

            # Navigate to inspection position
            self.navigation_system.set_navigation_goal(inspection_pose)

            if not self.wait_for_navigation_completion():
                self.get_logger().error('Navigation to inspection location failed')
                return False

            # Perform inspection actions
            # This could involve turning, looking around, taking pictures, etc.
            self.perform_inspection_actions()

            return True

        except Exception as e:
            self.get_logger().error(f'Inspection behavior error: {e}')
            return False

    def create_pose_from_location(self, location: str) -> PoseStamped:
        """Create a pose from a named location"""
        # In practice, this would look up location in a map
        # For simulation, return mock poses
        location_poses = {
            'kitchen': [2.0, 1.0, 0.0],
            'living_room': [0.0, 0.0, 0.0],
            'bedroom': [-2.0, 1.0, 0.0],
            'office': [1.0, -2.0, 0.0],
            'user': [0.5, 0.5, 0.0]  # Near user position
        }

        pos = location_poses.get(location, [0.0, 0.0, 0.0])

        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = pos[0]
        pose.pose.position.y = pos[1]
        pose.pose.position.z = pos[2]
        pose.pose.orientation.w = 1.0  # Identity orientation

        return pose

    def wait_for_navigation_completion(self) -> bool:
        """Wait for navigation to complete"""
        start_time = time.time()
        timeout = 30.0  # 30 second timeout

        while (time.time() - start_time) < timeout:
            if self.navigation_system.nav_state.status == 'completed':
                return True
            elif self.navigation_system.nav_state.status == 'path_failed':
                return False
            time.sleep(0.1)

        return False  # Timeout

    def execute_grasp_with_verification(self, grasp_plan: Dict[str, Any], object_info: Dict[str, Any]) -> bool:
        """Execute grasp and verify success"""
        # Execute the grasp
        success = self.manipulation_system.execute_grasp_callback(
            ExecuteGrasp.Request(
                object_info=object_info,
                grasp_plan=grasp_plan
            ),
            ExecuteGrasp.Response()
        ).success

        if not success:
            return False

        # Wait for manipulation to complete
        start_time = time.time()
        timeout = 10.0

        while (time.time() - start_time) < timeout:
            if self.manipulation_system.manip_state.manipulation_status in ['completed', 'failed']:
                break
            time.sleep(0.1)

        # Verify grasp success (in practice, use force/torque sensors)
        return self.manipulation_system.manip_state.manipulation_status == 'completed'

    def perform_inspection_actions(self):
        """Perform inspection-related actions"""
        # Turn robot to look around
        # Take pictures
        # Scan with laser
        # Detect objects
        # etc.
        pass

class BehaviorCoordinator:
    """
    Coordinator for managing complex multi-step behaviors
    """
    def __init__(self, behavior_system):
        self.system = behavior_system
        self.active_behaviors = []
        self.behavior_history = []

    def coordinate_complex_behavior(self, behavior_sequence: List[Dict[str, Any]]) -> bool:
        """Coordinate execution of a sequence of behaviors"""
        for i, behavior in enumerate(behavior_sequence):
            self.system.get_logger().info(f'Executing behavior {i+1}/{len(behavior_sequence)}: {behavior["type"]}')

            # Execute the behavior
            success = self.execute_single_behavior(behavior)

            if not success:
                self.system.get_logger().error(f'Behavior {behavior["type"]} failed')
                return False

            # Log successful behavior
            self.behavior_history.append({
                'behavior': behavior,
                'timestamp': time.time(),
                'success': True
            })

        return True

    def execute_single_behavior(self, behavior: Dict[str, Any]) -> bool:
        """Execute a single behavior"""
        behavior_type = behavior['type']
        parameters = behavior.get('parameters', {})

        # Use the main behavior execution methods
        if behavior_type == 'fetch_object':
            return self.system.execute_fetch_behavior(parameters)
        elif behavior_type == 'navigate_to_location':
            return self.system.execute_navigation_behavior(parameters)
        elif behavior_type == 'deliver_object':
            return self.system.execute_delivery_behavior(parameters)
        elif behavior_type == 'inspect_area':
            return self.system.execute_inspection_behavior(parameters)
        else:
            self.system.get_logger().error(f'Unknown behavior type: {behavior_type}')
            return False

    def handle_behavior_interruption(self, interruption_reason: str):
        """Handle interruption of ongoing behavior"""
        # Stop current navigation if active
        if self.system.navigation_system.navigation_active:
            self.system.navigation_system.stop_navigation()

        # Stop current manipulation if active
        if self.system.manipulation_system.manipulation_active:
            self.system.manipulation_system.manipulation_active = False

        # Log the interruption
        self.system.get_logger().warn(f'Behavior interrupted: {interruption_reason}')

    def resume_behavior_after_interruption(self):
        """Resume behavior after interruption"""
        # This would implement behavior recovery logic
        pass

def main(args=None):
    rclpy.init(args=args)

    behavior_system = AutonomousBehaviorSystem()

    try:
        rclpy.spin(behavior_system)
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup
        behavior_system.navigation_system.stop_navigation()
        behavior_system.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Behavior Trees for Complex Autonomy

For more sophisticated autonomous behaviors, let's implement a behavior tree system:

```python
from enum import Enum
from abc import ABC, abstractmethod
import random

class Status(Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RUNNING = "RUNNING"

class BehaviorNode(ABC):
    """Base class for behavior tree nodes"""
    def __init__(self, name: str):
        self.name = name
        self.status = Status.RUNNING

    @abstractmethod
    def tick(self, blackboard: Dict[str, Any]) -> Status:
        """Execute the behavior and return status"""
        pass

class SequenceNode(BehaviorNode):
    """Sequence node - executes children in order until one fails"""
    def __init__(self, name: str, children: List[BehaviorNode]):
        super().__init__(name)
        self.children = children
        self.current_child_idx = 0

    def tick(self, blackboard: Dict[str, Any]) -> Status:
        while self.current_child_idx < len(self.children):
            child_status = self.children[self.current_child_idx].tick(blackboard)

            if child_status == Status.FAILURE:
                self.current_child_idx = 0
                return Status.FAILURE
            elif child_status == Status.RUNNING:
                return Status.RUNNING
            else:  # SUCCESS
                self.current_child_idx += 1

        # All children succeeded
        self.current_child_idx = 0
        return Status.SUCCESS

class SelectorNode(BehaviorNode):
    """Selector node - executes children until one succeeds"""
    def __init__(self, name: str, children: List[BehaviorNode]):
        super().__init__(name)
        self.children = children
        self.current_child_idx = 0

    def tick(self, blackboard: Dict[str, Any]) -> Status:
        while self.current_child_idx < len(self.children):
            child_status = self.children[self.current_child_idx].tick(blackboard)

            if child_status == Status.SUCCESS:
                self.current_child_idx = 0
                return Status.SUCCESS
            elif child_status == Status.RUNNING:
                return Status.RUNNING
            else:  # FAILURE
                self.current_child_idx += 1

        # All children failed
        self.current_child_idx = 0
        return Status.FAILURE

class ConditionNode(BehaviorNode):
    """Condition node - checks a condition"""
    def __init__(self, name: str, condition_func):
        super().__init__(name)
        self.condition_func = condition_func

    def tick(self, blackboard: Dict[str, Any]) -> Status:
        if self.condition_func(blackboard):
            return Status.SUCCESS
        else:
            return Status.FAILURE

class ActionNode(BehaviorNode):
    """Action node - performs an action"""
    def __init__(self, name: str, action_func):
        super().__init__(name)
        self.action_func = action_func

    def tick(self, blackboard: Dict[str, Any]) -> Status:
        return self.action_func(blackboard)

# Concrete behavior implementations
class CheckBatteryLevel(ConditionNode):
    """Check if battery level is above threshold"""
    def __init__(self, threshold: float = 20.0):
        super().__init__("CheckBatteryLevel", self.check_condition)
        self.threshold = threshold

    def check_condition(self, blackboard: Dict[str, Any]) -> bool:
        battery_level = blackboard.get('battery_level', 100.0)
        return battery_level > self.threshold

class NavigateToLocation(ActionNode):
    """Navigate to a specific location"""
    def __init__(self, location: str):
        super().__init__(f"NavigateTo{location}", self.execute_action)
        self.location = location

    def execute_action(self, blackboard: Dict[str, Any]) -> Status:
        # In practice, this would interface with navigation system
        print(f"Navigating to {self.location}")
        # Simulate navigation
        time.sleep(1.0)
        return Status.SUCCESS

class DetectObject(ActionNode):
    """Detect an object of specific class"""
    def __init__(self, object_class: str):
        super().__init__(f"Detect{object_class}", self.execute_action)
        self.object_class = object_class

    def execute_action(self, blackboard: Dict[str, Any]) -> Status:
        # In practice, this would interface with vision system
        # For simulation, randomly detect object
        detected = random.random() > 0.3  # 70% chance of detection

        if detected:
            blackboard[f'{self.object_class}_detected'] = True
            blackboard[f'{self.object_class}_position'] = [random.uniform(-1, 1), random.uniform(-1, 1), 0]
            return Status.SUCCESS
        else:
            return Status.FAILURE

class GraspObject(ActionNode):
    """Grasp a detected object"""
    def __init__(self, object_class: str):
        super().__init__(f"Grasp{object_class}", self.execute_action)
        self.object_class = object_class

    def execute_action(self, blackboard: Dict[str, Any]) -> Status:
        # Check if object is detected
        if blackboard.get(f'{self.object_class}_detected', False):
            # In practice, this would interface with manipulation system
            print(f"Grasping {self.object_class}")
            time.sleep(1.0)
            blackboard[f'{self.object_class}_grasped'] = True
            return Status.SUCCESS
        else:
            return Status.FAILURE

class DeliverObject(ActionNode):
    """Deliver object to target location"""
    def __init__(self, target_location: str):
        super().__init__(f"DeliverToObject", self.execute_action)
        self.target_location = target_location

    def execute_action(self, blackboard: Dict[str, Any]) -> Status:
        # In practice, this would interface with navigation and manipulation systems
        print(f"Delivering object to {self.target_location}")
        time.sleep(1.0)
        blackboard['object_delivered'] = True
        return Status.SUCCESS

class AutonomousBehaviorTree:
    """
    Behavior tree for autonomous humanoid robot behaviors
    """
    def __init__(self, robot_node):
        self.node = robot_node
        self.blackboard = {
            'battery_level': 85.0,
            'current_location': 'home_base',
            'target_object': None,
            'delivery_location': 'user_station'
        }
        self.root = self.build_behavior_tree()

    def build_behavior_tree(self) -> BehaviorNode:
        """Build the main behavior tree"""
        # Main selector: if battery low, recharge; otherwise, perform tasks
        main_selector = SelectorNode("MainBehavior", [
            # Condition: battery low
            SequenceNode("RechargeSequence", [
                ConditionNode("BatteryLow", lambda bb: bb['battery_level'] < 20.0),
                NavigateToLocation("charging_station"),
                ActionNode("ChargeBattery", lambda bb: self.charge_battery(bb))
            ]),
            # Condition: perform tasks
            SequenceNode("TaskExecutionSequence", [
                CheckBatteryLevel(30.0),  # Must have sufficient battery
                self.build_task_sequence()
            ])
        ])

        return main_selector

    def build_task_sequence(self) -> BehaviorNode:
        """Build sequence for task execution"""
        return SelectorNode("TaskSelector", [
            # Fetch task
            SequenceNode("FetchTask", [
                ConditionNode("FetchRequested", lambda bb: bb.get('fetch_request', False)),
                NavigateToLocation("object_area"),
                DetectObject("cup"),
                GraspObject("cup"),
                NavigateToLocation("delivery_location"),
                DeliverObject("delivery_location"),
                ActionNode("ResetFetchRequest", lambda bb: self.reset_fetch_request(bb))
            ]),
            # Patrol task
            SequenceNode("PatrolTask", [
                ConditionNode("PatrolRequested", lambda bb: bb.get('patrol_request', False)),
                NavigateToLocation("patrol_point_1"),
                NavigateToLocation("patrol_point_2"),
                NavigateToLocation("patrol_point_3"),
                ActionNode("ResetPatrolRequest", lambda bb: self.reset_patrol_request(bb))
            ])
        ])

    def charge_battery(self, blackboard: Dict[str, Any]) -> Status:
        """Simulate battery charging"""
        print("Charging battery...")
        time.sleep(2.0)
        blackboard['battery_level'] = 100.0
        return Status.SUCCESS

    def reset_fetch_request(self, blackboard: Dict[str, Any]) -> Status:
        """Reset fetch request after completion"""
        blackboard['fetch_request'] = False
        return Status.SUCCESS

    def reset_patrol_request(self, blackboard: Dict[str, Any]) -> Status:
        """Reset patrol request after completion"""
        blackboard['patrol_request'] = False
        return Status.SUCCESS

    def tick(self) -> Status:
        """Execute one cycle of the behavior tree"""
        return self.root.tick(self.blackboard)

    def update_blackboard(self, key: str, value: Any):
        """Update blackboard with new information"""
        self.blackboard[key] = value

    def get_blackboard(self) -> Dict[str, Any]:
        """Get current blackboard state"""
        return self.blackboard.copy()
```

### Summary

In this chapter, we've implemented comprehensive autonomous navigation and manipulation capabilities:

- **Advanced Navigation System**: Complete navigation stack with path planning, obstacle avoidance, and balance monitoring
- **Advanced Manipulation System**: Grasp planning, motion planning, and execution control
- **Behavior Coordination**: Coordination between navigation and manipulation for complex tasks
- **Behavior Trees**: Sophisticated autonomous behavior trees for complex multi-step tasks

These systems enable humanoid robots to operate autonomously, performing complex tasks like fetching objects, navigating to destinations, and interacting with their environment. The integration of navigation and manipulation allows for sophisticated behaviors that combine mobility and dexterity.

In the next chapter, we'll cover testing and validation of the complete autonomous system.