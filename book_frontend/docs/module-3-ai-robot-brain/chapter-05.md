---
sidebar_position: 5
---

# Chapter 05: Perception-to-Action Loop Implementation

## Creating the Complete AI-Robot Brain System

In this final chapter of Module 3, we'll implement the complete perception-to-action loop that forms the AI-Robot Brain for humanoid robots. This involves integrating all the perception, navigation, and control components into a cohesive system that can process sensor data and generate appropriate actions in real-time.

### Understanding the Perception-to-Action Loop

The perception-to-action loop is the core of the AI-Robot Brain:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Perception    │ -> │   Decision      │ -> │   Action        │
│   (Sensors,     │    │   (Planning,    │    │   (Control,     │
│   Processing)   │    │   Reasoning)    │    │   Execution)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        ^                                           |
        |                                           |
        └───────────────────────────────────────────┘
                          Feedback
```

### System Architecture

The complete AI-Robot Brain system architecture:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, Imu, JointState
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Bool
from visualization_msgs.msg import MarkerArray
from tf2_ros import TransformListener, Buffer
import numpy as np
import threading
import time

class HumanoidAIBrain(Node):
    """
    Complete AI-Robot Brain system implementing the perception-to-action loop
    """
    def __init__(self):
        super().__init__('humanoid_ai_brain')

        # Initialize components
        self.initialize_perception_system()
        self.initialize_decision_system()
        self.initialize_action_system()
        self.initialize_feedback_system()

        # State management
        self.robot_state = {
            'position': np.array([0.0, 0.0, 0.0]),
            'orientation': np.array([0.0, 0.0, 0.0, 1.0]),  # quaternion
            'velocity': np.array([0.0, 0.0, 0.0]),
            'balance': True,
            'battery_level': 100.0,
            'current_task': 'idle',
            'safety_status': 'normal'
        }

        # Task queue for action planning
        self.task_queue = []
        self.current_task = None

        # Threading for real-time processing
        self.perception_thread = None
        self.decision_thread = None
        self.action_thread = None

        # Performance monitoring
        self.loop_times = []
        self.frame_times = []

        self.get_logger().info('Humanoid AI Brain initialized')

    def initialize_perception_system(self):
        """Initialize perception system components"""
        # TF buffer and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Sensor subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.process_image, 10
        )
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.process_laser_scan, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.process_imu, 10
        )
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.process_joint_states, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.process_odometry, 10
        )

        # Perception result publishers
        self.detection_pub = self.create_publisher(
            String, '/perception/detections', 10
        )
        self.environment_pub = self.create_publisher(
            String, '/perception/environment', 10
        )

        self.get_logger().info('Perception system initialized')

    def initialize_decision_system(self):
        """Initialize decision making system"""
        # Task planning subscribers
        self.task_sub = self.create_subscription(
            String, '/task_queue', self.add_task, 10
        )
        self.goal_sub = self.create_subscription(
            PoseStamped, '/move_base_simple/goal', self.set_navigation_goal, 10
        )

        # Decision result publishers
        self.action_pub = self.create_publisher(
            String, '/decision/action', 10
        )
        self.nav_goal_pub = self.create_publisher(
            PoseStamped, '/goal_pose', 10
        )

        # Initialize planners
        self.navigation_planner = NavigationPlanner()
        self.manipulation_planner = ManipulationPlanner()
        self.behavior_planner = BehaviorPlanner()

        self.get_logger().info('Decision system initialized')

    def initialize_action_system(self):
        """Initialize action execution system"""
        # Action command publishers
        self.cmd_vel_pub = self.create_publisher(
            Twist, '/cmd_vel', 10
        )
        self.joint_cmd_pub = self.create_publisher(
            JointState, '/joint_commands', 10
        )
        self.emergency_stop_pub = self.create_publisher(
            Bool, '/emergency_stop', 10
        )

        # Action status subscribers
        self.action_status_sub = self.create_subscription(
            String, '/action/status', self.update_action_status, 10
        )

        self.get_logger().info('Action system initialized')

    def initialize_feedback_system(self):
        """Initialize feedback and monitoring system"""
        # Performance monitoring
        self.performance_pub = self.create_publisher(
            String, '/brain/performance', 10
        )

        # State publishers
        self.state_pub = self.create_publisher(
            String, '/brain/state', 10
        )

        # Visualization
        self.visualization_pub = self.create_publisher(
            MarkerArray, '/brain/visualization', 10
        )

        # Performance monitoring timer
        self.perf_timer = self.create_timer(1.0, self.publish_performance_metrics)

        self.get_logger().info('Feedback system initialized')

    def process_image(self, msg):
        """Process camera image for perception"""
        start_time = time.time()

        # Convert ROS image to OpenCV
        from cv_bridge import CvBridge
        cv_bridge = CvBridge()
        cv_image = cv_bridge.imgmsg_to_cv2(msg, "bgr8")

        # Run object detection (using Isaac ROS accelerated detection)
        detections = self.run_object_detection(cv_image)

        # Update environment model
        self.update_environment_model(detections, msg.header)

        # Publish detections
        detection_msg = String()
        detection_msg.data = str(detections)
        self.detection_pub.publish(detection_msg)

        # Record processing time
        self.frame_times.append(time.time() - start_time)

    def process_laser_scan(self, msg):
        """Process laser scan data"""
        # Process obstacles and free space
        obstacles = self.extract_obstacles(msg)
        free_space = self.extract_free_space(msg)

        # Update local map
        self.update_local_map(obstacles, free_space)

    def process_imu(self, msg):
        """Process IMU data for balance monitoring"""
        # Check balance status
        balance_ok = self.check_balance(msg)

        # Update robot state
        self.robot_state['balance'] = balance_ok

        if not balance_ok:
            self.trigger_safety_procedure('Balance loss detected')

    def process_joint_states(self, msg):
        """Process joint state data"""
        # Update joint positions and velocities
        for i, name in enumerate(msg.name):
            if name in self.robot_state:
                self.robot_state[name] = {
                    'position': msg.position[i],
                    'velocity': msg.velocity[i],
                    'effort': msg.effort[i]
                }

    def process_odometry(self, msg):
        """Process odometry data for localization"""
        # Update position and orientation
        self.robot_state['position'] = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])

        self.robot_state['orientation'] = np.array([
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ])

        self.robot_state['velocity'] = np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z
        ])

    def run_object_detection(self, image):
        """Run object detection using Isaac ROS (placeholder)"""
        # In practice, this would use Isaac ROS DNN Inference
        # For demonstration, returning mock detections
        return [
            {'class': 'person', 'confidence': 0.9, 'bbox': [100, 100, 200, 200]},
            {'class': 'chair', 'confidence': 0.8, 'bbox': [300, 150, 400, 250]}
        ]

    def update_environment_model(self, detections, header):
        """Update environment model based on detections"""
        environment = {
            'timestamp': header.stamp.sec + header.stamp.nanosec * 1e-9,
            'objects': detections,
            'robot_pos': self.robot_state['position'].tolist()
        }

        # Publish environment update
        env_msg = String()
        env_msg.data = str(environment)
        self.environment_pub.publish(env_msg)

    def add_task(self, msg):
        """Add task to queue"""
        task = {
            'type': msg.data,
            'priority': 1,  # Default priority
            'created': time.time()
        }
        self.task_queue.append(task)
        self.get_logger().info(f'Added task: {task["type"]}')

    def set_navigation_goal(self, msg):
        """Set navigation goal"""
        self.get_logger().info(f'Setting navigation goal: {msg.pose}')
        # Add navigation task to queue
        nav_task = {
            'type': 'navigate',
            'goal': msg.pose,
            'priority': 2,
            'created': time.time()
        }
        self.task_queue.insert(0, nav_task)  # High priority

    def update_action_status(self, msg):
        """Update action execution status"""
        self.get_logger().debug(f'Action status: {msg.data}')

    def perception_to_action_loop(self):
        """Main perception-to-action loop"""
        loop_start = time.time()

        # Process perception data (this happens continuously via callbacks)
        # Now make decisions based on current state

        # Check for tasks to execute
        if self.task_queue and not self.current_task:
            self.current_task = self.task_queue.pop(0)
            self.execute_task(self.current_task)

        # Make decisions based on environment
        action = self.decision_making_process()

        # Execute action
        if action:
            self.execute_action(action)

        # Monitor performance
        loop_time = time.time() - loop_start
        self.loop_times.append(loop_time)

        # Ensure real-time performance (target 30 Hz)
        target_loop_time = 1.0 / 30.0  # 30 Hz
        sleep_time = max(0, target_loop_time - loop_time)
        if sleep_time > 0:
            time.sleep(sleep_time)

    def decision_making_process(self):
        """Main decision making process"""
        # Analyze current state and environment
        situation = self.analyze_situation()

        # Select appropriate action based on situation
        if situation['type'] == 'navigation':
            return self.plan_navigation(situation)
        elif situation['type'] == 'interaction':
            return self.plan_interaction(situation)
        elif situation['type'] == 'manipulation':
            return self.plan_manipulation(situation)
        elif situation['type'] == 'safety':
            return self.plan_safety_action(situation)
        else:
            return self.plan_default_action(situation)

    def analyze_situation(self):
        """Analyze current situation based on sensor data"""
        situation = {
            'type': 'idle',
            'objects': [],
            'obstacles': [],
            'navigation_needed': False,
            'interaction_opportunity': False,
            'safety_concerns': []
        }

        # Check for safety concerns first
        if not self.robot_state['balance']:
            situation['type'] = 'safety'
            situation['safety_concerns'].append('balance_loss')
            return situation

        # Check for objects requiring interaction
        if hasattr(self, 'environment_data'):
            for obj in self.environment_data.get('objects', []):
                if obj['class'] == 'person' and obj['confidence'] > 0.7:
                    situation['type'] = 'interaction'
                    situation['objects'].append(obj)
                    situation['interaction_opportunity'] = True

        # Check for navigation needs
        if self.task_queue and any(task['type'] == 'navigate' for task in self.task_queue):
            situation['type'] = 'navigation'
            situation['navigation_needed'] = True

        return situation

    def plan_navigation(self, situation):
        """Plan navigation action"""
        if self.task_queue:
            nav_task = next((t for t in self.task_queue if t['type'] == 'navigate'), None)
            if nav_task:
                goal = nav_task['goal']
                path = self.navigation_planner.plan_path(
                    self.robot_state['position'],
                    [goal.position.x, goal.position.y]
                )

                return {
                    'type': 'navigate',
                    'path': path,
                    'goal': [goal.position.x, goal.position.y]
                }

    def plan_interaction(self, situation):
        """Plan interaction action"""
        # Find the closest person to interact with
        closest_person = min(
            [obj for obj in situation['objects'] if obj['class'] == 'person'],
            key=lambda x: self.calculate_distance_to_object(x),
            default=None
        )

        if closest_person:
            return {
                'type': 'approach',
                'target': closest_person,
                'action': 'greet'
            }

    def plan_manipulation(self, situation):
        """Plan manipulation action"""
        # Placeholder for manipulation planning
        return None

    def plan_safety_action(self, situation):
        """Plan safety action"""
        return {
            'type': 'stop',
            'reason': 'safety',
            'emergency': True
        }

    def plan_default_action(self, situation):
        """Plan default action when no specific task"""
        return {
            'type': 'idle',
            'behavior': 'monitor'
        }

    def execute_action(self, action):
        """Execute the planned action"""
        if action['type'] == 'navigate':
            self.execute_navigation(action)
        elif action['type'] == 'approach':
            self.execute_approach(action)
        elif action['type'] == 'stop':
            self.execute_stop(action)
        elif action['type'] == 'idle':
            self.execute_idle(action)

    def execute_navigation(self, action):
        """Execute navigation action"""
        # Send navigation command
        twist = Twist()
        # Calculate appropriate velocity based on path
        if action['path']:
            # Simple proportional controller
            target_x, target_y = action['goal']
            current_x, current_y = self.robot_state['position'][:2]

            dx = target_x - current_x
            dy = target_y - current_y
            distance = np.sqrt(dx*dx + dy*dy)

            if distance > 0.1:  # If not close to goal
                twist.linear.x = min(0.3, distance * 0.5)  # Proportional speed
                twist.angular.z = np.arctan2(dy, dx) * 0.5  # Proportional turn

        self.cmd_vel_pub.publish(twist)

    def execute_approach(self, action):
        """Execute approach action"""
        target = action['target']
        # Move towards the target object
        self.get_logger().info(f'Approaching {target["class"]}')

    def execute_stop(self, action):
        """Execute stop action"""
        twist = Twist()
        self.cmd_vel_pub.publish(twist)

        if action.get('emergency'):
            # Trigger emergency procedures
            emergency_msg = Bool()
            emergency_msg.data = True
            self.emergency_stop_pub.publish(emergency_msg)

    def execute_idle(self, action):
        """Execute idle action"""
        # Continue monitoring environment
        pass

    def execute_task(self, task):
        """Execute a specific task from the queue"""
        self.get_logger().info(f'Executing task: {task["type"]}')
        self.current_task = task

    def trigger_safety_procedure(self, reason):
        """Trigger safety procedure"""
        self.get_logger().error(f'Safety procedure triggered: {reason}')

        # Stop all motion
        self.execute_stop({'type': 'stop', 'reason': reason, 'emergency': True})

        # Update safety status
        self.robot_state['safety_status'] = 'emergency_stop'

    def publish_performance_metrics(self):
        """Publish performance metrics"""
        if self.loop_times:
            avg_loop_time = sum(self.loop_times) / len(self.loop_times)
            max_loop_time = max(self.loop_times)
            min_loop_time = min(self.loop_times)

            # Keep only recent measurements
            self.loop_times = self.loop_times[-100:]

            perf_msg = String()
            perf_msg.data = f"Loop times - Avg: {avg_loop_time:.3f}s, Max: {max_loop_time:.3f}s, Min: {min_loop_time:.3f}s, Rate: {1.0/avg_loop_time:.1f}Hz"

            self.performance_pub.publish(perf_msg)

    def run(self):
        """Run the main AI brain loop"""
        # Start perception-to-action loop
        while rclpy.ok():
            self.perception_to_action_loop()

def main(args=None):
    rclpy.init(args=args)

    # Create the AI brain node
    ai_brain = HumanoidAIBrain()

    # Run the main loop in a separate thread to allow ROS2 callbacks
    brain_thread = threading.Thread(target=ai_brain.run)
    brain_thread.daemon = True
    brain_thread.start()

    try:
        rclpy.spin(ai_brain)
    except KeyboardInterrupt:
        pass
    finally:
        ai_brain.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Navigation Planner Component

```python
import numpy as np
from scipy.spatial import distance
import heapq

class NavigationPlanner:
    """
    Navigation planner for humanoid robots
    """
    def __init__(self):
        self.map_resolution = 0.05  # meters per cell
        self.robot_radius = 0.3     # meters
        self.step_height = 0.15     # maximum step height

    def plan_path(self, start_pos, goal_pos, costmap=None):
        """
        Plan path using A* algorithm with humanoid constraints
        """
        if costmap is None:
            # Create a simple grid for demonstration
            costmap = np.zeros((100, 100))  # 5m x 5m map

        # Convert world coordinates to grid coordinates
        start_grid = self.world_to_grid(start_pos[:2])
        goal_grid = self.world_to_grid(goal_pos)

        # Run A* path planning
        path = self.a_star_search(start_grid, goal_grid, costmap)

        # Convert grid path back to world coordinates
        world_path = [self.grid_to_world(grid_pos) for grid_pos in path]

        return world_path

    def a_star_search(self, start, goal, costmap):
        """
        A* path planning algorithm
        """
        # Define possible movements (8-connected grid)
        movements = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]

        # Heuristic function (Euclidean distance)
        def heuristic(pos):
            return distance.euclidean(pos, goal)

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
                if (0 <= neighbor[0] < costmap.shape[0] and
                    0 <= neighbor[1] < costmap.shape[1]):

                    # Check if walkable (cost < 255 means not an obstacle)
                    if costmap[neighbor[0], neighbor[1]] < 255:
                        tentative_g_score = g_score[current] + distance.euclidean(current, neighbor)

                        if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                            came_from[neighbor] = current
                            g_score[neighbor] = tentative_g_score
                            f_score[neighbor] = tentative_g_score + heuristic(neighbor)
                            heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return []  # No path found

    def world_to_grid(self, world_pos):
        """Convert world coordinates to grid coordinates"""
        grid_x = int(world_pos[0] / self.map_resolution)
        grid_y = int(world_pos[1] / self.map_resolution)
        return (grid_x, grid_y)

    def grid_to_world(self, grid_pos):
        """Convert grid coordinates to world coordinates"""
        world_x = grid_pos[0] * self.map_resolution
        world_y = grid_pos[1] * self.map_resolution
        return [world_x, world_y]
```

### Manipulation Planner Component

```python
import numpy as np

class ManipulationPlanner:
    """
    Manipulation planner for humanoid robot arms
    """
    def __init__(self):
        self.arm_joints = ['shoulder', 'elbow', 'wrist']
        self.reach_distance = 1.0  # meters

    def plan_reach(self, target_pos, current_pos):
        """
        Plan reaching motion to target position
        """
        # Calculate required joint angles using inverse kinematics
        # This is a simplified example - in practice, use full IK solver
        delta = np.array(target_pos) - np.array(current_pos)
        distance = np.linalg.norm(delta)

        if distance > self.reach_distance:
            # Scale to reach distance
            direction = delta / distance
            target_pos = current_pos + direction * self.reach_distance

        # Plan joint trajectory
        joint_trajectory = self.calculate_joint_trajectory(current_pos, target_pos)

        return joint_trajectory

    def calculate_joint_trajectory(self, start_pos, end_pos):
        """
        Calculate smooth joint trajectory between positions
        """
        # Simplified trajectory calculation
        # In practice, use proper trajectory generation
        trajectory = []

        # Linear interpolation in Cartesian space
        steps = 10
        for i in range(steps + 1):
            t = i / steps
            pos = start_pos + t * (np.array(end_pos) - np.array(start_pos))
            trajectory.append(pos.tolist())

        return trajectory
```

### Behavior Planner Component

```python
class BehaviorPlanner:
    """
    Behavior planner for high-level decision making
    """
    def __init__(self):
        self.behaviors = {
            'explore': self.explore_behavior,
            'follow': self.follow_behavior,
            'avoid': self.avoid_behavior,
            'greet': self.greet_behavior,
            'idle': self.idle_behavior
        }

    def select_behavior(self, environment_state):
        """
        Select appropriate behavior based on environment state
        """
        # Analyze environment state
        if self.detect_human_nearby(environment_state):
            return 'greet'
        elif self.detect_path_obstacle(environment_state):
            return 'avoid'
        elif self.detect_follow_target(environment_state):
            return 'follow'
        elif self.should_explore(environment_state):
            return 'explore'
        else:
            return 'idle'

    def detect_human_nearby(self, env_state):
        """Check if human is nearby"""
        if 'objects' in env_state:
            for obj in env_state['objects']:
                if obj.get('class') == 'person' and obj.get('confidence', 0) > 0.7:
                    return True
        return False

    def detect_path_obstacle(self, env_state):
        """Check if path is obstructed"""
        # Check for obstacles in planned path
        return False  # Placeholder

    def detect_follow_target(self, env_state):
        """Check if there's a target to follow"""
        return False  # Placeholder

    def should_explore(self, env_state):
        """Determine if robot should explore"""
        return True  # Placeholder

    def greet_behavior(self, env_state):
        """Greet behavior implementation"""
        return {
            'action': 'approach_person',
            'priority': 1
        }

    def avoid_behavior(self, env_state):
        """Avoid behavior implementation"""
        return {
            'action': 'move_around_obstacle',
            'priority': 2
        }

    def follow_behavior(self, env_state):
        """Follow behavior implementation"""
        return {
            'action': 'follow_target',
            'priority': 1
        }

    def explore_behavior(self, env_state):
        """Explore behavior implementation"""
        return {
            'action': 'explore_area',
            'priority': 0
        }

    def idle_behavior(self, env_state):
        """Idle behavior implementation"""
        return {
            'action': 'monitor_environment',
            'priority': 0
        }
```

### Performance Optimization

To optimize the perception-to-action loop:

```python
class OptimizedAIBrain(HumanoidAIBrain):
    """
    Optimized version of the AI brain with performance enhancements
    """
    def __init__(self):
        super().__init__()

        # Multi-threading for different processing stages
        self.perception_queue = queue.Queue(maxsize=2)
        self.decision_queue = queue.Queue(maxsize=2)
        self.action_queue = queue.Queue(maxsize=2)

        # Process pools for CPU-intensive tasks
        self.process_pool = ProcessPoolExecutor(max_workers=4)

        # Memory pools for reducing allocation overhead
        self.message_pool = []

        # Adaptive rate control
        self.target_rate = 30.0  # Hz
        self.current_rate = 30.0
        self.rate_history = []

    def process_image_optimized(self, msg):
        """
        Optimized image processing with multi-threading
        """
        # Skip frames if processing is behind
        if self.perception_queue.full():
            return  # Skip this frame

        # Process in separate thread
        future = self.process_pool.submit(self.run_object_detection, msg)

        # Add to processing queue
        self.perception_queue.put((msg.header.stamp, future))

    def adaptive_rate_control(self):
        """
        Adjust processing rate based on system load
        """
        # Monitor processing times
        if len(self.rate_history) > 10:
            avg_time = sum(self.rate_history[-10:]) / 10

            # Adjust target rate based on processing time
            if avg_time > 1.0 / self.target_rate * 0.8:  # 80% of target time
                self.target_rate = max(10.0, self.target_rate * 0.9)  # Reduce rate
            elif avg_time < 1.0 / self.target_rate * 0.5:  # 50% of target time
                self.target_rate = min(60.0, self.target_rate * 1.1)  # Increase rate

            self.rate_history = self.rate_history[-10:]  # Keep recent history
```

### Safety and Fault Tolerance

Implement safety measures for the perception-to-action loop:

```python
class SafeAIBrain(HumanoidAIBrain):
    """
    AI brain with enhanced safety and fault tolerance
    """
    def __init__(self):
        super().__init__()

        # Safety state machine
        self.safety_state = 'NORMAL'
        self.safety_transitions = {
            'NORMAL': ['WARNING', 'EMERGENCY'],
            'WARNING': ['NORMAL', 'EMERGENCY'],
            'EMERGENCY': ['NORMAL']
        }

        # Watchdog timers
        self.watchdogs = {
            'perception': 2.0,  # 2 seconds timeout
            'decision': 1.0,    # 1 second timeout
            'action': 0.5       # 0.5 seconds timeout
        }

        # Safety callbacks
        self.safety_callbacks = []

        # Initialize safety system
        self.initialize_safety_system()

    def initialize_safety_system(self):
        """Initialize safety monitoring system"""
        # Start watchdog timers
        for name, timeout in self.watchdogs.items():
            timer = self.create_timer(
                timeout,
                lambda n=name: self.check_watchdog(n)
            )
            self.watchdogs[name + '_timer'] = timer

    def check_watchdog(self, name):
        """Check if a component is responsive"""
        if not self.is_component_responsive(name):
            self.trigger_safety_action(name)

    def is_component_responsive(self, name):
        """Check if component is responding"""
        # Implementation to check component responsiveness
        return True  # Placeholder

    def trigger_safety_action(self, component_name):
        """Trigger safety action for unresponsive component"""
        self.get_logger().error(f'Safety: {component_name} not responding')

        # Transition to appropriate safety state
        if self.safety_state == 'NORMAL':
            self.safety_state = 'WARNING'
            self.slow_down_robot()
        elif self.safety_state == 'WARNING':
            self.safety_state = 'EMERGENCY'
            self.emergency_stop()

        # Notify other components
        for callback in self.safety_callbacks:
            callback(component_name, self.safety_state)

    def slow_down_robot(self):
        """Reduce robot speed for safety"""
        twist = Twist()
        # Reduce current velocity by 50%
        self.cmd_vel_pub.publish(twist)

    def emergency_stop(self):
        """Emergency stop the robot"""
        twist = Twist()
        self.cmd_vel_pub.publish(twist)

        # Trigger emergency stop
        emergency_msg = Bool()
        emergency_msg.data = True
        self.emergency_stop_pub.publish(emergency_msg)

    def add_safety_callback(self, callback):
        """Add safety callback function"""
        self.safety_callbacks.append(callback)
```

### Testing the Complete System

Create a comprehensive test for the perception-to-action loop:

```python
import unittest
from rclpy.time import Time
from geometry_msgs.msg import Pose, Twist

class TestPerceptionToActionLoop(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.ai_brain = HumanoidAIBrain()

    def test_basic_perception_loop(self):
        """Test basic perception functionality"""
        # Simulate sensor data
        test_image = self.create_test_image()
        test_scan = self.create_test_scan()

        # Process perception
        self.ai_brain.process_image(test_image)
        self.ai_brain.process_laser_scan(test_scan)

        # Verify perception results
        self.assertIsNotNone(self.ai_brain.environment_data)

    def test_decision_making(self):
        """Test decision making process"""
        # Set up test environment state
        self.ai_brain.environment_data = {
            'objects': [{'class': 'person', 'confidence': 0.9}]
        }

        # Analyze situation
        situation = self.ai_brain.analyze_situation()

        # Verify situation analysis
        self.assertEqual(situation['type'], 'interaction')
        self.assertTrue(situation['interaction_opportunity'])

    def test_action_execution(self):
        """Test action execution"""
        # Plan an action
        action = {
            'type': 'navigate',
            'path': [[1.0, 0.0], [2.0, 0.0]],
            'goal': [2.0, 0.0]
        }

        # Execute action
        self.ai_brain.execute_action(action)

        # Verify command was published
        # This would require mocking the publisher to verify

    def test_safety_procedures(self):
        """Test safety procedures"""
        # Trigger safety condition
        self.ai_brain.robot_state['balance'] = False

        # Check if safety procedure was triggered
        # This would check if emergency stop was published

    def create_test_image(self):
        """Create test image message"""
        # Implementation for creating test image
        pass

    def create_test_scan(self):
        """Create test laser scan message"""
        # Implementation for creating test scan
        pass

def run_comprehensive_tests():
    """Run all perception-to-action loop tests"""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPerceptionToActionLoop)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()
```

### Integration with Isaac Tools

Complete integration with Isaac tools for enhanced capabilities:

```python
class IsaacIntegratedAIBrain(SafeAIBrain):
    """
    AI brain with full Isaac tool integration
    """
    def __init__(self):
        super().__init__()

        # Isaac ROS integration
        self.isaac_perception = IsaacPerceptionInterface()
        self.isaac_navigation = IsaacNavigationInterface()
        self.isaac_simulation = IsaacSimulationInterface()

        # Initialize Isaac components
        self.initialize_isaac_components()

    def initialize_isaac_components(self):
        """Initialize Isaac-specific components"""
        # Initialize Isaac perception
        self.isaac_perception.initialize()

        # Initialize Isaac navigation
        self.isaac_navigation.initialize()

        # Initialize Isaac simulation interface
        self.isaac_simulation.initialize()

    def process_image_isaac(self, msg):
        """Process image using Isaac ROS acceleration"""
        # Use Isaac ROS for accelerated perception
        detections = self.isaac_perception.run_detection(msg)

        # Update environment model
        self.update_environment_model(detections, msg.header)

    def plan_navigation_isaac(self, goal):
        """Plan navigation using Isaac navigation tools"""
        # Use Isaac navigation for path planning
        path = self.isaac_navigation.plan_path(
            self.robot_state['position'][:2],
            [goal.position.x, goal.position.y]
        )

        return path

    def simulate_behavior_isaac(self, action):
        """Simulate behavior in Isaac Sim before execution"""
        # Simulate the action in Isaac Sim first
        success = self.isaac_simulation.test_action(action)

        if success:
            # Execute in real world
            self.execute_action(action)
        else:
            # Plan alternative action
            self.get_logger().warn('Action simulation failed, planning alternative')
```

### Summary

In this chapter, we've covered:
- The complete perception-to-action loop architecture
- Implementation of the AI-Robot Brain system
- Integration of navigation, manipulation, and behavior planning
- Performance optimization techniques
- Safety and fault tolerance mechanisms
- Testing and validation approaches
- Full integration with Isaac tools

This completes Module 3: The AI–Robot Brain (NVIDIA Isaac). You now have a comprehensive understanding of how to build intelligent perception and navigation systems for humanoid robots using NVIDIA Isaac tools.

In Module 4, we'll explore Vision-Language-Action integration, bringing together speech recognition, LLM reasoning, and vision-guided manipulation for complete humanoid robot intelligence.