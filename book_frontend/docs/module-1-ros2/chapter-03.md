---
sidebar_position: 3
---

# Chapter 03: Real-time Communication Patterns

## Ensuring Timely and Reliable Robot Control

In this chapter, we'll explore the real-time communication patterns essential for humanoid robot control. Unlike general-purpose applications, robots require predictable timing and reliability to maintain stability and safety. We'll cover Quality of Service (QoS) profiles, real-time considerations, and advanced communication patterns.

### Quality of Service (QoS) Profiles

QoS profiles allow you to specify how messages are delivered in terms of reliability, durability, and ordering. This is crucial for humanoid robots where some data streams require guaranteed delivery while others can tolerate some loss.

#### Reliability Policy

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy

# For critical data like emergency stop commands
critical_qos = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE  # Guaranteed delivery
)

# For streaming data like camera feeds where some loss is acceptable
streaming_qos = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.BEST_EFFORT  # Best effort delivery
)
```

#### Durability Policy

```python
from rclpy.qos import QoSProfile, DurabilityPolicy

# For static data like robot description that should persist
static_qos = QoSProfile(
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL  # Keep last message for late joiners
)

# For regular streaming data
default_qos = QoSProfile(
    depth=10,
    durability=DurabilityPolicy.VOLATILE  # No persistence
)
```

### Creating Publishers with QoS

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

class RealTimePublisher(Node):
    def __init__(self):
        super().__init__('real_time_publisher')

        # Critical joint state publisher with reliable delivery
        self.joint_pub = self.create_publisher(
            JointState,
            'joint_states',
            QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE
            )
        )

        # Sensor data publisher with best-effort delivery
        self.sensor_pub = self.create_publisher(
            String,
            'sensor_data',
            QoSProfile(
                depth=5,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE
            )
        )

        # Timer for periodic publishing
        self.timer = self.create_timer(0.01, self.timer_callback)  # 100Hz
        self.i = 0

    def timer_callback(self):
        # Publish joint states with high reliability
        joint_msg = JointState()
        joint_msg.name = ['joint1', 'joint2', 'joint3']
        joint_msg.position = [1.0, 2.0, 3.0]
        self.joint_pub.publish(joint_msg)

        # Publish sensor data with best effort
        sensor_msg = String()
        sensor_msg.data = f'Sensor reading: {self.i}'
        self.sensor_pub.publish(sensor_msg)

        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    publisher = RealTimePublisher()
    rclpy.spin(publisher)
    publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Time-Synchronized Communication

For humanoid robots, it's often necessary to synchronize data from multiple sensors or coordinate actions across multiple subsystems.

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from std_msgs.msg import Header
from builtin_interfaces.msg import Time
import time

class SynchronizedNode(Node):
    def __init__(self):
        super().__init__('synchronized_node')

        self.joint_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, 'imu_data', self.imu_callback, 10)

        self.sync_publisher = self.create_publisher(
            String, 'synced_data', 10)

        # Store timestamps for synchronization
        self.joint_timestamp = None
        self.imu_timestamp = None
        self.joint_data = None
        self.imu_data = None

    def joint_callback(self, msg):
        self.joint_timestamp = self.get_clock().now()
        self.joint_data = msg
        self.check_synchronization()

    def imu_callback(self, msg):
        self.imu_timestamp = self.get_clock().now()
        self.imu_data = msg
        self.check_synchronization()

    def check_synchronization(self):
        # Check if we have both data sets within a time threshold
        if self.joint_data and self.imu_data:
            time_diff = abs((self.joint_timestamp - self.imu_timestamp).nanoseconds)
            # If timestamps are within 10ms, publish synced data
            if time_diff < 10000000:  # 10ms in nanoseconds
                self.publish_synced_data()
                # Reset for next sync cycle
                self.joint_data = None
                self.imu_data = None

    def publish_synced_data(self):
        # Combine and publish synchronized data
        combined_msg = String()
        combined_msg.data = f"Synced data at {self.joint_timestamp}"
        self.sync_publisher.publish(combined_msg)
```

### Action Servers for Long-Running Tasks

Actions are perfect for humanoid robot tasks that take time to complete and may provide feedback, such as walking to a location or grasping an object.

```python
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import time

class WalkingActionServer(Node):
    def __init__(self):
        super().__init__('walking_action_server')

        # Use a reentrant callback group to handle multiple goals
        callback_group = ReentrantCallbackGroup()

        self._action_server = ActionServer(
            self,
            FollowJointTrajectory,
            'walk_to_goal',
            execute_callback=self.execute_callback,
            callback_group=callback_group,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback)

    def destroy(self):
        self._action_server.destroy()
        super().destroy_node()

    def goal_callback(self, goal_request):
        # Accept all goals
        self.get_logger().info('Received goal request')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        # Accept all cancel requests
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')

        feedback_msg = FollowJointTrajectory.Feedback()
        feedback_msg.joint_names = ['left_leg', 'right_leg', 'left_arm', 'right_arm']

        # Simulate walking by sending trajectory points
        for i in range(0, 100):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return FollowJointTrajectory.Result()

            # Update feedback
            feedback_msg.desired.positions = [i * 0.01] * 4
            goal_handle.publish_feedback(feedback_msg)

            # Sleep to simulate actual walking
            time.sleep(0.1)

        # Complete successfully
        goal_handle.succeed()
        result = FollowJointTrajectory.Result()
        result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
        self.get_logger().info('Goal succeeded')
        return result

def main(args=None):
    rclpy.init(args=args)

    action_server = WalkingActionServer()

    executor = MultiThreadedExecutor()
    executor.add_node(action_server)

    try:
        executor.spin()
    finally:
        action_server.destroy()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Real-time Considerations for Humanoid Robots

Humanoid robots have strict timing requirements for stability and safety. Here are key considerations:

#### Control Loop Timing

```python
class BalanceController(Node):
    def __init__(self):
        super().__init__('balance_controller')

        # High-frequency control loop for balance (typically 200-1000 Hz)
        self.balance_timer = self.create_timer(
            0.005,  # 200 Hz (5ms period)
            self.balance_callback,
            clock=Self.get_clock()
        )

        # Lower frequency for planning (typically 10-50 Hz)
        self.planning_timer = self.create_timer(
            0.05,  # 20 Hz (50ms period)
            self.planning_callback
        )

    def balance_callback(self):
        # Critical high-frequency balance control
        # This must execute within the timing budget
        pass

    def planning_callback(self):
        # Less critical planning tasks
        pass
```

#### Priority and CPU Affinity

For real-time performance, you might need to set thread priorities:

```python
import threading
import os

def set_realtime_priority():
    # Set high priority for critical threads
    try:
        # Linux-specific: set SCHED_FIFO scheduling
        import ctypes
        from ctypes import c_int, c_short, c_char_p

        # This requires appropriate system permissions
        # Usually requires running as root or with CAP_SYS_NICE capability
        pass
    except ImportError:
        pass
```

### Performance Monitoring

Monitor communication performance to ensure real-time requirements are met:

```python
class PerformanceMonitor(Node):
    def __init__(self):
        super().__init__('performance_monitor')

        self.joint_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_callback, 10)

        self.message_times = []
        self.timer = self.create_timer(1.0, self.report_performance)

    def joint_callback(self, msg):
        receive_time = self.get_clock().now()
        # Calculate latency if message has timestamp
        if msg.header.stamp.sec != 0 or msg.header.stamp.nanosec != 0:
            msg_time = Time(
                sec=msg.header.stamp.sec,
                nanosec=msg.header.stamp.nanosec
            )
            latency = (receive_time - msg_time).nanoseconds / 1_000_000  # in ms
            self.message_times.append(latency)

    def report_performance(self):
        if self.message_times:
            avg_latency = sum(self.message_times) / len(self.message_times)
            max_latency = max(self.message_times)
            min_latency = min(self.message_times)

            self.get_logger().info(
                f'Performance: avg={avg_latency:.2f}ms, '
                f'max={max_latency:.2f}ms, min={min_latency:.2f}ms'
            )

            # Reset for next interval
            self.message_times = []
```

### Summary

In this chapter, we've covered:
- Quality of Service (QoS) profiles for different communication requirements
- Time-synchronized communication patterns
- Action servers for long-running robot tasks
- Real-time considerations for humanoid robot control
- Performance monitoring techniques

These patterns are essential for creating stable and responsive humanoid robots. In the next chapter, we'll explore how to bridge AI logic with ROS 2 for intelligent robot behavior.