---
sidebar_position: 2
---

# Chapter 02: Nodes, Topics, and Services

## Building Communication Between Robot Components

In the previous chapter, we introduced the fundamental concepts of ROS 2 and created our first simple node. Now we'll dive deeper into the core communication patterns that make ROS 2 so powerful for humanoid robotics: nodes, topics, and services.

### Understanding Nodes

A node is a process that performs computation. In ROS 2, nodes are written in one of the supported languages (C++, Python, etc.) and communicate with other nodes through topics, services, actions, and parameters.

#### Node Structure

Every ROS 2 node follows a similar structure:

```python
import rclpy
from rclpy.node import Node

class MyRobotNode(Node):
    def __init__(self):
        # Initialize the node with a name
        super().__init__('node_name')

        # Create publishers, subscribers, services, etc.
        # Perform initial setup

    # Define callbacks and other methods

def main(args=None):
    rclpy.init(args=args)
    node = MyRobotNode()
    rclpy.spin(node)  # Keep the node running
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Topics and Publishers/Subscribers

Topics enable asynchronous, many-to-many communication using a publish/subscribe pattern. This is ideal for streaming data like sensor readings or joint states.

#### Creating a Publisher

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class JointStatePublisher(Node):
    def __init__(self):
        super().__init__('joint_state_publisher')
        self.publisher_ = self.create_publisher(String, 'joint_states', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Joint positions: {self.i}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    joint_publisher = JointStatePublisher()
    rclpy.spin(joint_publisher)
    joint_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### Creating a Subscriber

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class JointStateSubscriber(Node):
    def __init__(self):
        super().__init__('joint_state_subscriber')
        self.subscription = self.create_subscription(
            String,
            'joint_states',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')

def main(args=None):
    rclpy.init(args=args)
    joint_subscriber = JointStateSubscriber()
    rclpy.spin(joint_subscriber)
    joint_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Services

Services provide synchronous request/response communication, which is useful for operations that need to return a result, like setting parameters or executing specific commands.

#### Creating a Service Server

```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class MinimalService(Node):
    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Incoming request\na: {request.a}, b: {request.b}')
        return response

def main(args=None):
    rclpy.init(args=args)
    minimal_service = MinimalService()
    rclpy.spin(minimal_service)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### Creating a Service Client

```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class MinimalClientAsync(Node):
    def __init__(self):
        super().__init__('minimal_client_async')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

def main(args=None):
    rclpy.init(args=args)
    minimal_client = MinimalClientAsync()
    response = minimal_client.send_request(1, 2)
    minimal_client.get_logger().info(f'Result of add_two_ints: {response.sum}')
    minimal_client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Humanoid Robot Example

Let's create a practical example for a humanoid robot that demonstrates these concepts:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from example_interfaces.srv import SetBool

class HumanoidController(Node):
    def __init__(self):
        super().__init__('humanoid_controller')

        # Publishers
        self.joint_pub = self.create_publisher(JointState, 'joint_commands', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # Subscribers
        self.joint_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, 10)
        self.imu_sub = self.create_subscription(
            JointState, 'imu_data', self.imu_callback, 10)

        # Services
        self.balance_srv = self.create_service(
            SetBool, 'enable_balance', self.enable_balance_callback)

        self.get_logger().info('Humanoid Controller initialized')

    def joint_state_callback(self, msg):
        # Process joint state data
        self.get_logger().info(f'Received {len(msg.position)} joint positions')

    def imu_callback(self, msg):
        # Process IMU data for balance
        pass

    def enable_balance_callback(self, request, response):
        if request.data:
            self.get_logger().info('Balance control enabled')
        else:
            self.get_logger().info('Balance control disabled')
        response.success = True
        response.message = 'Balance control updated'
        return response

def main(args=None):
    rclpy.init(args=args)
    controller = HumanoidController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Best Practices for Humanoid Robotics

1. **Modular Design**: Keep each node focused on a single responsibility
2. **Consistent Naming**: Use descriptive names for topics and services
3. **Message Types**: Use appropriate message types from standard ROS 2 packages
4. **Error Handling**: Implement proper error handling and logging
5. **Resource Management**: Properly clean up resources when nodes are destroyed

### Summary

In this chapter, we've explored:
- The structure and creation of ROS 2 nodes
- Implementing publishers and subscribers for asynchronous communication
- Creating services for synchronous request/response patterns
- A practical example of a humanoid robot controller
- Best practices for humanoid robotics applications

In the next chapter, we'll continue exploring more advanced communication patterns and real-time considerations for humanoid robots.