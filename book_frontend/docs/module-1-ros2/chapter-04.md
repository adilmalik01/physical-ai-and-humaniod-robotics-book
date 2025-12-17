---
sidebar_position: 4
---

# Chapter 04: Bridging AI Logic with ROS 2

## Connecting Intelligent Systems to Robot Control

In this chapter, we'll explore how to bridge AI logic with ROS 2, creating the connection between high-level reasoning and low-level robot control. This integration is fundamental to Physical AI, where cognitive systems drive physical actions in real environments.

### AI-ROS Integration Architecture

The integration between AI systems and ROS 2 typically follows a layered architecture:

```
┌─────────────────┐
│   AI Layer      │ ← LLMs, perception models, planning
├─────────────────┤
│  Bridge Layer   │ ← ROS 2 interfaces, message conversion
├─────────────────┤
│ Control Layer   │ ← Joint control, sensor processing
└─────────────────┘
```

### Integrating Python AI Libraries with ROS 2

Most AI libraries in Python can be integrated with ROS 2 by running them within ROS 2 nodes:

```python
import rclpy
from rclpy.node import Node
import numpy as np
import tensorflow as tf  # Example AI library

from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Pose

class AIPerceptionNode(Node):
    def __init__(self):
        super().__init__('ai_perception_node')

        # Load AI model
        self.model = self.load_model()

        # Subscribe to sensor data
        self.image_sub = self.create_subscription(
            Image, 'camera/image_raw', self.image_callback, 10)

        # Publish AI results
        self.detection_pub = self.create_publisher(
            String, 'object_detections', 10)

    def load_model(self):
        # Load your AI model here
        # This could be a TensorFlow model, PyTorch model, etc.
        self.get_logger().info('Loading AI perception model...')
        # model = tf.keras.models.load_model('path/to/model')
        # return model
        return None  # Placeholder

    def image_callback(self, msg):
        # Convert ROS image to format expected by AI model
        image_data = self.ros_image_to_numpy(msg)

        # Run AI inference
        if self.model:
            results = self.run_inference(image_data)
        else:
            # Placeholder for demo
            results = {'objects': ['person', 'chair'], 'confidence': [0.9, 0.8]}

        # Publish results
        result_msg = String()
        result_msg.data = str(results)
        self.detection_pub.publish(result_msg)

    def ros_image_to_numpy(self, ros_image):
        # Convert ROS Image message to numpy array
        # Implementation depends on image encoding
        height = ros_image.height
        width = ros_image.width
        # ... conversion logic
        return np.zeros((height, width, 3))  # Placeholder

    def run_inference(self, image):
        # Run the AI model on the image
        # This is where the actual AI processing happens
        return {'objects': [], 'confidence': []}

def main(args=None):
    rclpy.init(args=args)
    ai_node = AIPerceptionNode()
    rclpy.spin(ai_node)
    ai_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### LLM Integration with ROS 2

Integrating Large Language Models (LLMs) with ROS 2 enables natural language interaction with robots:

```python
import rclpy
from rclpy.node import Node
import openai  # or other LLM API
import asyncio

from std_msgs.msg import String
from geometry_msgs.msg import Pose

class LLMCommandNode(Node):
    def __init__(self):
        super().__init__('llm_command_node')

        # Set up LLM client (requires API key)
        # openai.api_key = os.getenv("OPENAI_API_KEY")

        # Subscribe to voice commands
        self.command_sub = self.create_subscription(
            String, 'voice_command', self.command_callback, 10)

        # Publish parsed commands
        self.action_pub = self.create_publisher(
            String, 'parsed_action', 10)

        # Store robot context
        self.robot_context = {
            'location': {'x': 0.0, 'y': 0.0},
            'battery': 100.0,
            'capabilities': ['move', 'grasp', 'speak']
        }

    async def process_command_async(self, command_text):
        """Process command using LLM asynchronously"""
        try:
            # Prepare context for the LLM
            context = f"""
            You are controlling a humanoid robot. Current state:
            - Location: {self.robot_context['location']}
            - Battery: {self.robot_context['battery']}%
            - Capabilities: {self.robot_context['capabilities']}

            User command: {command_text}

            Respond with a specific action in JSON format:
            {{
                "action": "move_to|grasp|speak|etc",
                "parameters": {{"x": 1.0, "y": 2.0, "object": "cup", "text": "hello"}}
            }}
            """

            # Call LLM (using mock for this example)
            # response = await openai.ChatCompletion.acreate(
            #     model="gpt-3.5-turbo",
            #     messages=[{"role": "user", "content": context}],
            #     max_tokens=150
            # )
            # return response.choices[0].message.content

            # Mock response for demonstration
            return '{"action": "move_to", "parameters": {"x": 1.0, "y": 2.0}}'

        except Exception as e:
            self.get_logger().error(f'LLM processing error: {e}')
            return None

    def command_callback(self, msg):
        """Handle incoming voice command"""
        command_text = msg.data
        self.get_logger().info(f'Received command: {command_text}')

        # Process command in separate thread to avoid blocking
        future = asyncio.run(self.process_command_async(command_text))

        if future:
            # Publish the parsed action
            action_msg = String()
            action_msg.data = future
            self.action_pub.publish(action_msg)

def main(args=None):
    rclpy.init(args=args)
    llm_node = LLMCommandNode()

    # Note: In a real implementation, you'd need to handle async properly
    # This is a simplified example

    rclpy.spin(llm_node)
    llm_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Creating a Planning Node

Planning nodes bridge high-level goals with low-level robot actions:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose, Point
from nav_msgs.msg import Path
import numpy as np

class PlanningNode(Node):
    def __init__(self):
        super().__init__('planning_node')

        # Subscribe to high-level goals
        self.goal_sub = self.create_subscription(
            String, 'high_level_goal', self.goal_callback, 10)

        # Publish detailed plans
        self.path_pub = self.create_publisher(
            Path, 'navigation_path', 10)

        # Subscribe to map and localization
        self.map_sub = self.create_subscription(
            OccupancyGrid, 'map', self.map_callback, 10)
        self.pose_sub = self.create_subscription(
            Pose, 'robot_pose', self.pose_callback, 10)

        # Store current state
        self.current_pose = Pose()
        self.map_data = None

    def goal_callback(self, msg):
        """Process high-level goal and generate detailed plan"""
        goal_description = msg.data

        # Parse the goal description
        goal_pose = self.parse_goal(goal_description)

        if goal_pose:
            # Plan path from current position to goal
            path = self.plan_path(self.current_pose, goal_pose)

            if path:
                # Publish the path
                path_msg = self.create_path_message(path)
                self.path_pub.publish(path_msg)

    def parse_goal(self, goal_description):
        """Parse natural language goal into coordinates"""
        # This could use NLP or simple keyword matching
        if 'kitchen' in goal_description.lower():
            return Pose(position=Point(x=5.0, y=3.0, z=0.0))
        elif 'living room' in goal_description.lower():
            return Pose(position=Point(x=2.0, y=1.0, z=0.0))
        elif 'bedroom' in goal_description.lower():
            return Pose(position=Point(x=8.0, y=6.0, z=0.0))
        else:
            # Try to extract coordinates from description
            # In practice, this would use more sophisticated NLP
            return None

    def plan_path(self, start_pose, goal_pose):
        """Plan a path from start to goal using A* or similar algorithm"""
        if self.map_data is None:
            self.get_logger().warn('No map available for path planning')
            return None

        # Simplified path planning (in practice, use nav2 or similar)
        path = []

        # Create a straight line path as an example
        steps = 10
        for i in range(steps + 1):
            t = i / steps
            x = start_pose.position.x + t * (goal_pose.position.x - start_pose.position.x)
            y = start_pose.position.y + t * (goal_pose.position.y - start_pose.position.y)

            pose = Pose()
            pose.position.x = x
            pose.position.y = y
            pose.position.z = 0.0
            path.append(pose)

        return path

    def create_path_message(self, poses):
        """Convert list of poses to Path message"""
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'
        path_msg.poses = poses
        return path_msg

    def map_callback(self, msg):
        """Store map data for path planning"""
        self.map_data = msg

    def pose_callback(self, msg):
        """Update current robot pose"""
        self.current_pose = msg

def main(args=None):
    rclpy.init(args=args)
    planner = PlanningNode()
    rclpy.spin(planner)
    planner.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Perception-Action Loop

Creating a tight perception-action loop is essential for responsive robots:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
import numpy as np

class PerceptionActionNode(Node):
    def __init__(self):
        super().__init__('perception_action_node')

        # Perception inputs
        self.image_sub = self.create_subscription(
            Image, 'camera/image_raw', self.image_callback, 10)
        self.laser_sub = self.create_subscription(
            LaserScan, 'scan', self.laser_callback, 10)

        # Action outputs
        self.cmd_vel_pub = self.create_publisher(
            Twist, 'cmd_vel', 10)

        # Control parameters
        self.avoid_distance = 0.5  # meters
        self.current_cmd = Twist()

        # Create a high-frequency timer for control
        self.control_timer = self.create_timer(
            0.05,  # 20 Hz control loop
            self.control_callback
        )

        self.laser_data = None
        self.image_data = None

    def image_callback(self, msg):
        """Process image data"""
        # Convert and store image data
        self.image_data = self.ros_image_to_numpy(msg)

        # Run simple perception (e.g., detect obstacles, people)
        perception_result = self.process_image(self.image_data)

        # Update behavior based on perception
        self.update_behavior_from_image(perception_result)

    def laser_callback(self, msg):
        """Process laser scan data"""
        self.laser_data = msg.ranges

    def process_image(self, image):
        """Run simple image processing"""
        # In practice, this could run object detection, etc.
        # For this example, we'll just detect if there are bright areas
        if image is not None:
            avg_brightness = np.mean(image)
            has_object = avg_brightness > 100  # arbitrary threshold
            return {'has_object': has_object, 'brightness': avg_brightness}
        return {'has_object': False, 'brightness': 0}

    def update_behavior_from_image(self, perception_result):
        """Update robot behavior based on image perception"""
        if perception_result['has_object']:
            # Maybe the robot should pay attention to the object
            self.get_logger().info('Detected object in camera view')

    def control_callback(self):
        """Main control loop"""
        if self.laser_data:
            # Process laser data for obstacle avoidance
            self.avoid_obstacles()

        # Publish current command
        self.cmd_vel_pub.publish(self.current_cmd)

    def avoid_obstacles(self):
        """Simple obstacle avoidance based on laser data"""
        if not self.laser_data:
            return

        # Find the minimum distance in front of the robot
        front_ranges = self.laser_data[300:420]  # Assuming 720-degree scan, front is center
        if front_ranges:
            min_distance = min([r for r in front_ranges if not np.isnan(r) and r > 0], default=float('inf'))

            if min_distance < self.avoid_distance:
                # Obstacle detected, turn away
                self.current_cmd.linear.x = 0.0
                self.current_cmd.angular.z = 0.5  # Turn right
            else:
                # Clear path, move forward
                self.current_cmd.linear.x = 0.3
                self.current_cmd.angular.z = 0.0

def main(args=None):
    rclpy.init(args=args)
    perception_action = PerceptionActionNode()
    rclpy.spin(perception_action)
    perception_action.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Integration Best Practices

When bridging AI logic with ROS 2, consider these best practices:

1. **Separation of Concerns**: Keep AI processing separate from ROS communication
2. **Error Handling**: Implement robust error handling for AI model failures
3. **Resource Management**: AI models can be resource-intensive; monitor usage
4. **Timing**: AI inference can be slow; consider async processing
5. **Data Conversion**: Optimize conversion between ROS message types and AI input formats

### Summary

In this chapter, we've explored:
- The architecture for integrating AI systems with ROS 2
- How to incorporate Python AI libraries into ROS 2 nodes
- LLM integration for natural language interaction
- Planning nodes that bridge high-level goals with low-level actions
- Perception-action loops for responsive robot behavior
- Best practices for AI-ROS integration

These concepts form the foundation for creating intelligent humanoid robots that can perceive, reason, and act in physical environments. In the next chapter, we'll explore humanoid body modeling using URDF.