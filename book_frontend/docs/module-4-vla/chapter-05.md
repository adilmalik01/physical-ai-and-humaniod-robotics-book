---
sidebar_position: 5
---

# Chapter 05: End-to-End System Integration

## Bringing Together Vision, Language, and Action

In this final chapter of Module 4, we'll integrate all the components we've developed throughout the module into a complete end-to-end system. This chapter focuses on connecting speech recognition, LLM-based task planning, and vision-guided manipulation into a unified humanoid robot system that can understand natural language commands and execute complex tasks.

### System Architecture Overview

The complete Vision-Language-Action (VLA) system architecture:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Speech       │ -> │  Task Planning  │ -> │  Action         │
│   Recognition  │    │  (LLM)          │    │  Execution      │
│   (Whisper)    │    │  (GPT-4)        │    │  (Navigation,   │
└─────────────────┘    └─────────────────┘    │  Manipulation)   │
        │                       │               └─────────────────┘
        └───────────────────────┼─────────────────────────┘
                                │
                    ┌─────────────────┐
                    │  Perception     │
                    │  (Vision,       │
                    │  Localization)   │
                    └─────────────────┘
```

### Complete VLA System Implementation

Let's implement the complete integrated system:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from action_msgs.msg import GoalStatus
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
import threading
import asyncio
import json
import time
from typing import Dict, Any, List

class IntegratedVLASystem(Node):
    """
    Complete integrated Vision-Language-Action system
    """
    def __init__(self):
        super().__init__('integrated_vla_system')

        # Initialize all system components
        self.initialize_components()

        # Set up ROS 2 communication
        self.setup_communication_interfaces()

        # Initialize system state
        self.system_state = {
            'current_task': None,
            'task_status': 'idle',
            'robot_position': [0, 0, 0],
            'robot_orientation': [0, 0, 0, 1],
            'detected_objects': [],
            'conversation_history': []
        }

        # Task execution queue
        self.task_queue = asyncio.Queue()

        # Start task processing loop
        self.task_processing_thread = threading.Thread(target=self.task_processing_loop)
        self.task_processing_thread.daemon = True
        self.task_processing_thread.start()

        self.get_logger().info('Integrated VLA System initialized')

    def initialize_components(self):
        """Initialize all system components"""
        # Initialize speech recognition (using mock for this example)
        self.speech_recognizer = MockSpeechRecognizer()

        # Initialize LLM task planner (using mock for this example)
        self.task_planner = MockLLMTaskPlanner()

        # Initialize vision system
        self.vision_system = MockVisionSystem()

        # Initialize manipulation controller
        self.manipulation_controller = MockManipulationController()

        # Initialize navigation system
        self.navigation_system = MockNavigationSystem()

    def setup_communication_interfaces(self):
        """Set up all ROS 2 communication interfaces"""
        # Publishers
        self.response_pub = self.create_publisher(String, '/vla/response', 10)
        self.status_pub = self.create_publisher(String, '/vla/status', 10)
        self.action_pub = self.create_publisher(String, '/vla/action', 10)
        self.speech_pub = self.create_publisher(String, '/speech_recognition/text', 10)

        # Subscribers
        self.speech_sub = self.create_subscription(
            String, '/speech_recognition/text', self.speech_callback, 10
        )
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10
        )
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )

        # Service servers
        self.process_command_srv = self.create_service(
            ProcessCommand, '/vla/process_command', self.process_command_callback
        )

        # Action servers
        self.execute_task_server = ActionServer(
            self,
            ExecuteTask,
            'execute_vla_task',
            self.execute_task_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=ReentrantCallbackGroup()
        )

    def speech_callback(self, msg):
        """Handle incoming speech recognition results"""
        command = msg.data
        self.get_logger().info(f'Received speech command: {command}')

        # Add to task queue for processing
        asyncio.run_coroutine_threadsafe(
            self.task_queue.put({'type': 'command', 'data': command}),
            asyncio.get_event_loop()
        )

    def image_callback(self, msg):
        """Handle incoming image data"""
        # Process image for object detection and scene understanding
        objects = self.vision_system.process_image(msg)
        self.system_state['detected_objects'] = objects

        # Update vision system with new data
        self.vision_system.update_scene(objects)

    def odom_callback(self, msg):
        """Handle odometry updates"""
        self.system_state['robot_position'] = [
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ]
        self.system_state['robot_orientation'] = [
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ]

    def joint_state_callback(self, msg):
        """Handle joint state updates"""
        # Update manipulation controller with current joint states
        self.manipulation_controller.update_joint_states(msg)

    def task_processing_loop(self):
        """Main task processing loop running in separate thread"""
        # Create asyncio event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def process_tasks():
            while rclpy.ok():
                try:
                    # Get next task from queue
                    task = await self.task_queue.get()

                    if task['type'] == 'command':
                        await self.process_command(task['data'])
                    elif task['type'] == 'task':
                        await self.execute_task(task['data'])

                except Exception as e:
                    self.get_logger().error(f'Error in task processing: {e}')

                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting

        loop.run_until_complete(process_tasks())

    async def process_command(self, command: str):
        """Process a natural language command end-to-end"""
        self.get_logger().info(f'Processing command: {command}')

        # Update status
        status_msg = String()
        status_msg.data = f'Processing: {command}'
        self.status_pub.publish(status_msg)

        try:
            # 1. Get current environment state
            environment_state = self.get_current_environment_state()

            # 2. Plan task using LLM
            task_plan = self.task_planner.plan_task(command, environment_state)

            if not task_plan or not task_plan.get('actions'):
                # No plan could be generated
                response = f"Sorry, I couldn't understand how to {command}"
                self.publish_response(response)
                return

            # 3. Execute the planned actions
            success = await self.execute_action_plan(task_plan)

            if success:
                response = f"I have completed the task: {command}"
            else:
                response = f"I couldn't complete the task: {command}"

            self.publish_response(response)

        except Exception as e:
            self.get_logger().error(f'Error processing command: {e}')
            error_response = f"Sorry, I encountered an error processing: {command}"
            self.publish_response(error_response)

    def get_current_environment_state(self) -> Dict[str, Any]:
        """Get current environment state for task planning"""
        return {
            'robot_position': self.system_state['robot_position'],
            'robot_orientation': self.system_state['robot_orientation'],
            'detected_objects': self.system_state['detected_objects'],
            'conversation_history': self.system_state['conversation_history'][-5:],  # Last 5 interactions
            'robot_capabilities': self.get_robot_capabilities()
        }

    def get_robot_capabilities(self) -> Dict[str, Any]:
        """Get robot capabilities for task planning"""
        return {
            'navigation': {
                'max_speed': 0.5,
                'step_height': 0.15,
                'turn_speed': 0.5
            },
            'manipulation': {
                'reach_distance': 1.0,
                'grasp_types': ['pinch', 'power', 'hook'],
                'max_load': 2.0
            },
            'sensors': ['camera', 'lidar', 'imu', 'force_torque'],
            'speakers': True
        }

    async def execute_action_plan(self, task_plan: Dict[str, Any]) -> bool:
        """Execute a planned sequence of actions"""
        actions = task_plan.get('actions', [])
        total_actions = len(actions)

        for i, action in enumerate(actions):
            self.get_logger().info(f'Executing action {i+1}/{total_actions}: {action}')

            # Update status
            status_msg = String()
            status_msg.data = f'Executing: {action.get("description", "unknown action")}'
            self.status_pub.publish(status_msg)

            # Execute the action
            success = await self.execute_single_action(action)

            if not success:
                self.get_logger().error(f'Action failed: {action}')
                return False

            # Small delay between actions
            await asyncio.sleep(0.5)

        return True

    async def execute_single_action(self, action: Dict[str, Any]) -> bool:
        """Execute a single action based on its type"""
        action_type = action.get('type', '').lower()

        # Publish action for monitoring
        action_msg = String()
        action_msg.data = json.dumps(action)
        self.action_pub.publish(action_msg)

        if action_type == 'navigate':
            return self.navigation_system.navigate_to(action.get('parameters', {}))
        elif action_type == 'grasp':
            return self.manipulation_controller.grasp_object(action.get('parameters', {}))
        elif action_type == 'place':
            return self.manipulation_controller.place_object(action.get('parameters', {}))
        elif action_type == 'speak':
            return self.speak_response(action.get('parameters', {}).get('text', ''))
        elif action_type == 'detect':
            return self.vision_system.detect_object(action.get('parameters', {}))
        elif action_type == 'approach':
            return self.navigation_system.approach_object(action.get('parameters', {}))
        else:
            self.get_logger().warn(f'Unknown action type: {action_type}')
            return False

    def speak_response(self, text: str) -> bool:
        """Speak a response to the user"""
        try:
            # In practice, this would use a text-to-speech system
            self.get_logger().info(f'Robot says: {text}')

            # Publish to speech synthesis topic
            speech_msg = String()
            speech_msg.data = text
            self.speech_pub.publish(speech_msg)

            return True
        except Exception as e:
            self.get_logger().error(f'Error in speech synthesis: {e}')
            return False

    def publish_response(self, response: str):
        """Publish response to user"""
        response_msg = String()
        response_msg.data = response
        self.response_pub.publish(response_msg)

        # Add to conversation history
        self.system_state['conversation_history'].append({
            'timestamp': time.time(),
            'type': 'response',
            'content': response
        })

    def goal_callback(self, goal_request):
        """Handle task execution goal"""
        self.get_logger().info('Received task execution goal')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Handle task execution cancellation"""
        self.get_logger().info('Received task cancellation')
        return CancelResponse.ACCEPT

    async def execute_task_callback(self, goal_handle):
        """Execute a specific task via action server"""
        self.get_logger().info('Executing task via action server')

        feedback_msg = ExecuteTask.Feedback()
        result = ExecuteTask.Result()

        try:
            # Parse the task from goal
            task_data = json.loads(goal_handle.request.task)

            # Execute the task
            success = await self.execute_action_plan(task_data)

            if success:
                result.success = True
                result.message = "Task completed successfully"
            else:
                result.success = False
                result.message = "Task execution failed"

        except Exception as e:
            result.success = False
            result.message = f"Error executing task: {e}"

        goal_handle.succeed()
        return result

class MockSpeechRecognizer:
    """Mock speech recognition system for demonstration"""
    def __init__(self):
        pass

class MockLLMTaskPlanner:
    """Mock LLM task planner for demonstration"""
    def plan_task(self, command: str, environment_state: Dict[str, Any]) -> Dict[str, Any]:
        """Mock task planning - in practice, this would call an LLM"""
        # Simple command parsing for demonstration
        command_lower = command.lower()

        if 'bring me' in command_lower or 'get me' in command_lower:
            target_object = 'cup'  # Extract from command in practice
            return {
                'actions': [
                    {
                        'type': 'detect',
                        'parameters': {'object': target_object},
                        'description': f'Detect {target_object}'
                    },
                    {
                        'type': 'navigate',
                        'parameters': {'target_object': target_object},
                        'description': f'Navigate to {target_object}'
                    },
                    {
                        'type': 'grasp',
                        'parameters': {'object': target_object},
                        'description': f'Grasp {target_object}'
                    },
                    {
                        'type': 'navigate',
                        'parameters': {'target_location': [0, 0, 0]},  # Return to user
                        'description': 'Return to user'
                    },
                    {
                        'type': 'place',
                        'parameters': {'location': 'near_user'},
                        'description': 'Place near user'
                    }
                ],
                'confidence': 0.8,
                'reasoning': 'User requested to bring an object'
            }
        elif 'go to' in command_lower or 'move to' in command_lower:
            # Extract location from command
            return {
                'actions': [
                    {
                        'type': 'navigate',
                        'parameters': {'target_location': [1, 1, 0]},  # Example location
                        'description': 'Navigate to specified location'
                    }
                ],
                'confidence': 0.9,
                'reasoning': 'User requested navigation'
            }
        else:
            # Default response
            return {
                'actions': [
                    {
                        'type': 'speak',
                        'parameters': {'text': f"I'm not sure how to {command}"},
                        'description': f'Apologize for not understanding {command}'
                    }
                ],
                'confidence': 0.3,
                'reasoning': 'Command not understood'
            }

class MockVisionSystem:
    """Mock vision system for demonstration"""
    def __init__(self):
        self.current_scene = []

    def process_image(self, image_msg):
        """Mock image processing"""
        # In practice, this would run object detection and pose estimation
        return [
            {'class': 'cup', 'position': [0.5, 0.2, 0.1], 'confidence': 0.9},
            {'class': 'book', 'position': [0.8, -0.1, 0.1], 'confidence': 0.85}
        ]

    def update_scene(self, objects):
        """Update current scene with detected objects"""
        self.current_scene = objects

    def detect_object(self, params):
        """Mock object detection action"""
        target_object = params.get('object', 'unknown')
        # Find object in current scene
        for obj in self.current_scene:
            if obj['class'] == target_object:
                return True
        return False

class MockManipulationController:
    """Mock manipulation controller for demonstration"""
    def __init__(self):
        self.joint_states = None

    def update_joint_states(self, joint_msg):
        """Update with current joint states"""
        self.joint_states = joint_msg

    def grasp_object(self, params):
        """Mock grasping action"""
        target_object = params.get('object', 'unknown')
        self.get_logger().info(f'Attempting to grasp {target_object}')
        # In practice, this would execute grasp planning and control
        return True  # Simulate success

    def place_object(self, params):
        """Mock placing action"""
        location = params.get('location', 'default')
        self.get_logger().info(f'Attempting to place object at {location}')
        # In practice, this would execute placement planning and control
        return True  # Simulate success

class MockNavigationSystem:
    """Mock navigation system for demonstration"""
    def navigate_to(self, params):
        """Mock navigation action"""
        target = params.get('target_location') or params.get('target_object')
        self.get_logger().info(f'Navigating to {target}')
        # In practice, this would use Nav2 or similar navigation stack
        return True  # Simulate success

    def approach_object(self, params):
        """Mock approach action"""
        target_object = params.get('object', 'unknown')
        self.get_logger().info(f'Approaching {target_object}')
        return True  # Simulate success

def main(args=None):
    rclpy.init(args=args)

    vla_system = IntegratedVLASystem()

    try:
        rclpy.spin(vla_system)
    except KeyboardInterrupt:
        pass
    finally:
        vla_system.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Real-World Integration Example

Let's create a more realistic integration example that connects all the components we've developed:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from sensor_msgs.msg import Image, LaserScan, Imu, JointState
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from visualization_msgs.msg import MarkerArray
from action_msgs.msg import GoalStatus
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
import threading
import asyncio
import json
import time
import queue
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

@dataclass
class VLAPipelineState:
    """State of the VLA pipeline"""
    speech_input: Optional[str] = None
    vision_data: Optional[Dict] = None
    task_plan: Optional[Dict] = None
    execution_status: str = "idle"
    robot_state: Optional[Dict] = None
    environment_state: Optional[Dict] = None

class RealisticVLASystem(Node):
    """
    Realistic integrated VLA system with proper component integration
    """
    def __init__(self):
        super().__init__('realistic_vla_system')

        # Initialize pipeline state
        self.pipeline_state = VLAPipelineState()

        # Communication queues between components
        self.speech_to_planning = queue.Queue()
        self.planning_to_execution = queue.Queue()
        self.execution_to_monitoring = queue.Queue()

        # Initialize all components
        self.initialize_components()

        # Set up ROS 2 interfaces
        self.setup_ros_interfaces()

        # Start processing threads
        self.start_processing_threads()

        # Performance monitoring
        self.performance_stats = {
            'speech_to_response_time': [],
            'task_completion_rate': 0.0,
            'system_uptime': time.time()
        }

        self.get_logger().info('Realistic VLA System initialized')

    def initialize_components(self):
        """Initialize all system components with proper configuration"""
        # Speech recognition component
        self.speech_recognizer = RealisticSpeechRecognizer(
            node=self,
            callback_group=self.callback_group
        )

        # Vision processing component
        self.vision_processor = RealisticVisionProcessor(
            node=self,
            callback_group=self.callback_group
        )

        # Task planning component
        self.task_planner = RealisticTaskPlanner(
            node=self,
            callback_group=self.callback_group
        )

        # Action execution component
        self.action_executor = RealisticActionExecutor(
            node=self,
            callback_group=self.callback_group
        )

        # System monitoring component
        self.system_monitor = RealisticSystemMonitor(
            node=self,
            callback_group=self.callback_group
        )

    def setup_ros_interfaces(self):
        """Set up all ROS 2 communication interfaces"""
        # Create a reentrant callback group for multi-threading
        self.callback_group = ReentrantCallbackGroup()

        # Publishers
        self.response_pub = self.create_publisher(
            String, '/vla_system/response', 10, callback_group=self.callback_group
        )
        self.status_pub = self.create_publisher(
            String, '/vla_system/status', 10, callback_group=self.callback_group
        )
        self.visualization_pub = self.create_publisher(
            MarkerArray, '/vla_system/visualization', 10, callback_group=self.callback_group
        )
        self.system_metrics_pub = self.create_publisher(
            String, '/vla_system/metrics', 10, callback_group=self.callback_group
        )

        # Subscribers
        self.speech_sub = self.create_subscription(
            String, '/speech_recognition/text',
            self.speech_callback, 10, callback_group=self.callback_group
        )
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw',
            self.image_callback, 10, callback_group=self.callback_group
        )
        self.odom_sub = self.create_subscription(
            Odometry, '/odom',
            self.odom_callback, 10, callback_group=self.callback_group
        )
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan',
            self.scan_callback, 10, callback_group=self.callback_group
        )
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data',
            self.imu_callback, 10, callback_group=self.callback_group
        )
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states',
            self.joint_state_callback, 10, callback_group=self.callback_group
        )

    def start_processing_threads(self):
        """Start processing threads for each component"""
        # Speech processing thread
        self.speech_thread = threading.Thread(target=self.speech_processing_loop, daemon=True)
        self.speech_thread.start()

        # Vision processing thread
        self.vision_thread = threading.Thread(target=self.vision_processing_loop, daemon=True)
        self.vision_thread.start()

        # Task planning thread
        self.planning_thread = threading.Thread(target=self.planning_loop, daemon=True)
        self.planning_thread.start()

        # Action execution thread
        self.execution_thread = threading.Thread(target=self.execution_loop, daemon=True)
        self.execution_thread.start()

        # Monitoring thread
        self.monitoring_thread = threading.Thread(target=self.monitoring_loop, daemon=True)
        self.monitoring_thread.start()

    def speech_callback(self, msg):
        """Handle incoming speech recognition results"""
        self.speech_to_planning.put({
            'type': 'speech_input',
            'data': msg.data,
            'timestamp': time.time()
        })

    def image_callback(self, msg):
        """Handle incoming image data"""
        self.vision_processor.process_image_async(msg)

    def odom_callback(self, msg):
        """Handle odometry updates"""
        robot_state = {
            'position': [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z],
            'orientation': [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
                          msg.pose.pose.orientation.z, msg.pose.pose.orientation.w],
            'velocity': [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z]
        }
        self.pipeline_state.robot_state = robot_state

    def scan_callback(self, msg):
        """Handle laser scan data for navigation"""
        # Process scan for obstacle detection and navigation
        self.vision_processor.process_scan_data(msg)

    def imu_callback(self, msg):
        """Handle IMU data for balance and orientation"""
        imu_data = {
            'linear_acceleration': [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z],
            'angular_velocity': [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],
            'orientation': [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        }
        self.system_monitor.update_imu_data(imu_data)

    def joint_state_callback(self, msg):
        """Handle joint state updates"""
        joint_data = dict(zip(msg.name, msg.position))
        self.action_executor.update_joint_states(joint_data)

    def speech_processing_loop(self):
        """Process speech input and convert to actionable commands"""
        while rclpy.ok():
            try:
                # Get speech input from queue
                if not self.speech_to_planning.empty():
                    item = self.speech_to_planning.get_nowait()

                    if item['type'] == 'speech_input':
                        command = item['data']

                        # Update pipeline state
                        self.pipeline_state.speech_input = command

                        # Process the command and add to planning queue
                        planning_request = {
                            'command': command,
                            'timestamp': item['timestamp'],
                            'context': self.get_context_for_planning()
                        }

                        self.planning_to_execution.put({
                            'type': 'plan_request',
                            'data': planning_request
                        })

                        # Update status
                        status_msg = String()
                        status_msg.data = f'Processing speech: {command[:50]}...'
                        self.status_pub.publish(status_msg)

                time.sleep(0.01)  # Small delay to prevent busy waiting

            except queue.Empty:
                time.sleep(0.01)
            except Exception as e:
                self.get_logger().error(f'Error in speech processing: {e}')

    def vision_processing_loop(self):
        """Process visual input and update environment understanding"""
        while rclpy.ok():
            try:
                # Process any pending vision tasks
                vision_update = self.vision_processor.get_latest_update()
                if vision_update:
                    self.pipeline_state.vision_data = vision_update

                    # Update environment state
                    self.update_environment_state(vision_update)

                time.sleep(0.05)  # Process at 20Hz

            except Exception as e:
                self.get_logger().error(f'Error in vision processing: {e}')

    def planning_loop(self):
        """Plan tasks based on speech input and current state"""
        while rclpy.ok():
            try:
                if not self.planning_to_execution.empty():
                    item = self.planning_to_execution.get_nowait()

                    if item['type'] == 'plan_request':
                        request = item['data']

                        # Get current environment state
                        env_state = self.get_current_environment_state()

                        # Plan the task using the task planner
                        start_time = time.time()
                        task_plan = self.task_planner.plan_task(
                            request['command'],
                            env_state,
                            request['context']
                        )
                        planning_time = time.time() - start_time

                        if task_plan:
                            # Store the plan
                            self.pipeline_state.task_plan = task_plan

                            # Add to execution queue
                            execution_request = {
                                'plan': task_plan,
                                'original_command': request['command'],
                                'planning_time': planning_time
                            }

                            self.execution_to_monitoring.put({
                                'type': 'execution_request',
                                'data': execution_request
                            })

                            self.get_logger().info(f'Task planned successfully: {request["command"]}')
                        else:
                            # Failed to plan, respond to user
                            response = f"Sorry, I couldn't understand how to {request['command']}"
                            self.publish_response(response)

                time.sleep(0.01)

            except queue.Empty:
                time.sleep(0.01)
            except Exception as e:
                self.get_logger().error(f'Error in planning loop: {e}')

    def execution_loop(self):
        """Execute planned tasks"""
        while rclpy.ok():
            try:
                if not self.execution_to_monitoring.empty():
                    item = self.execution_to_monitoring.get_nowait()

                    if item['type'] == 'execution_request':
                        request = item['data']

                        # Update execution status
                        self.pipeline_state.execution_status = 'executing'

                        # Execute the task plan
                        start_time = time.time()
                        success = self.action_executor.execute_plan(request['plan'])
                        execution_time = time.time() - start_time

                        # Update performance stats
                        self.performance_stats['speech_to_response_time'].append(
                            request['planning_time'] + execution_time
                        )

                        # Publish result
                        if success:
                            response = f"I have completed: {request['original_command']}"
                            self.performance_stats['task_completion_rate'] += 1
                        else:
                            response = f"I couldn't complete: {request['original_command']}"

                        self.publish_response(response)

                        # Update execution status
                        self.pipeline_state.execution_status = 'idle'

                time.sleep(0.01)

            except queue.Empty:
                time.sleep(0.01)
            except Exception as e:
                self.get_logger().error(f'Error in execution loop: {e}')

    def monitoring_loop(self):
        """Monitor system performance and safety"""
        while rclpy.ok():
            try:
                # Update system metrics
                self.publish_system_metrics()

                # Check system health
                if self.system_monitor.is_system_healthy():
                    status = 'healthy'
                else:
                    status = 'degraded'
                    self.get_logger().warn('System health degraded')

                # Update status
                status_msg = String()
                status_msg.data = f'System status: {status}, Executing: {self.pipeline_state.execution_status}'
                self.status_pub.publish(status_msg)

                time.sleep(1.0)  # Update metrics every second

            except Exception as e:
                self.get_logger().error(f'Error in monitoring loop: {e}')

    def get_context_for_planning(self) -> Dict[str, Any]:
        """Get context for task planning"""
        return {
            'robot_state': self.pipeline_state.robot_state,
            'environment_state': self.pipeline_state.environment_state,
            'vision_data': self.pipeline_state.vision_data,
            'conversation_history': getattr(self.task_planner, 'conversation_history', [])
        }

    def get_current_environment_state(self) -> Dict[str, Any]:
        """Get current environment state"""
        return {
            'robot_state': self.pipeline_state.robot_state or {},
            'detected_objects': self.vision_processor.get_detected_objects() if hasattr(self, 'vision_processor') else [],
            'obstacles': self.vision_processor.get_obstacles() if hasattr(self, 'vision_processor') else [],
            'navigation_map': self.vision_processor.get_navigation_map() if hasattr(self, 'vision_processor') else {},
            'battery_level': self.system_monitor.get_battery_level() if hasattr(self, 'system_monitor') else 100.0
        }

    def update_environment_state(self, vision_update: Dict[str, Any]):
        """Update environment state with new vision data"""
        if self.pipeline_state.environment_state is None:
            self.pipeline_state.environment_state = {}

        self.pipeline_state.environment_state.update(vision_update)

    def publish_response(self, response: str):
        """Publish response to user"""
        response_msg = String()
        response_msg.data = response
        self.response_pub.publish(response_msg)

    def publish_system_metrics(self):
        """Publish system performance metrics"""
        metrics = {
            'speech_to_response_time_avg': sum(self.performance_stats['speech_to_response_time']) / len(self.performance_stats['speech_to_response_time']) if self.performance_stats['speech_to_response_time'] else 0,
            'task_completion_rate': self.performance_stats['task_completion_rate'],
            'system_uptime': time.time() - self.performance_stats['system_uptime'],
            'current_status': self.pipeline_state.execution_status
        }

        metrics_msg = String()
        metrics_msg.data = json.dumps(metrics)
        self.system_metrics_pub.publish(metrics_msg)

class RealisticSpeechRecognizer:
    """Realistic speech recognition component"""
    def __init__(self, node, callback_group):
        self.node = node
        self.callback_group = callback_group
        self.active = True

    def process_speech(self, audio_data):
        """Process audio data and return text"""
        # In practice, this would interface with Whisper or similar
        pass

class RealisticVisionProcessor:
    """Realistic vision processing component"""
    def __init__(self, node, callback_group):
        self.node = node
        self.callback_group = callback_group
        self.detected_objects = []
        self.obstacles = []
        self.navigation_map = {}
        self.latest_image = None

    def process_image_async(self, image_msg):
        """Process image asynchronously"""
        # In practice, this would run object detection, pose estimation, etc.
        # For now, store the image and process in background
        self.latest_image = image_msg

    def get_latest_update(self):
        """Get latest vision update"""
        # Process the latest image if available
        if self.latest_image:
            # Simulate processing
            update = {
                'timestamp': time.time(),
                'objects': self.detected_objects,
                'obstacles': self.obstacles,
                'map_update': self.navigation_map
            }
            self.latest_image = None
            return update
        return None

    def get_detected_objects(self):
        """Get currently detected objects"""
        return self.detected_objects

    def get_obstacles(self):
        """Get detected obstacles"""
        return self.obstacles

    def get_navigation_map(self):
        """Get navigation map"""
        return self.navigation_map

    def process_scan_data(self, scan_msg):
        """Process laser scan data for navigation"""
        # Process scan for obstacle detection
        pass

class RealisticTaskPlanner:
    """Realistic task planning component"""
    def __init__(self, node, callback_group):
        self.node = node
        self.callback_group = callback_group
        self.conversation_history = []

    def plan_task(self, command: str, environment_state: Dict[str, Any], context: Dict[str, Any]):
        """Plan a task using LLM with context"""
        # In practice, this would call an LLM with proper prompt engineering
        # For demonstration, using simple rule-based planning

        # Add to conversation history
        self.conversation_history.append({
            'command': command,
            'timestamp': time.time()
        })

        # Simple task planning based on command
        if 'bring' in command.lower() or 'get' in command.lower():
            return self.plan_retrieval_task(command, environment_state)
        elif 'go to' in command.lower() or 'navigate to' in command.lower():
            return self.plan_navigation_task(command, environment_state)
        elif 'pick up' in command.lower() or 'grasp' in command.lower():
            return self.plan_grasp_task(command, environment_state)
        else:
            return self.plan_generic_task(command, environment_state)

    def plan_retrieval_task(self, command, env_state):
        """Plan object retrieval task"""
        return {
            'actions': [
                {'type': 'detect', 'parameters': {'object': 'target_object'}, 'description': 'Detect target object'},
                {'type': 'navigate', 'parameters': {'target': 'object_location'}, 'description': 'Navigate to object'},
                {'type': 'grasp', 'parameters': {'object': 'target_object'}, 'description': 'Grasp the object'},
                {'type': 'navigate', 'parameters': {'target': 'delivery_location'}, 'description': 'Navigate to delivery location'},
                {'type': 'place', 'parameters': {'location': 'delivery_location'}, 'description': 'Place the object'}
            ],
            'confidence': 0.8,
            'reasoning': 'User requested to retrieve an object'
        }

    def plan_navigation_task(self, command, env_state):
        """Plan navigation task"""
        return {
            'actions': [
                {'type': 'navigate', 'parameters': {'target': 'destination'}, 'description': 'Navigate to destination'}
            ],
            'confidence': 0.9,
            'reasoning': 'User requested navigation'
        }

    def plan_grasp_task(self, command, env_state):
        """Plan grasping task"""
        return {
            'actions': [
                {'type': 'detect', 'parameters': {'object': 'target_object'}, 'description': 'Detect target object'},
                {'type': 'approach', 'parameters': {'object': 'target_object'}, 'description': 'Approach the object'},
                {'type': 'grasp', 'parameters': {'object': 'target_object'}, 'description': 'Grasp the object'}
            ],
            'confidence': 0.85,
            'reasoning': 'User requested to grasp an object'
        }

    def plan_generic_task(self, command, env_state):
        """Plan generic task"""
        return {
            'actions': [
                {'type': 'speak', 'parameters': {'text': f"I'm not sure how to {command}"}, 'description': 'Apologize for not understanding'}
            ],
            'confidence': 0.3,
            'reasoning': 'Command not understood'
        }

class RealisticActionExecutor:
    """Realistic action execution component"""
    def __init__(self, node, callback_group):
        self.node = node
        self.callback_group = callback_group
        self.joint_states = {}

    def execute_plan(self, plan: Dict[str, Any]) -> bool:
        """Execute a planned sequence of actions"""
        actions = plan.get('actions', [])

        for action in actions:
            success = self.execute_single_action(action)
            if not success:
                self.node.get_logger().error(f'Action failed: {action}')
                return False

        return True

    def execute_single_action(self, action: Dict[str, Any]) -> bool:
        """Execute a single action"""
        action_type = action.get('type', '').lower()

        if action_type == 'navigate':
            return self.execute_navigation(action)
        elif action_type == 'grasp':
            return self.execute_grasp(action)
        elif action_type == 'place':
            return self.execute_place(action)
        elif action_type == 'detect':
            return self.execute_detection(action)
        elif action_type == 'approach':
            return self.execute_approach(action)
        elif action_type == 'speak':
            return self.execute_speech(action)
        else:
            self.node.get_logger().warn(f'Unknown action type: {action_type}')
            return False

    def execute_navigation(self, action):
        """Execute navigation action"""
        # In practice, this would interface with Nav2
        self.node.get_logger().info('Executing navigation action')
        return True

    def execute_grasp(self, action):
        """Execute grasping action"""
        # In practice, this would interface with manipulation stack
        self.node.get_logger().info('Executing grasp action')
        return True

    def execute_place(self, action):
        """Execute placement action"""
        # In practice, this would interface with manipulation stack
        self.node.get_logger().info('Executing place action')
        return True

    def execute_detection(self, action):
        """Execute detection action"""
        # In practice, this would trigger vision processing
        self.node.get_logger().info('Executing detection action')
        return True

    def execute_approach(self, action):
        """Execute approach action"""
        # In practice, this would move robot closer to object
        self.node.get_logger().info('Executing approach action')
        return True

    def execute_speech(self, action):
        """Execute speech action"""
        text = action.get('parameters', {}).get('text', '')
        self.node.get_logger().info(f'Executing speech: {text}')
        return True

    def update_joint_states(self, joint_data: Dict[str, float]):
        """Update with current joint states"""
        self.joint_states.update(joint_data)

class RealisticSystemMonitor:
    """Realistic system monitoring component"""
    def __init__(self, node, callback_group):
        self.node = node
        self.callback_group = callback_group
        self.imu_data = {}
        self.battery_level = 100.0
        self.system_health = True

    def is_system_healthy(self) -> bool:
        """Check if system is healthy"""
        # Check various system parameters
        if self.battery_level < 10.0:
            return False
        if not self.system_health:
            return False

        # Check IMU data for balance issues
        if 'orientation' in self.imu_data:
            # Check if robot is tilted beyond safe limits
            orientation = self.imu_data['orientation']
            # Simple check for significant tilt
            if abs(orientation[0]) > 0.5 or abs(orientation[1]) > 0.5:  # 30 degrees
                return False

        return True

    def update_imu_data(self, imu_data: Dict[str, Any]):
        """Update with new IMU data"""
        self.imu_data.update(imu_data)

    def get_battery_level(self) -> float:
        """Get current battery level"""
        return self.battery_level

def main(args=None):
    rclpy.init(args=args)

    vla_system = RealisticVLASystem()

    try:
        rclpy.spin(vla_system)
    except KeyboardInterrupt:
        pass
    finally:
        vla_system.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### System Integration Best Practices

When integrating all VLA components, consider these best practices:

```python
class VLASystemBestPractices:
    """
    Best practices for VLA system integration
    """

    @staticmethod
    def handle_component_failures():
        """
        Best practice: Implement graceful degradation when components fail
        """
        # Example implementation
        class ResilientVLAComponent:
            def __init__(self):
                self.primary_component = None
                self.backup_component = None
                self.fallback_behavior = None

            def execute_with_fallback(self, input_data):
                try:
                    # Try primary component
                    result = self.primary_component.process(input_data)
                    return result
                except Exception as e:
                    print(f"Primary component failed: {e}")

                    try:
                        # Try backup component
                        result = self.backup_component.process(input_data)
                        return result
                    except Exception as e2:
                        print(f"Backup component failed: {e2}")

                        # Execute fallback behavior
                        return self.fallback_behavior.execute(input_data)

    @staticmethod
    def manage_real_time_constraints():
        """
        Best practice: Ensure real-time performance across all components
        """
        # Example: Priority-based task scheduling
        class PriorityScheduler:
            def __init__(self):
                self.high_priority_tasks = []  # Safety, balance
                self.medium_priority_tasks = []  # Navigation, basic control
                self.low_priority_tasks = []    # Vision processing, planning

            def schedule_tasks(self):
                # Execute high priority tasks first
                for task in self.high_priority_tasks:
                    if task.is_ready():
                        task.execute()

                # Execute medium priority tasks if time allows
                if self.time_remaining() > 0.01:  # 10ms buffer
                    for task in self.medium_priority_tasks:
                        if task.is_ready():
                            task.execute()

                # Execute low priority tasks with remaining time
                if self.time_remaining() > 0.05:  # 50ms buffer
                    for task in self.low_priority_tasks:
                        if task.is_ready():
                            task.execute()

    @staticmethod
    def ensure_data_consistency():
        """
        Best practice: Maintain data consistency across components
        """
        # Example: Timestamp-based data synchronization
        class SynchronizedDataBuffer:
            def __init__(self, max_age=1.0):  # Maximum age in seconds
                self.buffers = {}
                self.max_age = max_age

            def add_data(self, source, data, timestamp=None):
                if timestamp is None:
                    timestamp = time.time()

                self.buffers[source] = {
                    'data': data,
                    'timestamp': timestamp,
                    'valid': True
                }

            def get_synchronized_data(self, sources, target_time=None):
                if target_time is None:
                    target_time = time.time()

                result = {}
                for source in sources:
                    if source in self.buffers:
                        buffer = self.buffers[source]
                        age = target_time - buffer['timestamp']

                        if age <= self.max_age and buffer['valid']:
                            result[source] = buffer['data']
                        else:
                            result[source] = None  # Data too old
                    else:
                        result[source] = None  # No data available

                return result

    @staticmethod
    def implement_safety_systems():
        """
        Best practice: Implement comprehensive safety systems
        """
        # Example: Multi-layered safety architecture
        class SafetySystem:
            def __init__(self):
                self.hard_limits = HardLimitChecker()
                self.soft_limits = SoftLimitChecker()
                self.emergency_stop = EmergencyStopSystem()
                self.monitoring = ContinuousMonitoring()

            def check_safety(self, action):
                # Check hard limits first (immediate stop if violated)
                if not self.hard_limits.check(action):
                    self.emergency_stop.trigger()
                    return False

                # Check soft limits (warning and potential adjustment)
                soft_check = self.soft_limits.check(action)
                if not soft_check['safe']:
                    # Adjust action to be within soft limits
                    action = self.adjust_for_safety(action, soft_check['adjustment'])

                # Continue monitoring during execution
                self.monitoring.start_monitoring(action)

                return True

    @staticmethod
    def optimize_resource_usage():
        """
        Best practice: Optimize resource usage across components
        """
        # Example: Resource-aware component activation
        class ResourceAwareSystem:
            def __init__(self):
                self.resource_monitor = ResourceMonitor()
                self.components = {}
                self.resource_budgets = {}

            def update_component_resources(self):
                available_resources = self.resource_monitor.get_available_resources()

                for component_name, component in self.components.items():
                    budget = self.resource_budgets[component_name]

                    if available_resources >= budget['minimum']:
                        component.activate()
                        if available_resources >= budget['preferred']:
                            component.set_performance_mode('high')
                        else:
                            component.set_performance_mode('medium')
                    else:
                        component.deactivate()

    @staticmethod
    def handle_async_communication():
        """
        Best practice: Properly handle asynchronous communication
        """
        # Example: Async communication with proper error handling
        import asyncio
        import concurrent.futures

        class AsyncVLACommunicator:
            def __init__(self):
                self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
                self.timeout = 5.0  # seconds

            async def communicate_with_component(self, component, message):
                try:
                    # Run the communication in a separate thread
                    loop = asyncio.get_event_loop()
                    result = await asyncio.wait_for(
                        loop.run_in_executor(self.executor, component.process, message),
                        timeout=self.timeout
                    )
                    return result
                except asyncio.TimeoutError:
                    print(f"Timeout communicating with {component}")
                    return None
                except Exception as e:
                    print(f"Error communicating with {component}: {e}")
                    return None
```

### Performance Monitoring and Optimization

Implement comprehensive monitoring for the integrated system:

```python
import time
import statistics
from collections import deque
import threading

class VLASystemMonitor:
    """
    Comprehensive monitoring for VLA system performance
    """
    def __init__(self):
        self.metrics = {
            'speech_recognition': deque(maxlen=100),
            'task_planning': deque(maxlen=100),
            'action_execution': deque(maxlen=100),
            'vision_processing': deque(maxlen=100),
            'end_to_end_latency': deque(maxlen=100)
        }

        self.counters = {
            'total_commands': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'safety_interventions': 0
        }

        self.system_resources = {
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'gpu_usage': 0.0,
            'network_latency': 0.0
        }

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self.monitor_resources)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()

    def record_metric(self, component: str, value: float):
        """Record a performance metric"""
        if component in self.metrics:
            self.metrics[component].append(value)

    def increment_counter(self, counter: str):
        """Increment a performance counter"""
        if counter in self.counters:
            self.counters[counter] += 1

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        report = {}

        # Calculate averages for each metric
        for component, values in self.metrics.items():
            if values:
                report[f'{component}_avg_time'] = statistics.mean(values)
                report[f'{component}_min_time'] = min(values)
                report[f'{component}_max_time'] = max(values)
                report[f'{component}_stddev'] = statistics.stdev(values) if len(values) > 1 else 0

        # Calculate success rate
        total = self.counters['successful_executions'] + self.counters['failed_executions']
        if total > 0:
            report['success_rate'] = self.counters['successful_executions'] / total
        else:
            report['success_rate'] = 0.0

        # Add system resource usage
        report['system_resources'] = self.system_resources.copy()

        # Add counters
        report['counters'] = self.counters.copy()

        return report

    def monitor_resources(self):
        """Monitor system resources in background"""
        while self.monitoring_active:
            try:
                # Monitor CPU usage (simplified)
                import psutil
                self.system_resources['cpu_usage'] = psutil.cpu_percent()
                self.system_resources['memory_usage'] = psutil.virtual_memory().percent

                # Monitor GPU usage if available
                try:
                    import GPUtil
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        self.system_resources['gpu_usage'] = gpus[0].load * 100
                except ImportError:
                    self.system_resources['gpu_usage'] = 0.0

                time.sleep(1.0)  # Update every second

            except Exception as e:
                print(f"Error monitoring resources: {e}")
                time.sleep(1.0)

    def should_slow_down_components(self) -> bool:
        """Check if system should reduce performance to manage resources"""
        high_resource_usage = (
            self.system_resources['cpu_usage'] > 80.0 or
            self.system_resources['memory_usage'] > 85.0 or
            self.system_resources['gpu_usage'] > 85.0
        )

        return high_resource_usage
```

### Summary

In this chapter, we've integrated all the components of the Vision-Language-Action system into a complete, end-to-end humanoid robot system. We covered:

- Complete system architecture and component integration
- Realistic implementation of the integrated pipeline
- Best practices for system integration
- Performance monitoring and optimization strategies
- Safety considerations for integrated systems

This completes Module 4: Vision-Language-Action (VLA), providing you with the knowledge to build complete humanoid robot systems that can understand natural language commands and execute complex tasks using vision-guided manipulation.