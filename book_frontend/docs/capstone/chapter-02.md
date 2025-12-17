---
sidebar_position: 2
---

# Chapter 02: Voice Command to Action Pipeline

## Implementing the Complete VLA Pipeline

In this chapter, we'll implement the complete Vision-Language-Action (VLA) pipeline that transforms natural language commands into physical robot actions. This is the heart of the humanoid robot's ability to understand and respond to human commands in a natural way.

### Understanding the VLA Pipeline

The Vision-Language-Action pipeline is a complex system that connects human language understanding with robot action execution. The pipeline consists of several interconnected stages:

```
┌─────────────────┐    ┌─────────────────┐    └─────────────────┘
│   Human        │ -> │  Language       │ -> │  Action         │
│   Voice        │    │  Processing     │    │  Planning &     │
│   Command      │    │  (LLM, NLP)     │    │  Execution      │
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

### Pipeline Architecture

The complete VLA pipeline architecture:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from humanoid_robot_msgs.msg import VoiceCommand, ActionPlan, SystemState
from humanoid_robot_msgs.srv import ProcessVoiceCommand
import asyncio
import threading
import time
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

@dataclass
class PipelineState:
    """State of the VLA pipeline"""
    current_command: Optional[str] = None
    command_timestamp: Optional[float] = None
    language_processing_result: Optional[Dict[str, Any]] = None
    vision_data: Optional[Dict[str, Any]] = None
    action_plan: Optional[Dict[str, Any]] = None
    execution_status: str = "idle"
    error_state: Optional[str] = None

class VoiceToActionPipeline(Node):
    """
    Complete Voice-to-Action pipeline for humanoid robot
    """
    def __init__(self):
        super().__init__('vla_pipeline')

        # Initialize pipeline state
        self.pipeline_state = PipelineState()

        # Initialize components
        self.initialize_components()

        # Set up ROS 2 interfaces
        self.setup_interfaces()

        # Initialize processing queues
        self.command_queue = asyncio.Queue()
        self.result_queue = asyncio.Queue()

        # Start processing threads
        self.pipeline_thread = threading.Thread(target=self.pipeline_processing_loop, daemon=True)
        self.pipeline_thread.start()

        # Performance monitoring
        self.processing_times = []
        self.success_count = 0
        self.failure_count = 0

        self.get_logger().info('Voice-to-Action Pipeline initialized')

    def initialize_components(self):
        """Initialize all pipeline components"""
        # Speech recognition component
        self.speech_recognizer = SpeechRecognitionComponent(self)

        # Natural Language Processing component
        self.nlp_processor = NLPProcessorComponent(self)

        # Vision processing component
        self.vision_processor = VisionProcessorComponent(self)

        # Task planning component
        self.task_planner = TaskPlanningComponent(self)

        # Action execution component
        self.action_executor = ActionExecutionComponent(self)

    def setup_interfaces(self):
        """Set up ROS 2 interfaces"""
        # Publishers
        self.response_pub = self.create_publisher(String, '/vla/response', 10)
        self.status_pub = self.create_publisher(String, '/vla/status', 10)
        self.action_plan_pub = self.create_publisher(ActionPlan, '/vla/action_plan', 10)
        self.visualization_pub = self.create_publisher(String, '/vla/visualization', 10)

        # Subscribers
        self.voice_command_sub = self.create_subscription(
            VoiceCommand, '/voice_command', self.voice_command_callback, 10
        )
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10
        )

        # Services
        self.process_command_srv = self.create_service(
            ProcessVoiceCommand, '/vla/process_command', self.process_command_service
        )

    def voice_command_callback(self, msg):
        """Handle incoming voice command"""
        command = msg.command
        self.get_logger().info(f'Received voice command: {command}')

        # Update pipeline state
        self.pipeline_state.current_command = command
        self.pipeline_state.command_timestamp = time.time()
        self.pipeline_state.execution_status = 'processing'

        # Add to processing queue
        asyncio.run_coroutine_threadsafe(
            self.command_queue.put({
                'type': 'voice_command',
                'data': command,
                'timestamp': time.time()
            }),
            asyncio.get_event_loop()
        )

        # Update status
        status_msg = String()
        status_msg.data = f'Processing: {command[:50]}...'
        self.status_pub.publish(status_msg)

    def image_callback(self, msg):
        """Handle incoming image data"""
        # Process image for vision data
        vision_result = self.vision_processor.process_image(msg)

        # Update pipeline state
        self.pipeline_state.vision_data = vision_result

    def pipeline_processing_loop(self):
        """Main pipeline processing loop"""
        # Create event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def process_pipeline():
            while rclpy.ok():
                try:
                    # Get next item from queue
                    if not self.command_queue.empty():
                        item = await self.command_queue.get()

                        if item['type'] == 'voice_command':
                            await self.process_voice_command(item['data'], item['timestamp'])

                    # Small delay to prevent busy waiting
                    await asyncio.sleep(0.01)

                except Exception as e:
                    self.get_logger().error(f'Pipeline processing error: {e}')
                    await asyncio.sleep(0.1)

        loop.run_until_complete(process_pipeline())

    async def process_voice_command(self, command: str, timestamp: float):
        """Process a voice command through the complete pipeline"""
        start_time = time.time()

        try:
            # Stage 1: Language Processing
            language_result = await self.process_language(command)

            if not language_result:
                error_msg = f"Failed to process language: {command}"
                self.handle_error(error_msg)
                return

            # Update pipeline state
            self.pipeline_state.language_processing_result = language_result

            # Stage 2: Context Integration (incorporate vision data)
            context = await self.build_context(language_result)

            # Stage 3: Task Planning
            task_plan = await self.plan_task(language_result, context)

            if not task_plan:
                error_msg = f"Failed to plan task for command: {command}"
                self.handle_error(error_msg)
                return

            # Update pipeline state
            self.pipeline_state.action_plan = task_plan

            # Stage 4: Action Execution
            execution_result = await self.execute_action_plan(task_plan)

            # Calculate processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)

            # Update success/failure counts
            if execution_result.get('success', False):
                self.success_count += 1
                response = f"I have completed: {command}"
            else:
                self.failure_count += 1
                response = f"I couldn't complete: {command}. {execution_result.get('error', '')}"

            # Publish response
            response_msg = String()
            response_msg.data = response
            self.response_pub.publish(response_msg)

            # Update status
            status_msg = String()
            status_msg.data = f'Completed: {command[:50]}... (Time: {processing_time:.2f}s)'
            self.status_pub.publish(status_msg)

            # Update pipeline state
            self.pipeline_state.execution_status = 'idle'
            self.pipeline_state.error_state = None

        except Exception as e:
            error_msg = f"Pipeline error processing command '{command}': {e}"
            self.handle_error(error_msg)
            self.failure_count += 1

    async def process_language(self, command: str) -> Optional[Dict[str, Any]]:
        """Process natural language command using NLP component"""
        try:
            # Use NLP component to understand the command
            result = await self.nlp_processor.process_command(command)

            self.get_logger().info(f'Language processing result: {result}')
            return result

        except Exception as e:
            self.get_logger().error(f'Language processing error: {e}')
            return None

    async def build_context(self, language_result: Dict[str, Any]) -> Dict[str, Any]:
        """Build context incorporating vision and environment data"""
        context = {
            'language_result': language_result,
            'vision_data': self.pipeline_state.vision_data or {},
            'robot_state': await self.get_robot_state(),
            'environment_map': await self.get_environment_map(),
            'conversation_history': getattr(self.nlp_processor, 'conversation_history', [])
        }

        return context

    async def plan_task(self, language_result: Dict[str, Any], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Plan task based on language understanding and context"""
        try:
            # Use task planner component
            plan = await self.task_planner.create_plan(language_result, context)

            self.get_logger().info(f'Task plan created: {plan}')

            # Publish action plan for visualization/monitoring
            action_plan_msg = ActionPlan()
            action_plan_msg.plan_data = json.dumps(plan)
            self.action_plan_pub.publish(action_plan_msg)

            return plan

        except Exception as e:
            self.get_logger().error(f'Task planning error: {e}')
            return None

    async def execute_action_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the planned action sequence"""
        try:
            # Execute the plan using action executor
            result = await self.action_executor.execute_plan(plan)

            self.get_logger().info(f'Action execution result: {result}')
            return result

        except Exception as e:
            self.get_logger().error(f'Action execution error: {e}')
            return {'success': False, 'error': str(e)}

    async def get_robot_state(self) -> Dict[str, Any]:
        """Get current robot state"""
        # In practice, this would get robot state from TF, joint states, etc.
        return {
            'position': [0, 0, 0],
            'orientation': [0, 0, 0, 1],
            'joint_states': {},
            'battery_level': 85.0,
            'operational_status': 'normal'
        }

    async def get_environment_map(self) -> Dict[str, Any]:
        """Get environment map and obstacles"""
        # In practice, this would get map from navigation system
        return {
            'known_locations': ['kitchen', 'living_room', 'bedroom'],
            'obstacles': [],
            'navigation_map': {}
        }

    def handle_error(self, error_msg: str):
        """Handle pipeline errors"""
        self.get_logger().error(error_msg)

        # Update pipeline state
        self.pipeline_state.error_state = error_msg
        self.pipeline_state.execution_status = 'error'

        # Publish error response
        response_msg = String()
        response_msg.data = f"Sorry, I encountered an error: {error_msg[:100]}..."
        self.response_pub.publish(response_msg)

        # Update status
        status_msg = String()
        status_msg.data = f'Error: {error_msg[:50]}...'
        self.status_pub.publish(status_msg)

    def process_command_service(self, request, response):
        """Service callback for processing voice commands"""
        # This allows external systems to trigger command processing
        command = request.command

        # Add to processing queue (non-async context)
        asyncio.run_coroutine_threadsafe(
            self.command_queue.put({
                'type': 'voice_command',
                'data': command,
                'timestamp': time.time()
            }),
            asyncio.get_event_loop()
        )

        response.success = True
        response.message = f"Command '{command}' queued for processing"
        return response

def main(args=None):
    rclpy.init(args=args)

    pipeline = VoiceToActionPipeline()

    try:
        rclpy.spin(pipeline)
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Speech Recognition Component

```python
class SpeechRecognitionComponent:
    """
    Speech recognition component for the VLA pipeline
    """
    def __init__(self, node):
        self.node = node
        self.is_listening = False
        self.recognition_model = None
        self.wake_word = "robot"
        self.sensitivity = 0.5

    async def initialize_model(self):
        """Initialize speech recognition model"""
        # In practice, this would load Whisper or similar model
        # For now, we'll simulate with mock recognition
        self.recognition_model = "mock_model"
        self.node.get_logger().info('Speech recognition model initialized')

    async def recognize_speech(self, audio_data) -> Optional[str]:
        """Recognize speech from audio data"""
        try:
            # In practice, this would call the actual recognition model
            # For simulation, we'll return a mock result
            recognized_text = self.simulate_recognition(audio_data)

            if recognized_text and len(recognized_text.strip()) > 0:
                self.node.get_logger().info(f'Speech recognized: {recognized_text}')
                return recognized_text
            else:
                self.node.get_logger().warn('No speech recognized')
                return None

        except Exception as e:
            self.node.get_logger().error(f'Speech recognition error: {e}')
            return None

    def simulate_recognition(self, audio_data):
        """Simulate speech recognition for demonstration"""
        # In a real system, this would process actual audio
        # For simulation, we'll return predefined commands
        return "Go to the kitchen and bring me a cup"

    async def detect_wake_word(self, audio_data) -> bool:
        """Detect wake word to activate listening"""
        # Simulate wake word detection
        # In practice, this would use keyword spotting models
        return True  # Always active for simulation
```

### Natural Language Processing Component

```python
class NLPProcessorComponent:
    """
    Natural Language Processing component for understanding commands
    """
    def __init__(self, node):
        self.node = node
        self.llm_client = None  # Will be initialized with OpenAI or similar
        self.conversation_history = []
        self.max_history = 10
        self.command_templates = self.load_command_templates()

    def load_command_templates(self) -> Dict[str, Any]:
        """Load command templates and parsing rules"""
        return {
            'navigation': {
                'patterns': [
                    r'go to (.+)',
                    r'move to (.+)',
                    r'go to the (.+)',
                    r'navigate to (.+)'
                ],
                'intent': 'navigate'
            },
            'manipulation': {
                'patterns': [
                    r'pick up (.+)',
                    r'grab (.+)',
                    r'bring me (.+)',
                    r'get (.+)'
                ],
                'intent': 'manipulate'
            },
            'interaction': {
                'patterns': [
                    r'say (.+)',
                    r'tell (.+)',
                    r'greet (.+)'
                ],
                'intent': 'interact'
            }
        }

    async def process_command(self, command: str) -> Dict[str, Any]:
        """Process natural language command and extract intent and entities"""
        try:
            # First, try pattern matching for quick parsing
            parsed_command = self.parse_command_patterns(command)

            if parsed_command:
                # Use LLM for more sophisticated understanding if needed
                enhanced_result = await self.enhance_with_llm(parsed_command, command)
                return enhanced_result

            # If pattern matching failed, use LLM directly
            llm_result = await self.process_with_llm(command)
            return llm_result

        except Exception as e:
            self.node.get_logger().error(f'NLP processing error: {e}')
            return {
                'intent': 'unknown',
                'entities': {},
                'action_type': 'unknown',
                'parameters': {},
                'confidence': 0.0
            }

    def parse_command_patterns(self, command: str) -> Optional[Dict[str, Any]]:
        """Parse command using pattern matching"""
        import re

        command_lower = command.lower()

        for category, data in self.command_templates.items():
            for pattern in data['patterns']:
                match = re.search(pattern, command_lower)
                if match:
                    entities = match.groups()

                    return {
                        'intent': data['intent'],
                        'action_type': category,
                        'entities': {'target': entities[0] if entities else ''},
                        'original_command': command,
                        'confidence': 0.8
                    }

        return None

    async def enhance_with_llm(self, parsed_command: Dict[str, Any], original_command: str) -> Dict[str, Any]:
        """Enhance parsed command with LLM understanding"""
        # In practice, this would call an LLM to refine understanding
        # For simulation, we'll return the parsed command with some enhancement
        parsed_command['original_command'] = original_command
        parsed_command['enhanced'] = True

        # Add context if available
        if 'target' in parsed_command['entities']:
            target = parsed_command['entities']['target']
            parsed_command['parameters'] = {
                'target_object': target if parsed_command['action_type'] == 'manipulation' else None,
                'target_location': target if parsed_command['action_type'] == 'navigation' else None
            }

        return parsed_command

    async def process_with_llm(self, command: str) -> Dict[str, Any]:
        """Process command directly with LLM"""
        # Simulate LLM processing
        # In practice, this would call OpenAI GPT or similar
        result = self.simulate_llm_processing(command)

        # Add to conversation history
        self.add_to_history(command, result)

        return result

    def simulate_llm_processing(self, command: str) -> Dict[str, Any]:
        """Simulate LLM processing for demonstration"""
        command_lower = command.lower()

        if any(word in command_lower for word in ['go', 'move', 'navigate', 'walk']):
            return {
                'intent': 'navigate',
                'action_type': 'navigation',
                'entities': {'location': self.extract_location(command)},
                'parameters': {'target_location': self.extract_location(command)},
                'confidence': 0.85,
                'original_command': command
            }
        elif any(word in command_lower for word in ['pick', 'grab', 'bring', 'get']):
            return {
                'intent': 'manipulate',
                'action_type': 'manipulation',
                'entities': {'object': self.extract_object(command)},
                'parameters': {'target_object': self.extract_object(command)},
                'confidence': 0.80,
                'original_command': command
            }
        elif any(word in command_lower for word in ['say', 'tell', 'speak']):
            return {
                'intent': 'interact',
                'action_type': 'interaction',
                'entities': {'message': self.extract_message(command)},
                'parameters': {'message': self.extract_message(command)},
                'confidence': 0.90,
                'original_command': command
            }
        else:
            return {
                'intent': 'unknown',
                'action_type': 'unknown',
                'entities': {},
                'parameters': {},
                'confidence': 0.3,
                'original_command': command
            }

    def extract_location(self, command: str) -> str:
        """Extract location from command"""
        locations = ['kitchen', 'living room', 'bedroom', 'office', 'bathroom', 'dining room']
        command_lower = command.lower()

        for loc in locations:
            if loc in command_lower:
                return loc.replace(' ', '_')

        return 'unknown_location'

    def extract_object(self, command: str) -> str:
        """Extract object from command"""
        objects = ['cup', 'bottle', 'book', 'phone', 'keys', 'water', 'coffee', 'food']
        command_lower = command.lower()

        for obj in objects:
            if obj in command_lower:
                return obj

        return 'unknown_object'

    def extract_message(self, command: str) -> str:
        """Extract message from command"""
        import re
        # Look for text after common verbs
        patterns = [
            r'say (.+)',
            r'tell (.+)',
            r'to (.+)',
            r'hello (.+)',
            r'hi (.+)'
        ]

        for pattern in patterns:
            match = re.search(pattern, command.lower())
            if match:
                return match.group(1).strip()

        return command

    def add_to_history(self, command: str, result: Dict[str, Any]):
        """Add interaction to conversation history"""
        entry = {
            'command': command,
            'result': result,
            'timestamp': time.time()
        }

        self.conversation_history.append(entry)

        # Keep only recent history
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]

    async def initialize_llm(self, api_key: str):
        """Initialize LLM client"""
        # In practice, this would initialize OpenAI or similar client
        pass
```

### Vision Processing Component

```python
class VisionProcessorComponent:
    """
    Vision processing component for object detection and scene understanding
    """
    def __init__(self, node):
        self.node = node
        self.object_detector = None
        self.pose_estimator = None
        self.scene_analyzer = None
        self.detected_objects = []
        self.last_processed_image = None

    async def process_image(self, image_msg) -> Dict[str, Any]:
        """Process image and extract scene information"""
        try:
            # Convert ROS image to format suitable for processing
            cv_image = self.ros_image_to_cv2(image_msg)

            # Detect objects in the image
            objects = await self.detect_objects(cv_image)

            # Estimate poses of detected objects
            objects_with_poses = await self.estimate_poses(objects, cv_image)

            # Analyze scene relationships
            scene_analysis = await self.analyze_scene(objects_with_poses)

            # Update internal state
            self.detected_objects = objects_with_poses
            self.last_processed_image = image_msg

            result = {
                'objects': objects_with_poses,
                'scene_analysis': scene_analysis,
                'timestamp': image_msg.header.stamp.sec + image_msg.header.stamp.nanosec * 1e-9,
                'image_dimensions': [cv_image.shape[1], cv_image.shape[0]]  # width, height
            }

            self.node.get_logger().info(f'Vision processing result: {len(objects)} objects detected')
            return result

        except Exception as e:
            self.node.get_logger().error(f'Vision processing error: {e}')
            return {'objects': [], 'scene_analysis': {}, 'error': str(e)}

    def ros_image_to_cv2(self, image_msg):
        """Convert ROS image message to OpenCV format"""
        from cv_bridge import CvBridge
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(image_msg, "bgr8")
        return cv_image

    async def detect_objects(self, cv_image) -> List[Dict[str, Any]]:
        """Detect objects in the image"""
        # In practice, this would use YOLO, Detectron2, or Isaac ROS vision
        # For simulation, we'll return mock detections
        return [
            {'class': 'cup', 'confidence': 0.92, 'bbox': [100, 150, 200, 250], 'center': [150, 200]},
            {'class': 'book', 'confidence': 0.88, 'bbox': [300, 100, 450, 200], 'center': [375, 150]},
            {'class': 'bottle', 'confidence': 0.85, 'bbox': [250, 300, 350, 450], 'center': [300, 375]}
        ]

    async def estimate_poses(self, objects: List[Dict], cv_image) -> List[Dict[str, Any]]:
        """Estimate 3D poses of detected objects"""
        # In practice, this would use 6D pose estimation models
        # For simulation, we'll add mock 3D positions
        for obj in objects:
            # Convert 2D center to 3D position (simplified)
            center_x, center_y = obj['center']

            # Mock 3D position based on 2D position and object class
            obj['position_3d'] = [
                (center_x - cv_image.shape[1]/2) / 1000.0,  # X (relative to center)
                (cv_image.shape[0]/2 - center_y) / 1000.0,  # Y (relative to center)
                0.8  # Z (distance - mock value)
            ]

            # Mock orientation
            obj['orientation'] = [0, 0, 0, 1]  # Quaternion (identity)

        return objects

    async def analyze_scene(self, objects_with_poses: List[Dict]) -> Dict[str, Any]:
        """Analyze scene relationships and context"""
        # Analyze spatial relationships between objects
        relationships = []

        for i, obj1 in enumerate(objects_with_poses):
            for j, obj2 in enumerate(objects_with_poses):
                if i != j:
                    # Calculate distance between objects
                    pos1 = obj1['position_3d']
                    pos2 = obj2['position_3d']
                    distance = sum((a - b)**2 for a, b in zip(pos1, pos2))**0.5

                    if distance < 0.5:  # Within 50cm
                        relationships.append({
                            'object1': obj1['class'],
                            'object2': obj2['class'],
                            'relationship': 'near',
                            'distance': distance
                        })

        # Identify potential interaction targets
        interaction_targets = [
            obj for obj in objects_with_poses
            if obj['class'] in ['cup', 'bottle', 'book', 'phone']
        ]

        return {
            'relationships': relationships,
            'interaction_targets': interaction_targets,
            'reachable_objects': interaction_targets[:3],  # First 3 are reachable
            'environment_description': f"Scene contains {len(objects_with_poses)} objects"
        }

    def get_object_by_class(self, class_name: str) -> Optional[Dict[str, Any]]:
        """Get a specific object by class name"""
        for obj in self.detected_objects:
            if obj['class'] == class_name:
                return obj
        return None

    def get_closest_object(self, target_class: str, reference_position: List[float]) -> Optional[Dict[str, Any]]:
        """Get the closest object of a specific class to a reference position"""
        candidates = [obj for obj in self.detected_objects if obj['class'] == target_class]

        if not candidates:
            return None

        # Find closest based on 3D position
        closest = min(candidates, key=lambda obj:
                     sum((a - b)**2 for a, b in zip(obj['position_3d'], reference_position))**0.5)

        return closest
```

### Task Planning Component

```python
class TaskPlanningComponent:
    """
    Task planning component for converting intents to action plans
    """
    def __init__(self, node):
        self.node = node
        self.action_library = self.initialize_action_library()
        self.robot_capabilities = self.get_robot_capabilities()

    def initialize_action_library(self) -> Dict[str, Any]:
        """Initialize library of available actions"""
        return {
            'navigate': {
                'preconditions': ['location_known', 'path_clear'],
                'effects': ['robot_at_location'],
                'implementation': self.implement_navigation
            },
            'grasp': {
                'preconditions': ['object_visible', 'arm_free', 'reachable'],
                'effects': ['object_grasped'],
                'implementation': self.implement_grasp
            },
            'place': {
                'preconditions': ['object_grasped', 'location_reachable'],
                'effects': ['object_placed'],
                'implementation': self.implement_place
            },
            'speak': {
                'preconditions': ['speakers_working'],
                'effects': ['message_delivered'],
                'implementation': self.implement_speak
            },
            'detect': {
                'preconditions': ['camera_working'],
                'effects': ['object_detected'],
                'implementation': self.implement_detect
            }
        }

    def get_robot_capabilities(self) -> Dict[str, Any]:
        """Get robot capabilities for planning"""
        return {
            'navigation': {
                'max_speed': 0.5,
                'step_height': 0.15,
                'turn_speed': 0.5
            },
            'manipulation': {
                'reach_distance': 1.0,
                'max_load': 2.0,
                'grasp_types': ['pinch', 'power', 'hook']
            },
            'sensors': ['camera', 'lidar', 'imu', 'force_torque'],
            'speakers': True
        }

    async def create_plan(self, language_result: Dict[str, Any], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create action plan based on language understanding and context"""
        try:
            intent = language_result.get('intent', 'unknown')
            action_type = language_result.get('action_type', 'unknown')
            entities = language_result.get('entities', {})
            parameters = language_result.get('parameters', {})

            # Create plan based on intent
            if intent == 'navigate':
                plan = await self.create_navigation_plan(entities, parameters, context)
            elif intent == 'manipulate':
                plan = await self.create_manipulation_plan(entities, parameters, context)
            elif intent == 'interact':
                plan = await self.create_interaction_plan(entities, parameters, context)
            else:
                plan = await self.create_generic_plan(language_result, context)

            if plan:
                # Validate plan feasibility
                is_feasible = await self.validate_plan(plan, context)

                if is_feasible:
                    self.node.get_logger().info(f'Task plan created with {len(plan["actions"])} actions')
                    return plan
                else:
                    self.node.get_logger().warn('Generated plan is not feasible')
                    return None
            else:
                self.node.get_logger().warn('No plan could be generated')
                return None

        except Exception as e:
            self.node.get_logger().error(f'Task planning error: {e}')
            return None

    async def create_navigation_plan(self, entities: Dict, parameters: Dict, context: Dict) -> Optional[Dict[str, Any]]:
        """Create navigation plan"""
        target_location = parameters.get('target_location') or entities.get('location')

        if not target_location:
            self.node.get_logger().warn('No target location specified for navigation')
            return None

        # Check if location is known
        if target_location not in context.get('environment_map', {}).get('known_locations', []):
            # Need to explore or ask for more information
            return await self.create_exploration_plan(target_location, context)

        # Create navigation sequence
        actions = [
            {
                'action_type': 'navigate',
                'parameters': {
                    'target_location': target_location
                },
                'description': f'Navigate to {target_location}'
            }
        ]

        return {
            'actions': actions,
            'intent': 'navigate',
            'target': target_location,
            'confidence': 0.9
        }

    async def create_manipulation_plan(self, entities: Dict, parameters: Dict, context: Dict) -> Optional[Dict[str, Any]]:
        """Create manipulation plan"""
        target_object = parameters.get('target_object') or entities.get('object')

        if not target_object:
            self.node.get_logger().warn('No target object specified for manipulation')
            return None

        # Get object from vision data
        vision_data = context.get('vision_data', {})
        objects = vision_data.get('objects', [])

        target_obj_data = None
        for obj in objects:
            if obj['class'] == target_object:
                target_obj_data = obj
                break

        if not target_obj_data:
            # Object not visible, need to navigate somewhere first
            return await self.create_search_plan(target_object, context)

        # Create manipulation sequence
        actions = [
            {
                'action_type': 'navigate',
                'parameters': {
                    'target_location': self.calculate_approach_position(target_obj_data)
                },
                'description': f'Navigate to approach {target_object}'
            },
            {
                'action_type': 'detect',
                'parameters': {
                    'target_object': target_object
                },
                'description': f'Detect {target_object} for precise grasp'
            },
            {
                'action_type': 'grasp',
                'parameters': {
                    'target_object': target_object,
                    'grasp_type': self.select_grasp_type(target_obj_data)
                },
                'description': f'Grasp {target_object}'
            }
        ]

        # Add delivery action if needed (e.g., "bring me a cup")
        if 'bring' in context.get('language_result', {}).get('original_command', '').lower():
            actions.append({
                'action_type': 'navigate',
                'parameters': {
                    'target_location': 'user_position'  # Would be determined from context
                },
                'description': 'Navigate to user position'
            })
            actions.append({
                'action_type': 'place',
                'parameters': {
                    'location': 'user_position'
                },
                'description': 'Place object near user'
            })

        return {
            'actions': actions,
            'intent': 'manipulate',
            'target': target_object,
            'confidence': 0.85
        }

    async def create_interaction_plan(self, entities: Dict, parameters: Dict, context: Dict) -> Optional[Dict[str, Any]]:
        """Create interaction plan"""
        message = parameters.get('message') or entities.get('message')

        if not message:
            self.node.get_logger().warn('No message specified for interaction')
            return None

        actions = [
            {
                'action_type': 'speak',
                'parameters': {
                    'text': message
                },
                'description': f'Speak: {message}'
            }
        ]

        return {
            'actions': actions,
            'intent': 'interact',
            'target': message,
            'confidence': 0.95
        }

    async def create_generic_plan(self, language_result: Dict, context: Dict) -> Optional[Dict[str, Any]]:
        """Create generic plan for unknown intents"""
        original_command = language_result.get('original_command', '')

        # Try to determine intent from command
        if any(word in original_command.lower() for word in ['help', 'what', 'how']):
            # Information request
            return {
                'actions': [{
                    'action_type': 'speak',
                    'parameters': {
                        'text': "I'm not sure how to help with that. Could you please rephrase your request?"
                    },
                    'description': 'Provide help response'
                }],
                'intent': 'unknown',
                'target': 'information',
                'confidence': 0.3
            }
        else:
            # Default response
            return {
                'actions': [{
                    'action_type': 'speak',
                    'parameters': {
                        'text': f"I'm not sure how to {original_command}. Could you please be more specific?"
                    },
                    'description': 'Clarify request'
                }],
                'intent': 'unknown',
                'target': 'clarification',
                'confidence': 0.2
            }

    async def create_exploration_plan(self, target_location: str, context: Dict) -> Optional[Dict[str, Any]]:
        """Create exploration plan when location is unknown"""
        actions = [
            {
                'action_type': 'speak',
                'parameters': {
                    'text': f"I don't know where the {target_location} is. Could you please guide me there?"
                },
                'description': f'Ask for guidance to {target_location}'
            }
        ]

        return {
            'actions': actions,
            'intent': 'explore',
            'target': target_location,
            'confidence': 0.6
        }

    async def create_search_plan(self, target_object: str, context: Dict) -> Optional[Dict[str, Any]]:
        """Create search plan when object is not visible"""
        actions = [
            {
                'action_type': 'speak',
                'parameters': {
                    'text': f"I don't see a {target_object}. Where should I look for it?"
                },
                'description': f'Ask about location of {target_object}'
            }
        ]

        return {
            'actions': actions,
            'intent': 'search',
            'target': target_object,
            'confidence': 0.5
        }

    def calculate_approach_position(self, object_data: Dict) -> List[float]:
        """Calculate approach position for object manipulation"""
        obj_pos = object_data['position_3d']
        # Approach from front (positive Y direction)
        approach_pos = [obj_pos[0], obj_pos[1] + 0.5, obj_pos[2]]  # 50cm in front
        return approach_pos

    def select_grasp_type(self, object_data: Dict) -> str:
        """Select appropriate grasp type based on object properties"""
        obj_class = object_data['class']

        grasp_preferences = {
            'cup': 'pinch',
            'bottle': 'power',
            'book': 'power',
            'phone': 'pinch',
            'keys': 'pinch'
        }

        return grasp_preferences.get(obj_class, 'power')

    async def validate_plan(self, plan: Dict, context: Dict) -> bool:
        """Validate that the plan is feasible given current context"""
        actions = plan.get('actions', [])

        for action in actions:
            action_type = action.get('action_type')
            parameters = action.get('parameters', {})

            # Check if action is supported by robot
            if action_type not in self.action_library:
                self.node.get_logger().warn(f'Action {action_type} not supported')
                return False

            # Check preconditions
            preconditions = self.action_library[action_type].get('preconditions', [])
            if not await self.check_preconditions(action_type, parameters, context):
                self.node.get_logger().warn(f'Preconditions not met for {action_type}')
                return False

        return True

    async def check_preconditions(self, action_type: str, parameters: Dict, context: Dict) -> bool:
        """Check if action preconditions are satisfied"""
        preconditions = self.action_library[action_type].get('preconditions', [])

        for precondition in preconditions:
            if precondition == 'location_known':
                target_loc = parameters.get('target_location')
                if target_loc and target_loc not in context.get('environment_map', {}).get('known_locations', []):
                    return False
            elif precondition == 'object_visible':
                target_obj = parameters.get('target_object')
                if target_obj:
                    vision_data = context.get('vision_data', {})
                    objects = vision_data.get('objects', [])
                    if not any(obj['class'] == target_obj for obj in objects):
                        return False
            elif precondition == 'arm_free':
                # Check if arm is currently holding something
                # This would interface with manipulation system state
                pass
            elif precondition == 'reachable':
                # Check if target is within reach
                # This would use robot kinematics
                pass

        return True

    async def implement_navigation(self, parameters: Dict) -> Dict[str, Any]:
        """Implement navigation action"""
        # This would interface with navigation stack
        return {'success': True, 'message': 'Navigation implemented'}

    async def implement_grasp(self, parameters: Dict) -> Dict[str, Any]:
        """Implement grasp action"""
        # This would interface with manipulation stack
        return {'success': True, 'message': 'Grasp implemented'}

    async def implement_place(self, parameters: Dict) -> Dict[str, Any]:
        """Implement place action"""
        # This would interface with manipulation stack
        return {'success': True, 'message': 'Place implemented'}

    async def implement_speak(self, parameters: Dict) -> Dict[str, Any]:
        """Implement speak action"""
        # This would interface with TTS system
        return {'success': True, 'message': 'Speech implemented'}

    async def implement_detect(self, parameters: Dict) -> Dict[str, Any]:
        """Implement detection action"""
        # This would trigger vision processing
        return {'success': True, 'message': 'Detection implemented'}
```

### Action Execution Component

```python
class ActionExecutionComponent:
    """
    Action execution component for carrying out planned actions
    """
    def __init__(self, node):
        self.node = node
        self.current_action = None
        self.execution_status = 'idle'
        self.joint_states = {}
        self.navigation_client = None
        self.manipulation_client = None
        self.tts_client = None

    async def execute_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a complete action plan"""
        actions = plan.get('actions', [])
        total_actions = len(actions)

        if total_actions == 0:
            return {'success': False, 'error': 'No actions in plan'}

        self.node.get_logger().info(f'Executing plan with {total_actions} actions')

        results = []
        success = True
        error_message = ""

        for i, action in enumerate(actions):
            self.node.get_logger().info(f'Executing action {i+1}/{total_actions}: {action.get("description", "unknown")}')

            # Update execution status
            self.execution_status = f'executing_action_{i+1}'
            self.current_action = action

            # Execute the action
            action_result = await self.execute_single_action(action)

            # Check if action was successful
            if not action_result.get('success', False):
                success = False
                error_message = f"Action {i+1} failed: {action_result.get('error', 'Unknown error')}"
                self.node.get_logger().error(error_message)
                break

            results.append(action_result)

            # Small delay between actions
            await asyncio.sleep(0.5)

        # Update final status
        self.execution_status = 'completed' if success else 'failed'
        self.current_action = None

        return {
            'success': success,
            'results': results,
            'total_actions': total_actions,
            'completed_actions': len(results) if success else len(results) + 1,
            'error': error_message if not success else None
        }

    async def execute_single_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single action"""
        action_type = action.get('action_type', '').lower()
        parameters = action.get('parameters', {})

        try:
            if action_type == 'navigate':
                return await self.execute_navigation_action(parameters)
            elif action_type == 'grasp':
                return await self.execute_grasp_action(parameters)
            elif action_type == 'place':
                return await self.execute_place_action(parameters)
            elif action_type == 'speak':
                return await self.execute_speak_action(parameters)
            elif action_type == 'detect':
                return await self.execute_detect_action(parameters)
            else:
                return {
                    'success': False,
                    'error': f'Unknown action type: {action_type}',
                    'action_type': action_type
                }

        except Exception as e:
            error_msg = f'Error executing {action_type} action: {e}'
            self.node.get_logger().error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'action_type': action_type
            }

    async def execute_navigation_action(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute navigation action"""
        target_location = parameters.get('target_location')

        if not target_location:
            return {'success': False, 'error': 'No target location specified'}

        try:
            # In practice, this would call navigation action server
            # For simulation, we'll simulate navigation

            self.node.get_logger().info(f'Navigating to {target_location}')

            # Simulate navigation execution
            await asyncio.sleep(2.0)  # Simulate navigation time

            # Check if navigation was successful
            # In practice, this would check navigation feedback
            success = True  # Simulate success

            if success:
                return {
                    'success': True,
                    'message': f'Navigated to {target_location} successfully',
                    'action_type': 'navigate',
                    'target_location': target_location
                }
            else:
                return {
                    'success': False,
                    'error': f'Failed to navigate to {target_location}',
                    'action_type': 'navigate',
                    'target_location': target_location
                }

        except Exception as e:
            return {
                'success': False,
                'error': f'Navigation error: {e}',
                'action_type': 'navigate',
                'target_location': target_location
            }

    async def execute_grasp_action(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute grasp action"""
        target_object = parameters.get('target_object')
        grasp_type = parameters.get('grasp_type', 'power')

        if not target_object:
            return {'success': False, 'error': 'No target object specified'}

        try:
            self.node.get_logger().info(f'Attempting to grasp {target_object} with {grasp_type} grasp')

            # Simulate grasp execution
            await asyncio.sleep(1.5)  # Simulate grasp time

            # Simulate success/failure based on object properties
            import random
            success = random.random() > 0.1  # 90% success rate for simulation

            if success:
                return {
                    'success': True,
                    'message': f'Successfully grasped {target_object}',
                    'action_type': 'grasp',
                    'target_object': target_object,
                    'grasp_type': grasp_type
                }
            else:
                return {
                    'success': False,
                    'error': f'Failed to grasp {target_object}',
                    'action_type': 'grasp',
                    'target_object': target_object,
                    'grasp_type': grasp_type
                }

        except Exception as e:
            return {
                'success': False,
                'error': f'Grasp error: {e}',
                'action_type': 'grasp',
                'target_object': target_object,
                'grasp_type': grasp_type
            }

    async def execute_place_action(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute place action"""
        location = parameters.get('location', 'default')

        try:
            self.node.get_logger().info(f'Attempting to place object at {location}')

            # Simulate place execution
            await asyncio.sleep(1.0)  # Simulate place time

            # Simulate success
            return {
                'success': True,
                'message': f'Successfully placed object at {location}',
                'action_type': 'place',
                'location': location
            }

        except Exception as e:
            return {
                'success': False,
                'error': f'Place error: {e}',
                'action_type': 'place',
                'location': location
            }

    async def execute_speak_action(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute speak action"""
        text = parameters.get('text', '')

        if not text:
            return {'success': False, 'error': 'No text to speak'}

        try:
            self.node.get_logger().info(f'Speaking: {text}')

            # In practice, this would call TTS service
            # For simulation, we'll just log the text

            # Simulate speech time based on text length
            speech_time = len(text.split()) * 0.2  # ~0.2 seconds per word
            await asyncio.sleep(speech_time)

            return {
                'success': True,
                'message': f'Spoke: {text}',
                'action_type': 'speak',
                'text': text
            }

        except Exception as e:
            return {
                'success': False,
                'error': f'Speech error: {e}',
                'action_type': 'speak',
                'text': text
            }

    async def execute_detect_action(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute detect action"""
        target_object = parameters.get('target_object')

        try:
            self.node.get_logger().info(f'Detecting {target_object if target_object else "scene"}')

            # In practice, this would trigger vision processing
            # For simulation, we'll return mock detection result

            await asyncio.sleep(0.5)  # Simulate detection time

            return {
                'success': True,
                'message': f'Detected {target_object if target_object else "environment"}',
                'action_type': 'detect',
                'target_object': target_object
            }

        except Exception as e:
            return {
                'success': False,
                'error': f'Detection error: {e}',
                'action_type': 'detect',
                'target_object': target_object
            }

    def update_joint_states(self, joint_states: Dict[str, float]):
        """Update with current joint states"""
        self.joint_states.update(joint_states)

    async def get_robot_state(self) -> Dict[str, Any]:
        """Get current robot state"""
        return {
            'execution_status': self.execution_status,
            'current_action': self.current_action,
            'joint_states': self.joint_states.copy(),
            'navigation_status': 'idle',  # Would come from navigation system
            'manipulation_status': 'idle'  # Would come from manipulation system
        }
```

### Pipeline Optimization and Error Handling

#### Advanced Error Recovery
```python
class PipelineErrorRecovery:
    """
    Advanced error recovery for the VLA pipeline
    """
    def __init__(self, pipeline_node):
        self.pipeline = pipeline_node
        self.recovery_strategies = self.define_recovery_strategies()
        self.error_history = []

    def define_recovery_strategies(self) -> Dict[str, callable]:
        """Define various recovery strategies"""
        return {
            'retry': self.retry_strategy,
            'simplify': self.simplify_command_strategy,
            'ask_for_help': self.ask_for_help_strategy,
            'alternative_action': self.alternative_action_strategy,
            'degrade_gracefully': self.degrade_gracefully_strategy
        }

    async def handle_pipeline_error(self, error_context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle pipeline error with appropriate recovery"""
        error_type = error_context.get('error_type', 'unknown')
        error_stage = error_context.get('stage', 'unknown')
        command = error_context.get('command', '')

        self.log_error(error_context)

        # Determine appropriate recovery strategy
        recovery_strategy = self.select_recovery_strategy(error_type, error_stage, command)

        if recovery_strategy:
            return await recovery_strategy(error_context)
        else:
            return await self.default_recovery(error_context)

    def select_recovery_strategy(self, error_type: str, stage: str, command: str) -> callable:
        """Select appropriate recovery strategy based on error context"""
        if error_type == 'speech_recognition_failed':
            return self.recovery_strategies['ask_for_help']
        elif error_type == 'object_not_found' and stage == 'manipulation':
            return self.recovery_strategies['ask_for_help']
        elif error_type == 'navigation_failed':
            return self.recovery_strategies['alternative_action']
        elif error_type == 'grasp_failed':
            return self.recovery_strategies['retry']
        else:
            return self.recovery_strategies['degrade_gracefully']

    async def retry_strategy(self, error_context: Dict[str, Any]) -> Dict[str, Any]:
        """Retry the failed action"""
        action = error_context.get('failed_action')
        max_retries = error_context.get('max_retries', 3)
        retries = error_context.get('retries', 0)

        if retries < max_retries:
            # Retry the action
            await asyncio.sleep(1.0)  # Brief delay before retry
            return await self.pipeline.execute_single_action(action)
        else:
            return {'success': False, 'error': 'Max retries exceeded'}

    async def simplify_command_strategy(self, error_context: Dict[str, Any]) -> Dict[str, Any]:
        """Simplify the command and try again"""
        original_command = error_context.get('command', '')

        # Simplify command by removing complex parts
        simplified_command = self.simplify_command(original_command)

        if simplified_command != original_command:
            self.pipeline.get_logger().info(f'Simplified command from "{original_command}" to "{simplified_command}"')
            # Process the simplified command
            return await self.pipeline.process_voice_command(simplified_command, time.time())
        else:
            return {'success': False, 'error': 'Command could not be simplified'}

    def simplify_command(self, command: str) -> str:
        """Simplify a complex command"""
        # Remove qualifiers and complex modifiers
        import re

        # Remove spatial qualifiers like "the one on the left"
        command = re.sub(r'the one on the \w+', '', command)

        # Remove temporal qualifiers like "the one I showed you earlier"
        command = re.sub(r'the one \w+ \w+', '', command)

        # Keep only the core action and object
        core_elements = []
        if 'bring' in command or 'get' in command:
            core_elements.append('bring')
        elif 'go' in command or 'move' in command:
            core_elements.append('go')
        elif 'pick' in command or 'grasp' in command:
            core_elements.append('pick')

        # Extract main object
        objects = ['cup', 'bottle', 'book', 'phone', 'keys', 'water', 'coffee']
        for obj in objects:
            if obj in command.lower():
                core_elements.append(obj)
                break

        return ' '.join(core_elements) if core_elements else command

    async def ask_for_help_strategy(self, error_context: Dict[str, Any]) -> Dict[str, Any]:
        """Ask user for help or clarification"""
        error_stage = error_context.get('stage', 'unknown')
        original_command = error_context.get('command', '')

        if error_stage == 'object_not_found':
            help_message = f"I couldn't find the object you mentioned in '{original_command}'. Could you please point it out or give me more details?"
        elif error_stage == 'location_unknown':
            help_message = f"I don't know where you mean by '{original_command}'. Could you guide me there?"
        else:
            help_message = f"I'm not sure how to '{original_command}'. Could you please rephrase or be more specific?"

        # Speak the help request
        help_result = await self.pipeline.action_executor.execute_speak_action({'text': help_message})

        return {
            'success': True,
            'message': help_message,
            'action_type': 'request_clarification',
            'recovery_strategy': 'ask_for_help'
        }

    async def alternative_action_strategy(self, error_context: Dict[str, Any]) -> Dict[str, Any]:
        """Try an alternative action"""
        original_action = error_context.get('failed_action', {})
        action_type = original_action.get('action_type', 'unknown')

        if action_type == 'navigate':
            # Try alternative navigation approach
            return await self.try_alternative_navigation(original_action)
        elif action_type == 'grasp':
            # Try alternative grasp type
            return await self.try_alternative_grasp(original_action)
        else:
            return {'success': False, 'error': 'No alternative action available'}

    async def try_alternative_navigation(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Try alternative navigation approach"""
        # In practice, this would try different navigation strategies
        # For example: different planner, different path, or ask for guidance
        target = action['parameters'].get('target_location', 'unknown')

        alternative_msg = f"Having trouble navigating to {target}. Let me try a different approach."
        await self.pipeline.action_executor.execute_speak_action({'text': alternative_msg})

        # Simulate trying alternative navigation
        await asyncio.sleep(1.0)

        # Return success for simulation
        return {
            'success': True,
            'message': f'Used alternative navigation to reach {target}',
            'action_type': 'navigate',
            'strategy': 'alternative_navigation'
        }

    async def try_alternative_grasp(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Try alternative grasp approach"""
        target_obj = action['parameters'].get('target_object', 'unknown')
        current_grasp = action['parameters'].get('grasp_type', 'power')

        # Try different grasp type
        alternative_grasp = 'pinch' if current_grasp == 'power' else 'power'

        alternative_msg = f"Trying alternative grasp for {target_obj}."
        await self.pipeline.action_executor.execute_speak_action({'text': alternative_msg})

        # Execute alternative grasp
        new_action = action.copy()
        new_action['parameters']['grasp_type'] = alternative_grasp

        return await self.pipeline.action_executor.execute_single_action(new_action)

    async def degrade_gracefully_strategy(self, error_context: Dict[str, Any]) -> Dict[str, Any]:
        """Degrade gracefully by providing partial functionality"""
        original_command = error_context.get('command', '')

        graceful_msg = f"I'm having trouble with '{original_command}', but I can still help with other tasks. Is there something else I can do for you?"

        await self.pipeline.action_executor.execute_speak_action({'text': graceful_msg})

        return {
            'success': True,
            'message': graceful_msg,
            'action_type': 'graceful_degradation',
            'recovery_strategy': 'degrade_gracefully'
        }

    async def default_recovery(self, error_context: Dict[str, Any]) -> Dict[str, Any]:
        """Default recovery when no specific strategy applies"""
        error_msg = error_context.get('error', 'Unknown error')

        default_response = f"I encountered an issue: {error_msg}. I'll need to restart this task."

        await self.pipeline.action_executor.execute_speak_action({'text': default_response})

        return {
            'success': False,
            'error': error_msg,
            'message': default_response,
            'recovery_strategy': 'default'
        }

    def log_error(self, error_context: Dict[str, Any]):
        """Log error for analysis and improvement"""
        error_entry = {
            'timestamp': time.time(),
            'error_context': error_context,
            'recovery_attempted': True
        }
        self.error_history.append(error_entry)

        # Keep only recent errors
        if len(self.error_history) > 100:
            self.error_history = self.error_history[-100:]
```

### Performance Monitoring and Optimization

```python
class PipelinePerformanceMonitor:
    """
    Performance monitoring for the VLA pipeline
    """
    def __init__(self, pipeline_node):
        self.pipeline = pipeline_node
        self.metrics = {
            'command_processing_times': [],
            'success_rates': [],
            'error_types': {},
            'component_latencies': {
                'speech_recognition': [],
                'nlp_processing': [],
                'vision_processing': [],
                'task_planning': [],
                'action_execution': []
            }
        }
        self.performance_window = 100  # Track last 100 commands

    def record_command_processing_time(self, processing_time: float):
        """Record command processing time"""
        self.metrics['command_processing_times'].append(processing_time)
        if len(self.metrics['command_processing_times']) > self.performance_window:
            self.metrics['command_processing_times'].pop(0)

    def record_success_rate(self, success: bool):
        """Record success/failure of command processing"""
        self.metrics['success_rates'].append(success)
        if len(self.metrics['success_rates']) > self.performance_window:
            self.metrics['success_rates'].pop(0)

    def record_error_type(self, error_type: str):
        """Record error type for analysis"""
        if error_type in self.metrics['error_types']:
            self.metrics['error_types'][error_type] += 1
        else:
            self.metrics['error_types'][error_type] = 1

    def record_component_latency(self, component: str, latency: float):
        """Record latency for a specific component"""
        if component in self.metrics['component_latencies']:
            self.metrics['component_latencies'][component].append(latency)
            if len(self.metrics['component_latencies'][component]) > self.performance_window:
                self.metrics['component_latencies'][component].pop(0)

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        import statistics

        report = {
            'overall_metrics': self.calculate_overall_metrics(),
            'component_analysis': self.calculate_component_analysis(),
            'trend_analysis': self.calculate_trend_analysis()
        }

        return report

    def calculate_overall_metrics(self) -> Dict[str, Any]:
        """Calculate overall performance metrics"""
        proc_times = self.metrics['command_processing_times']
        success_rates = self.metrics['success_rates']

        return {
            'average_processing_time': statistics.mean(proc_times) if proc_times else 0,
            'median_processing_time': statistics.median(proc_times) if proc_times else 0,
            'max_processing_time': max(proc_times) if proc_times else 0,
            'min_processing_time': min(proc_times) if proc_times else 0,
            'stddev_processing_time': statistics.stdev(proc_times) if len(proc_times) > 1 else 0,
            'success_rate': sum(success_rates) / len(success_rates) if success_rates else 0,
            'total_commands_processed': len(success_rates),
            'error_distribution': self.metrics['error_types']
        }

    def calculate_component_analysis(self) -> Dict[str, Any]:
        """Calculate component-specific performance analysis"""
        analysis = {}

        for component, latencies in self.metrics['component_latencies'].items():
            if latencies:
                analysis[component] = {
                    'average_latency': statistics.mean(latencies),
                    'median_latency': statistics.median(latencies),
                    'max_latency': max(latencies),
                    'min_latency': min(latencies),
                    'stddev_latency': statistics.stdev(latencies) if len(latencies) > 1 else 0,
                    'sample_count': len(latencies)
                }

        return analysis

    def calculate_trend_analysis(self) -> Dict[str, Any]:
        """Calculate performance trend analysis"""
        proc_times = self.metrics['command_processing_times']

        if len(proc_times) < 10:
            return {'trend': 'insufficient_data', 'recommendations': []}

        # Compare first and last 25% of data
        split_point = len(proc_times) // 4
        first_quarter = proc_times[:split_point]
        last_quarter = proc_times[-split_point:]

        if not first_quarter or not last_quarter:
            return {'trend': 'insufficient_data', 'recommendations': []}

        avg_first = statistics.mean(first_quarter)
        avg_last = statistics.mean(last_quarter)

        if avg_last > avg_first * 1.1:  # Degraded by more than 10%
            trend = 'degrading'
            recommendations = [
                'Consider optimizing slowest components',
                'Check for resource constraints',
                'Review recent code changes'
            ]
        elif avg_last < avg_first * 0.9:  # Improved by more than 10%
            trend = 'improving'
            recommendations = [
                'Maintain current optimizations',
                'Consider applying similar optimizations elsewhere'
            ]
        else:
            trend = 'stable'
            recommendations = [
                'Continue monitoring',
                'Look for optimization opportunities'
            ]

        return {
            'trend': trend,
            'first_quarter_avg': avg_first,
            'last_quarter_avg': avg_last,
            'change_percentage': ((avg_last - avg_first) / avg_first) * 100 if avg_first != 0 else 0,
            'recommendations': recommendations
        }

    def get_performance_alerts(self) -> List[str]:
        """Get performance alerts based on metrics"""
        alerts = []

        # Check processing time
        proc_times = self.metrics['command_processing_times']
        if proc_times:
            avg_time = statistics.mean(proc_times)
            if avg_time > 5.0:  # More than 5 seconds average
                alerts.append(f'High average processing time: {avg_time:.2f}s')

        # Check success rate
        success_rates = self.metrics['success_rates']
        if success_rates:
            success_rate = sum(success_rates) / len(success_rates)
            if success_rate < 0.7:  # Less than 70% success rate
                alerts.append(f'Low success rate: {success_rate:.2%}')

        # Check for frequent errors
        error_counts = self.metrics['error_types']
        total_errors = sum(error_counts.values())
        if total_errors > 10:  # More than 10 errors
            most_common_error = max(error_counts, key=error_counts.get)
            alerts.append(f'Most common error: {most_common_error} ({error_counts[most_common_error]} occurrences)')

        return alerts
```

### Summary

In this chapter, we've implemented the complete Voice-to-Action pipeline for humanoid robots:

- **Pipeline Architecture**: Complete system architecture connecting voice input to action execution
- **Speech Recognition**: Component for converting speech to text
- **Natural Language Processing**: Component for understanding commands and extracting intent
- **Vision Processing**: Component for scene understanding and object detection
- **Task Planning**: Component for converting high-level goals to executable actions
- **Action Execution**: Component for carrying out planned actions
- **Error Recovery**: Advanced error handling and recovery mechanisms
- **Performance Monitoring**: Comprehensive system performance tracking

This pipeline represents the core of a humanoid robot's ability to understand and respond to natural language commands. In the next chapter, we'll explore autonomous navigation and manipulation behaviors that build upon this foundation.