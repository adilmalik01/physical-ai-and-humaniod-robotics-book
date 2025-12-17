---
sidebar_position: 1
---

# Chapter 01: System Integration

## Combining All Components into a Unified Architecture

In this first chapter of our capstone project, we'll focus on system integration - the critical task of combining all the individual components we've developed throughout the book into a unified, cohesive humanoid robot system. This chapter covers the architecture, integration patterns, and best practices for creating a complete system.

### Understanding System Integration Challenges

Integrating a complete humanoid robot system presents unique challenges:

#### Component Interoperability
- **Different development timelines**: Components developed separately may have different interfaces
- **Communication protocols**: Ensuring all components can communicate effectively
- **Data formats**: Standardizing data formats across components
- **Timing constraints**: Managing real-time requirements across different components

#### Performance Considerations
- **Resource sharing**: Multiple components competing for CPU, GPU, and memory
- **Latency management**: Ensuring end-to-end latency requirements are met
- **Throughput balancing**: Balancing the processing rates of different components

#### Safety and Reliability
- **Fail-safe mechanisms**: Ensuring system remains safe when components fail
- **Error propagation**: Preventing errors from cascading through the system
- **Monitoring and recovery**: Implementing comprehensive monitoring and recovery mechanisms

### System Architecture Overview

The integrated humanoid robot system architecture consists of several layers:

```
┌─────────────────────────────────────────────────────────────┐
│                      Application Layer                       │
│  (Voice Commands, Task Planning, High-level Behaviors)     │
├─────────────────────────────────────────────────────────────┤
│                     AI/ML Layer                              │
│  (LLM Planning, Perception, Navigation Planning)           │
├─────────────────────────────────────────────────────────────┤
│                   Control Layer                              │
│  (Motion Control, Manipulation Control, Navigation Control) │
├─────────────────────────────────────────────────────────────┤
│                   Perception Layer                           │
│  (Vision, Audio, Sensory Processing)                       │
├─────────────────────────────────────────────────────────────┤
│                    Hardware Layer                            │
│  (Sensors, Actuators, Computing Platform)                  │
└─────────────────────────────────────────────────────────────┘
```

### Integration Patterns

#### 1. Message-Based Integration
Using ROS 2 topics and services for component communication:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import PoseStamped
from humanoid_robot_msgs.msg import SystemState, TaskCommand
from humanoid_robot_msgs.srv import ExecuteTask, QueryState

class IntegratedSystemNode(Node):
    """
    Main integration node that coordinates all system components
    """
    def __init__(self):
        super().__init__('integrated_system')

        # Publishers for system state and commands
        self.system_state_pub = self.create_publisher(SystemState, '/system/state', 10)
        self.task_command_pub = self.create_publisher(TaskCommand, '/system/task_command', 10)

        # Publishers for individual components
        self.voice_command_pub = self.create_publisher(String, '/voice_command', 10)
        self.navigation_goal_pub = self.create_publisher(PoseStamped, '/navigation/goal', 10)
        self.manipulation_command_pub = self.create_publisher(String, '/manipulation/command', 10)

        # Subscribers for component status
        self.voice_status_sub = self.create_subscription(String, '/voice/status', self.voice_status_callback, 10)
        self.vision_status_sub = self.create_subscription(String, '/vision/status', self.vision_status_callback, 10)
        self.navigation_status_sub = self.create_subscription(String, '/navigation/status', self.navigation_status_callback, 10)
        self.manipulation_status_sub = self.create_subscription(String, '/manipulation/status', self.manipulation_status_callback, 10)

        # Services for system-wide operations
        self.execute_task_srv = self.create_service(ExecuteTask, '/system/execute_task', self.execute_task_callback)
        self.query_state_srv = self.create_service(QueryState, '/system/query_state', self.query_state_callback)

        # System state tracking
        self.component_statuses = {
            'voice': 'unknown',
            'vision': 'unknown',
            'navigation': 'unknown',
            'manipulation': 'unknown'
        }

        # Task execution queue
        self.task_queue = []
        self.current_task = None

        self.get_logger().info('Integrated System Node initialized')

    def voice_status_callback(self, msg):
        """Handle voice component status updates"""
        self.component_statuses['voice'] = msg.data
        self.publish_system_state()

    def vision_status_callback(self, msg):
        """Handle vision component status updates"""
        self.component_statuses['vision'] = msg.data
        self.publish_system_state()

    def navigation_status_callback(self, msg):
        """Handle navigation component status updates"""
        self.component_statuses['navigation'] = msg.data
        self.publish_system_state()

    def manipulation_status_callback(self, msg):
        """Handle manipulation component status updates"""
        self.component_statuses['manipulation'] = msg.data
        self.publish_system_state()

    def publish_system_state(self):
        """Publish overall system state"""
        state_msg = SystemState()
        state_msg.timestamp = self.get_clock().now().to_msg()
        state_msg.voice_status = self.component_statuses['voice']
        state_msg.vision_status = self.component_statuses['vision']
        state_msg.navigation_status = self.component_statuses['navigation']
        state_msg.manipulation_status = self.component_statuses['manipulation']

        self.system_state_pub.publish(state_msg)

    def execute_task_callback(self, request, response):
        """Handle task execution requests"""
        self.get_logger().info(f'Executing task: {request.task_description}')

        # Add task to queue
        self.task_queue.append(request)

        # Process tasks if none currently executing
        if self.current_task is None:
            self.process_next_task()

        response.success = True
        response.message = f'Task queued: {request.task_description}'
        return response

    def query_state_callback(self, request, response):
        """Handle system state queries"""
        response.state = SystemState()
        response.state.voice_status = self.component_statuses['voice']
        response.state.vision_status = self.component_statuses['vision']
        response.state.navigation_status = self.component_statuses['navigation']
        response.state.manipulation_status = self.component_statuses['manipulation']

        response.success = True
        return response

    def process_next_task(self):
        """Process the next task in the queue"""
        if self.task_queue:
            self.current_task = self.task_queue.pop(0)

            # Determine task type and route appropriately
            task_type = self.current_task.task_type
            if task_type == 'navigation':
                self.route_navigation_task()
            elif task_type == 'manipulation':
                self.route_manipulation_task()
            elif task_type == 'interactive':
                self.route_interactive_task()
            else:
                self.route_generic_task()

    def route_navigation_task(self):
        """Route navigation tasks to navigation component"""
        goal = PoseStamped()
        # Extract goal from task
        goal.header.frame_id = 'map'
        goal.pose.position.x = self.current_task.parameters.get('x', 0.0)
        goal.pose.position.y = self.current_task.parameters.get('y', 0.0)
        goal.pose.orientation.w = 1.0

        self.navigation_goal_pub.publish(goal)

    def route_manipulation_task(self):
        """Route manipulation tasks to manipulation component"""
        command = String()
        command.data = self.current_task.task_description
        self.manipulation_command_pub.publish(command)

    def route_interactive_task(self):
        """Route interactive tasks to appropriate components"""
        # Interactive tasks typically involve multiple components
        # Process voice command first
        voice_cmd = String()
        voice_cmd.data = self.current_task.task_description
        self.voice_command_pub.publish(voice_cmd)

    def route_generic_task(self):
        """Route generic tasks"""
        command = TaskCommand()
        command.description = self.current_task.task_description
        command.parameters = self.current_task.parameters
        self.task_command_pub.publish(command)

def main(args=None):
    rclpy.init(args=args)

    system_node = IntegratedSystemNode()

    try:
        rclpy.spin(system_node)
    except KeyboardInterrupt:
        pass
    finally:
        system_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### 2. Service-Based Integration
Using ROS 2 services for synchronous operations:

```python
class SystemIntegrationServices:
    """
    Service-based integration for synchronous operations
    """
    def __init__(self, node):
        self.node = node
        self.services = {}

        # Create system-wide services
        self.create_services()

    def create_services(self):
        """Create all system services"""
        self.services['execute_task'] = self.node.create_service(
            ExecuteTask,
            '/system/execute_task',
            self.handle_execute_task
        )

        self.services['query_state'] = self.node.create_service(
            QueryState,
            '/system/query_state',
            self.handle_query_state
        )

        self.services['abort_task'] = self.node.create_service(
            AbortTask,
            '/system/abort_task',
            self.handle_abort_task
        )

        self.services['calibrate_system'] = self.node.create_service(
            CalibrateSystem,
            '/system/calibrate',
            self.handle_calibrate_system
        )

    def handle_execute_task(self, request, response):
        """Handle task execution request with full system coordination"""
        try:
            # Validate task request
            if not self.validate_task_request(request):
                response.success = False
                response.message = "Invalid task request"
                return response

            # Coordinate components for task execution
            coordination_result = self.coordinate_task_execution(request)

            response.success = coordination_result['success']
            response.message = coordination_result['message']

            return response

        except Exception as e:
            self.node.get_logger().error(f'Error executing task: {e}')
            response.success = False
            response.message = f'Error executing task: {e}'
            return response

    def validate_task_request(self, request):
        """Validate task request before execution"""
        # Check if all required components are available
        required_components = self.get_required_components(request.task_type)

        for component in required_components:
            if not self.is_component_available(component):
                return False

        return True

    def coordinate_task_execution(self, request):
        """Coordinate multiple components for task execution"""
        # This is where the magic happens - coordinating all components
        # 1. Prepare components
        preparation_result = self.prepare_components(request)
        if not preparation_result['success']:
            return preparation_result

        # 2. Execute task with coordinated components
        execution_result = self.execute_coordinated_task(request)

        # 3. Cleanup components
        cleanup_result = self.cleanup_components()

        return {
            'success': execution_result['success'] and cleanup_result['success'],
            'message': f"Task execution: {execution_result['message']}, Cleanup: {cleanup_result['message']}"
        }

    def prepare_components(self, request):
        """Prepare all required components for task execution"""
        # Activate required components
        required_components = self.get_required_components(request.task_type)

        for component in required_components:
            activation_result = self.activate_component(component)
            if not activation_result['success']:
                return {
                    'success': False,
                    'message': f"Failed to activate {component}: {activation_result['message']}"
                }

        return {
            'success': True,
            'message': 'All components prepared successfully'
        }

    def execute_coordinated_task(self, request):
        """Execute task with coordinated components"""
        # This would implement the specific coordination logic
        # based on the task type
        task_type = request.task_type

        if task_type == 'navigation_with_interaction':
            return self.execute_navigation_with_interaction(request)
        elif task_type == 'manipulation_with_vision':
            return self.execute_manipulation_with_vision(request)
        elif task_type == 'speech_with_navigation':
            return self.execute_speech_with_navigation(request)
        else:
            return {
                'success': False,
                'message': f'Unknown task type: {task_type}'
            }

    def execute_navigation_with_interaction(self, request):
        """Execute navigation task with interaction capabilities"""
        # 1. Plan navigation path
        nav_result = self.plan_and_execute_navigation(request.navigation_params)

        # 2. While navigating, monitor for interaction opportunities
        interaction_result = self.monitor_interaction_opportunities()

        return {
            'success': nav_result['success'] and interaction_result['success'],
            'message': f"Navigation: {nav_result['message']}, Interaction: {interaction_result['message']}"
        }

    def get_required_components(self, task_type):
        """Get required components for a specific task type"""
        component_requirements = {
            'navigation': ['navigation', 'sensors', 'localization'],
            'manipulation': ['manipulation', 'vision', 'grasping'],
            'interaction': ['speech_recognition', 'tts', 'vision'],
            'navigation_with_interaction': ['navigation', 'sensors', 'localization', 'speech_recognition', 'tts'],
            'manipulation_with_vision': ['manipulation', 'vision', 'grasping'],
            'speech_with_navigation': ['speech_recognition', 'navigation', 'sensors', 'localization']
        }

        return component_requirements.get(task_type, [])

    def is_component_available(self, component_name):
        """Check if a component is available"""
        # In practice, this would check component status
        return True  # Simplified for example

    def activate_component(self, component_name):
        """Activate a specific component"""
        # In practice, this would send activation commands to the component
        return {
            'success': True,
            'message': f'{component_name} activated'
        }

    def cleanup_components(self):
        """Cleanup components after task execution"""
        # Deactivate components that were activated
        return {
            'success': True,
            'message': 'Components cleaned up successfully'
        }
```

#### 3. Action-Based Integration
Using ROS 2 actions for long-running operations:

```python
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from humanoid_robot_msgs.action import ExecuteComplexTask

class SystemActionServer:
    """
    Action server for complex, long-running tasks
    """
    def __init__(self, node):
        self.node = node
        self.action_server = ActionServer(
            node,
            ExecuteComplexTask,
            'execute_complex_task',
            self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )

        self.current_goals = {}

    def goal_callback(self, goal_request):
        """Handle goal requests"""
        self.node.get_logger().info(f'Received complex task goal: {goal_request.task_description}')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Handle goal cancellations"""
        self.node.get_logger().info(f'Cancelling complex task: {goal_handle.goal_id}')
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        """Execute complex task with full system integration"""
        self.node.get_logger().info('Starting complex task execution')

        task_request = goal_handle.request
        feedback_msg = ExecuteComplexTask.Feedback()
        result = ExecuteComplexTask.Result()

        try:
            # Initialize feedback
            feedback_msg.current_phase = "INITIALIZING"
            feedback_msg.progress = 0.0
            goal_handle.publish_feedback(feedback_msg)

            # Step 1: Analyze task requirements
            feedback_msg.current_phase = "ANALYZING_TASK"
            feedback_msg.progress = 0.1
            goal_handle.publish_feedback(feedback_msg)

            task_analysis = await self.analyze_task_requirements(task_request)
            if not task_analysis['success']:
                result.success = False
                result.message = task_analysis['message']
                goal_handle.succeed()
                return result

            # Step 2: Coordinate system components
            feedback_msg.current_phase = "COORDINATING_COMPONENTS"
            feedback_msg.progress = 0.2
            goal_handle.publish_feedback(feedback_msg)

            coordination_result = await self.coordinate_system_components(task_request, task_analysis)
            if not coordination_result['success']:
                result.success = False
                result.message = coordination_result['message']
                goal_handle.succeed()
                return result

            # Step 3: Execute task phases
            total_phases = len(task_analysis['phases'])
            for i, phase in enumerate(task_analysis['phases']):
                feedback_msg.current_phase = f"EXECUTING_PHASE_{phase['name']}"
                feedback_msg.progress = 0.2 + (0.7 * (i + 1) / total_phases)
                goal_handle.publish_feedback(feedback_msg)

                phase_result = await self.execute_task_phase(phase, task_request)

                if not phase_result['success']:
                    result.success = False
                    result.message = f"Phase {phase['name']} failed: {phase_result['message']}"
                    goal_handle.succeed()
                    return result

            # Step 4: Finalize and cleanup
            feedback_msg.current_phase = "FINALIZING"
            feedback_msg.progress = 0.95
            goal_handle.publish_feedback(feedback_msg)

            finalize_result = await self.finalize_task_execution(task_request)

            feedback_msg.current_phase = "COMPLETED"
            feedback_msg.progress = 1.0
            goal_handle.publish_feedback(feedback_msg)

            result.success = finalize_result['success']
            result.message = finalize_result['message']

        except Exception as e:
            self.node.get_logger().error(f'Error in complex task execution: {e}')
            result.success = False
            result.message = f"Execution error: {e}"

        goal_handle.succeed()
        return result

    async def analyze_task_requirements(self, task_request):
        """Analyze task requirements and determine execution plan"""
        # Analyze the complex task and break it down into phases
        try:
            # Example task analysis
            if 'navigate' in task_request.task_description.lower():
                phases = [
                    {'name': 'navigation_setup', 'type': 'navigation', 'params': {}},
                    {'name': 'path_planning', 'type': 'planning', 'params': {}},
                    {'name': 'navigation_execution', 'type': 'navigation', 'params': {}}
                ]
            elif 'manipulate' in task_request.task_description.lower():
                phases = [
                    {'name': 'object_detection', 'type': 'vision', 'params': {}},
                    {'name': 'grasp_planning', 'type': 'planning', 'params': {}},
                    {'name': 'manipulation_execution', 'type': 'manipulation', 'params': {}}
                ]
            else:
                phases = [
                    {'name': 'task_breakdown', 'type': 'planning', 'params': {}},
                    {'name': 'component_coordination', 'type': 'coordination', 'params': {}},
                    {'name': 'execution', 'type': 'execution', 'params': {}}
                ]

            return {
                'success': True,
                'phases': phases,
                'required_components': self.get_required_components_for_task(task_request)
            }

        except Exception as e:
            return {
                'success': False,
                'message': f'Task analysis failed: {e}'
            }

    async def coordinate_system_components(self, task_request, task_analysis):
        """Coordinate system components for task execution"""
        try:
            required_components = task_analysis['required_components']

            # Activate required components
            for component in required_components:
                activation_result = await self.activate_component_async(component)
                if not activation_result['success']:
                    return {
                        'success': False,
                        'message': f'Failed to activate {component}: {activation_result["message"]}'
                    }

            return {
                'success': True,
                'message': 'All components coordinated successfully'
            }

        except Exception as e:
            return {
                'success': False,
                'message': f'Component coordination failed: {e}'
            }

    async def execute_task_phase(self, phase, task_request):
        """Execute a single task phase"""
        try:
            # Execute phase based on type
            if phase['type'] == 'navigation':
                return await self.execute_navigation_phase(phase, task_request)
            elif phase['type'] == 'manipulation':
                return await self.execute_manipulation_phase(phase, task_request)
            elif phase['type'] == 'vision':
                return await self.execute_vision_phase(phase, task_request)
            elif phase['type'] == 'planning':
                return await self.execute_planning_phase(phase, task_request)
            else:
                return await self.execute_generic_phase(phase, task_request)

        except Exception as e:
            return {
                'success': False,
                'message': f'Phase execution failed: {e}'
            }

    async def execute_navigation_phase(self, phase, task_request):
        """Execute navigation phase"""
        # In practice, this would interface with navigation system
        return {'success': True, 'message': 'Navigation phase completed'}

    async def execute_manipulation_phase(self, phase, task_request):
        """Execute manipulation phase"""
        # In practice, this would interface with manipulation system
        return {'success': True, 'message': 'Manipulation phase completed'}

    async def execute_vision_phase(self, phase, task_request):
        """Execute vision phase"""
        # In practice, this would interface with vision system
        return {'success': True, 'message': 'Vision phase completed'}

    async def execute_planning_phase(self, phase, task_request):
        """Execute planning phase"""
        # In practice, this would interface with planning system
        return {'success': True, 'message': 'Planning phase completed'}

    async def execute_generic_phase(self, phase, task_request):
        """Execute generic phase"""
        # In practice, this would handle generic operations
        return {'success': True, 'message': 'Generic phase completed'}

    async def finalize_task_execution(self, task_request):
        """Finalize task execution and cleanup"""
        try:
            # Cleanup and deactivate components
            # In practice, this would handle cleanup operations

            return {
                'success': True,
                'message': 'Task execution completed successfully'
            }

        except Exception as e:
            return {
                'success': False,
                'message': f'Finalization failed: {e}'
            }

    def get_required_components_for_task(self, task_request):
        """Get required components for a specific task"""
        # Analyze task description to determine required components
        desc = task_request.task_description.lower()
        components = []

        if 'navigate' in desc or 'move' in desc or 'go' in desc:
            components.extend(['navigation', 'localization', 'sensors'])

        if 'grasp' in desc or 'pick' in desc or 'manipulate' in desc:
            components.extend(['manipulation', 'vision', 'grippers'])

        if 'see' in desc or 'find' in desc or 'detect' in desc:
            components.extend(['vision', 'object_detection'])

        if 'speak' in desc or 'talk' in desc or 'say' in desc:
            components.extend(['speech_synthesis', 'audio'])

        return list(set(components))  # Remove duplicates

    async def activate_component_async(self, component):
        """Asynchronously activate a component"""
        # In practice, this would send activation commands
        return {'success': True, 'message': f'{component} activated'}
```

### Component Coordination Strategies

#### Event-Driven Coordination
```python
class EventDrivenCoordinator:
    """
    Event-driven coordination for system components
    """
    def __init__(self):
        self.event_handlers = {}
        self.event_queue = asyncio.Queue()
        self.component_states = {}
        self.event_dispatcher_task = None

    async def initialize(self):
        """Initialize event-driven coordinator"""
        # Register event handlers
        self.register_event_handlers()

        # Start event dispatcher
        self.event_dispatcher_task = asyncio.create_task(self.event_dispatcher())

        self.get_logger().info('Event-driven coordinator initialized')

    def register_event_handlers(self):
        """Register event handlers for different event types"""
        self.event_handlers = {
            'task_started': self.handle_task_started,
            'component_ready': self.handle_component_ready,
            'data_available': self.handle_data_available,
            'error_occurred': self.handle_error_occurred,
            'task_completed': self.handle_task_completed
        }

    async def event_dispatcher(self):
        """Main event dispatcher loop"""
        while True:
            try:
                event = await self.event_queue.get()

                # Get appropriate handler
                event_type = event.get('type')
                if event_type in self.event_handlers:
                    await self.event_handlers[event_type](event)
                else:
                    self.get_logger().warn(f'No handler for event type: {event_type}')

                self.event_queue.task_done()

            except Exception as e:
                self.get_logger().error(f'Error in event dispatcher: {e}')

    async def handle_task_started(self, event):
        """Handle task started event"""
        task_id = event.get('task_id')
        self.get_logger().info(f'Task {task_id} started')

        # Notify relevant components
        await self.notify_components_of_task_start(task_id)

    async def handle_component_ready(self, event):
        """Handle component ready event"""
        component_name = event.get('component')
        self.component_states[component_name] = 'ready'

        # Check if all required components are ready for pending tasks
        await self.check_pending_tasks_readiness()

    async def handle_data_available(self, event):
        """Handle data availability event"""
        data_type = event.get('data_type')
        data = event.get('data')

        # Route data to appropriate consumers
        await self.route_data_to_consumers(data_type, data)

    async def handle_error_occurred(self, event):
        """Handle error occurrence event"""
        error_info = event.get('error')
        component = event.get('component', 'unknown')

        self.get_logger().error(f'Error in {component}: {error_info}')

        # Trigger error recovery procedures
        await self.trigger_error_recovery(event)

    async def handle_task_completed(self, event):
        """Handle task completion event"""
        task_id = event.get('task_id')
        success = event.get('success', False)

        self.get_logger().info(f'Task {task_id} completed with success: {success}')

        # Clean up task resources
        await self.cleanup_task_resources(task_id)

    async def publish_event(self, event_type, data):
        """Publish an event to the system"""
        event = {
            'type': event_type,
            'timestamp': time.time(),
            'data': data
        }
        await self.event_queue.put(event)

    def get_logger(self):
        """Get logger - in practice, this would interface with ROS logger"""
        import logging
        return logging.getLogger(__name__)
```

### Performance Optimization

#### Resource Management
```python
class ResourceManager:
    """
    Resource management for integrated system
    """
    def __init__(self):
        self.resources = {
            'cpu': {'total': 100.0, 'available': 100.0, 'allocated': {}},
            'gpu': {'total': 100.0, 'available': 100.0, 'allocated': {}},
            'memory': {'total': 8192, 'available': 8192, 'allocated': {}},  # MB
            'bandwidth': {'total': 1000, 'available': 1000, 'allocated': {}}  # Mbps
        }

        self.component_requirements = {}
        self.priority_levels = {
            'safety': 10,
            'navigation': 8,
            'manipulation': 7,
            'perception': 6,
            'communication': 5,
            'planning': 4,
            'logging': 2,
            'diagnostics': 1
        }

    def allocate_resources(self, component_name, requirements, priority='normal'):
        """Allocate resources to a component"""
        try:
            # Calculate required resources
            cpu_req = requirements.get('cpu', 0)
            gpu_req = requirements.get('gpu', 0)
            mem_req = requirements.get('memory', 0)
            bw_req = requirements.get('bandwidth', 0)

            # Check availability
            if (cpu_req > self.resources['cpu']['available'] or
                gpu_req > self.resources['gpu']['available'] or
                mem_req > self.resources['memory']['available'] or
                bw_req > self.resources['bandwidth']['available']):

                # Try to free up resources by reducing lower priority components
                success = self.attempt_resource_rebalancing(
                    cpu_req, gpu_req, mem_req, bw_req, priority
                )

                if not success:
                    return {
                        'success': False,
                        'message': 'Insufficient resources and rebalancing failed'
                    }

            # Allocate resources
            self.resources['cpu']['allocated'][component_name] = cpu_req
            self.resources['cpu']['available'] -= cpu_req

            self.resources['gpu']['allocated'][component_name] = gpu_req
            self.resources['gpu']['available'] -= gpu_req

            self.resources['memory']['allocated'][component_name] = mem_req
            self.resources['memory']['available'] -= mem_req

            self.resources['bandwidth']['allocated'][component_name] = bw_req
            self.resources['bandwidth']['available'] -= bw_req

            self.component_requirements[component_name] = requirements

            return {
                'success': True,
                'message': f'Resources allocated to {component_name}',
                'allocation': {
                    'cpu': cpu_req,
                    'gpu': gpu_req,
                    'memory': mem_req,
                    'bandwidth': bw_req
                }
            }

        except Exception as e:
            return {
                'success': False,
                'message': f'Allocation error: {e}'
            }

    def attempt_resource_rebalancing(self, cpu_req, gpu_req, mem_req, bw_req, priority):
        """Attempt to rebalance resources by reducing allocation to lower priority components"""
        priority_value = self.priority_levels.get(priority, 5)

        # Find lower priority components to reduce
        for comp_name, comp_prio in self.priority_levels.items():
            if comp_prio < priority_value and comp_name in self.resources['cpu']['allocated']:
                # Reduce allocation to this component (in practice, do this gradually)
                reduction_factor = 0.2  # Reduce by 20%

                current_cpu = self.resources['cpu']['allocated'][comp_name]
                current_gpu = self.resources['gpu']['allocated'][comp_name]
                current_mem = self.resources['memory']['allocated'][comp_name]
                current_bw = self.resources['bandwidth']['allocated'][comp_name]

                # Calculate potential freed resources
                freed_cpu = current_cpu * reduction_factor
                freed_gpu = current_gpu * reduction_factor
                freed_mem = current_mem * reduction_factor
                freed_bw = current_bw * reduction_factor

                # Check if this frees enough resources
                if (freed_cpu >= cpu_req and freed_gpu >= gpu_req and
                    freed_mem >= mem_req and freed_bw >= bw_req):

                    # Apply reduction
                    self.reduce_allocation(comp_name, reduction_factor)
                    return True

        return False

    def reduce_allocation(self, component_name, factor):
        """Reduce allocation to a component by a factor"""
        if component_name in self.resources['cpu']['allocated']:
            # Reduce allocation by factor
            self.resources['cpu']['available'] += self.resources['cpu']['allocated'][component_name] * factor
            self.resources['cpu']['allocated'][component_name] *= (1 - factor)

            self.resources['gpu']['available'] += self.resources['gpu']['allocated'][component_name] * factor
            self.resources['gpu']['allocated'][component_name] *= (1 - factor)

            self.resources['memory']['available'] += self.resources['memory']['allocated'][component_name] * factor
            self.resources['memory']['allocated'][component_name] *= (1 - factor)

            self.resources['bandwidth']['available'] += self.resources['bandwidth']['allocated'][component_name] * factor
            self.resources['bandwidth']['allocated'][component_name] *= (1 - factor)

    def deallocate_resources(self, component_name):
        """Deallocate resources from a component"""
        if component_name in self.resources['cpu']['allocated']:
            self.resources['cpu']['available'] += self.resources['cpu']['allocated'][component_name]
            del self.resources['cpu']['allocated'][component_name]

        if component_name in self.resources['gpu']['allocated']:
            self.resources['gpu']['available'] += self.resources['gpu']['allocated'][component_name]
            del self.resources['gpu']['allocated'][component_name]

        if component_name in self.resources['memory']['allocated']:
            self.resources['memory']['available'] += self.resources['memory']['allocated'][component_name]
            del self.resources['memory']['allocated'][component_name]

        if component_name in self.resources['bandwidth']['allocated']:
            self.resources['bandwidth']['available'] += self.resources['bandwidth']['allocated'][component_name]
            del self.resources['bandwidth']['allocated'][component_name]

        if component_name in self.component_requirements:
            del self.component_requirements[component_name]
```

### Safety and Error Handling

#### Comprehensive Safety System
```python
class SafetySystem:
    """
    Comprehensive safety system for integrated humanoid robot
    """
    def __init__(self):
        self.safety_levels = {
            'operational': 0,
            'caution': 1,
            'warning': 2,
            'danger': 3,
            'emergency': 4
        }

        self.current_safety_level = 'operational'
        self.safety_rules = []
        self.emergency_procedures = {}
        self.safety_monitoring_task = None

    def initialize_safety_system(self):
        """Initialize safety system with rules and procedures"""
        # Define safety rules
        self.safety_rules = [
            self.check_collision_avoidance,
            self.monitor_balance_stability,
            self.verify_actuator_limits,
            self.check_power_consumption,
            self.validate_sensor_data
        ]

        # Define emergency procedures
        self.emergency_procedures = {
            'collision_imminent': self.emergency_stop,
            'balance_lost': self.balance_recovery,
            'actuator_overload': self.actuator_protection,
            'power_critical': self.power_management,
            'sensor_failure': self.sensor_backup_activation
        }

        # Start safety monitoring
        self.safety_monitoring_task = asyncio.create_task(self.safety_monitoring_loop())

    async def safety_monitoring_loop(self):
        """Continuous safety monitoring loop"""
        while True:
            try:
                # Check all safety rules
                violations = []
                for rule in self.safety_rules:
                    violation = await rule()
                    if violation:
                        violations.append(violation)

                # Handle violations
                if violations:
                    await self.handle_safety_violations(violations)

                # Small delay to prevent busy waiting
                await asyncio.sleep(0.1)  # 100ms monitoring interval

            except Exception as e:
                print(f'Safety monitoring error: {e}')
                await asyncio.sleep(1.0)  # Longer delay on error

    async def check_collision_avoidance(self):
        """Check for potential collision violations"""
        # In practice, this would interface with collision detection system
        # Return violation details if collision is imminent
        return None

    async def monitor_balance_stability(self):
        """Monitor robot balance stability"""
        # Check IMU data for balance issues
        # Return violation if robot is losing balance
        return None

    async def verify_actuator_limits(self):
        """Verify actuator limits are not exceeded"""
        # Check joint positions, velocities, and torques
        # Return violation if limits are exceeded
        return None

    async def check_power_consumption(self):
        """Check power consumption levels"""
        # Monitor battery level and power draw
        # Return violation if power is critically low
        return None

    async def validate_sensor_data(self):
        """Validate sensor data integrity"""
        # Check for sensor failures or anomalous data
        # Return violation if sensor data is invalid
        return None

    async def handle_safety_violations(self, violations):
        """Handle detected safety violations"""
        # Determine the most critical violation
        critical_violation = max(violations, key=lambda v: v.get('severity', 0))

        # Update safety level based on violation severity
        violation_severity = critical_violation.get('severity', 0)
        if violation_severity >= 4:  # Emergency level
            self.current_safety_level = 'emergency'
        elif violation_severity >= 3:  # Danger level
            self.current_safety_level = 'danger'
        elif violation_severity >= 2:  # Warning level
            self.current_safety_level = 'warning'
        elif violation_severity >= 1:  # Caution level
            self.current_safety_level = 'caution'

        # Execute appropriate emergency procedure
        violation_type = critical_violation.get('type', 'unknown')
        if violation_type in self.emergency_procedures:
            await self.emergency_procedures[violation_type](critical_violation)

    async def emergency_stop(self, violation):
        """Execute emergency stop procedure"""
        print(f'EMERGENCY STOP: {violation.get("description", "Unknown violation")}')
        # In practice, this would send emergency stop to all actuators
        # and halt all motion

    async def balance_recovery(self, violation):
        """Execute balance recovery procedure"""
        print(f'BALANCE RECOVERY: {violation.get("description", "Balance lost")}')
        # In practice, this would execute balance recovery algorithms

    async def actuator_protection(self, violation):
        """Execute actuator protection procedure"""
        print(f'ACTUATOR PROTECTION: {violation.get("description", "Overload detected")}')
        # In practice, this would reduce actuator commands to safe levels

    async def power_management(self, violation):
        """Execute power management procedure"""
        print(f'POWER MANAGEMENT: {violation.get("description", "Power critical")}')
        # In practice, this would reduce power consumption and possibly shut down non-critical systems

    async def sensor_backup_activation(self, violation):
        """Execute sensor backup activation procedure"""
        print(f'SENSOR BACKUP ACTIVATION: {violation.get("description", "Sensor failure")}')
        # In practice, this would activate backup sensors or switch to safe mode
```

### Testing Integration

#### Integration Testing Framework
```python
import unittest
import asyncio
from unittest.mock import Mock, patch

class IntegrationTests(unittest.IsolatedAsyncioTestCase):
    """
    Integration tests for the complete system
    """
    async def asyncSetUp(self):
        """Set up test environment"""
        self.system = MockSystemIntegration()
        await self.system.initialize()

    async def test_voice_command_to_navigation_integration(self):
        """Test complete voice command to navigation execution"""
        # Simulate voice command
        command = "Go to the kitchen"

        # Execute the command through the integrated system
        result = await self.system.process_voice_command(command)

        # Verify that navigation component was properly invoked
        self.assertTrue(result['navigation_initiated'])
        self.assertEqual(result['target_location'], 'kitchen')
        self.assertIn('kitchen', result['navigation_goal'])

    async def test_vision_guided_manipulation_integration(self):
        """Test vision-guided manipulation integration"""
        # Set up mock vision data showing an object
        mock_object_data = {
            'class': 'cup',
            'position': [1.0, 0.5, 0.2],
            'pose': [0, 0, 0, 1]
        }

        # Process vision data through system
        result = await self.system.process_vision_data(mock_object_data)

        # Verify manipulation planning was triggered
        self.assertTrue(result['manipulation_planned'])
        self.assertEqual(result['target_object'], 'cup')
        self.assertEqual(result['grasp_pose'], mock_object_data['pose'])

    async def test_multi_component_coordination(self):
        """Test coordination between multiple components"""
        # Simulate complex task requiring multiple components
        task = {
            'type': 'bring_object',
            'object': 'water_bottle',
            'destination': 'user_location'
        }

        # Execute task through integrated system
        result = await self.system.execute_coordinated_task(task)

        # Verify all required components participated
        self.assertTrue(result['voice_recognition_success'])
        self.assertTrue(result['object_detection_success'])
        self.assertTrue(result['navigation_success'])
        self.assertTrue(result['manipulation_success'])
        self.assertTrue(result['delivery_success'])

    async def test_error_propagation_handling(self):
        """Test error propagation and handling"""
        # Simulate a component failure
        with patch.object(self.system.navigation_component, 'navigate', side_effect=Exception("Navigation failed")):
            result = await self.system.execute_task_with_error_handling({
                'type': 'navigation',
                'destination': 'kitchen'
            })

        # Verify graceful degradation
        self.assertFalse(result['navigation_success'])
        self.assertTrue(result['fallback_engaged'])
        self.assertEqual(result['recovery_action'], 'return_to_home')

class MockSystemIntegration:
    """
    Mock system integration for testing purposes
    """
    def __init__(self):
        self.voice_recognizer = Mock()
        self.vision_processor = Mock()
        self.navigation_component = Mock()
        self.manipulation_component = Mock()
        self.task_planner = Mock()

    async def initialize(self):
        """Initialize mock system"""
        pass

    async def process_voice_command(self, command):
        """Mock voice command processing"""
        if "kitchen" in command.lower():
            self.navigation_component.navigate.return_value = True
            return {
                'navigation_initiated': True,
                'target_location': 'kitchen',
                'navigation_goal': ['kitchen']
            }
        return {'navigation_initiated': False}

    async def process_vision_data(self, data):
        """Mock vision data processing"""
        if data['class'] == 'cup':
            self.manipulation_component.plan_grasp.return_value = True
            return {
                'manipulation_planned': True,
                'target_object': 'cup',
                'grasp_pose': data['pose']
            }
        return {'manipulation_planned': False}

    async def execute_coordinated_task(self, task):
        """Mock coordinated task execution"""
        return {
            'voice_recognition_success': True,
            'object_detection_success': True,
            'navigation_success': True,
            'manipulation_success': True,
            'delivery_success': True
        }

    async def execute_task_with_error_handling(self, task):
        """Mock task execution with error handling"""
        return {
            'navigation_success': False,
            'fallback_engaged': True,
            'recovery_action': 'return_to_home'
        }
```

### Summary

In this chapter, we've covered the essential aspects of system integration for humanoid robots:

- **Integration patterns**: Message-based, service-based, and action-based integration approaches
- **Component coordination**: Event-driven and centralized coordination strategies
- **Performance optimization**: Resource management and allocation techniques
- **Safety systems**: Comprehensive safety monitoring and emergency procedures
- **Testing strategies**: Integration testing for complex systems

The key to successful system integration is careful planning, clear interfaces between components, and robust error handling. In the next chapter, we'll explore the complete voice command to action pipeline that brings together all these integration concepts.