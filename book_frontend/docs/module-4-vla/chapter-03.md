---
sidebar_position: 3
---

# Chapter 03: LLM-Based Task Planning

## Intelligent Task Planning with Large Language Models

In this chapter, we'll explore how to use Large Language Models (LLMs) for intelligent task planning in humanoid robots. LLMs excel at understanding natural language commands, reasoning about the world, and breaking down complex tasks into executable steps - making them ideal for bridging the gap between human communication and robot action.

### Understanding LLM-Based Task Planning

LLM-based task planning leverages the reasoning and language understanding capabilities of large language models to:
- Interpret natural language commands
- Generate task plans from high-level goals
- Handle ambiguous or complex instructions
- Adapt plans based on context and environment

The LLM task planning pipeline:

```
┌─────────────────┐    ┌─────────────────┐    └─────────────────┘
│  Natural        │ -> │  LLM Task       │ -> │  Executable     │
│  Language       │    │  Generation     │    │  Action Plan    │
│  Command        │    │  & Reasoning    │    │  & Parameters   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Setting Up LLM Integration

For this chapter, we'll use OpenAI's GPT models, but the concepts apply to other LLMs as well:

```bash
# Install required packages
pip install openai python-dotenv
pip install langchain langchain-community  # For advanced LLM integration
```

### Basic LLM Task Planner

Let's start with a basic implementation:

```python
import openai
import json
import re
from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum

class ActionType(Enum):
    """Types of actions the robot can perform"""
    NAVIGATE = "navigate"
    GRASP = "grasp"
    PLACE = "place"
    SPEAK = "speak"
    DETECT = "detect"
    WAIT = "wait"
    APPROACH = "approach"

@dataclass
class RobotAction:
    """Represents a single robot action"""
    type: ActionType
    parameters: Dict[str, Any]
    description: str

class LLMBasicTaskPlanner:
    """
    Basic LLM-based task planner for humanoid robots
    """
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        openai.api_key = api_key
        self.model = model

        # Define robot capabilities
        self.capabilities = {
            "navigation": ["move_to", "navigate_to", "go_to", "walk_to"],
            "manipulation": ["grasp", "pick_up", "place", "put_down", "hold"],
            "interaction": ["speak", "greet", "introduce", "communicate"],
            "perception": ["detect", "find", "look_for", "search_for"]
        }

    def plan_task(self, command: str, robot_state: Dict = None) -> List[RobotAction]:
        """
        Plan a task based on natural language command

        Args:
            command: Natural language command
            robot_state: Current state of the robot

        Returns:
            List of RobotAction objects
        """
        if robot_state is None:
            robot_state = {
                "position": [0, 0, 0],
                "orientation": [0, 0, 0, 1],
                "battery_level": 100.0,
                "available_objects": ["cup", "book", "bottle", "chair"]
            }

        # Create a prompt for the LLM
        prompt = self._create_task_planning_prompt(command, robot_state)

        try:
            # Call the LLM
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )

            # Parse the response
            actions = self._parse_llm_response(response.choices[0].message.content)
            return actions

        except Exception as e:
            print(f"Error in LLM task planning: {e}")
            return self._create_fallback_plan(command)

    def _create_task_planning_prompt(self, command: str, robot_state: Dict) -> str:
        """Create prompt for task planning"""
        prompt = f"""
        Human command: "{command}"

        Current robot state:
        - Position: {robot_state['position']}
        - Available objects: {', '.join(robot_state['available_objects'])}
        - Battery level: {robot_state['battery_level']}%

        Robot capabilities:
        - Navigation: move to locations
        - Manipulation: grasp and place objects
        - Interaction: speak and communicate
        - Perception: detect and find objects

        Please break down this command into a sequence of executable actions for the humanoid robot.
        Each action should be one of: navigate, grasp, place, speak, detect, wait, approach
        Return the actions in JSON format as a list of objects with 'type' and 'parameters'.
        """
        return prompt

    def _get_system_prompt(self) -> str:
        """Get system prompt for consistent behavior"""
        return """
        You are an expert at planning tasks for a humanoid robot. Your role is to:
        1. Understand the human's natural language command
        2. Break it down into simple, executable actions
        3. Consider the robot's current state and capabilities
        4. Return a sequence of actions in JSON format

        Actions must be one of: navigate, grasp, place, speak, detect, wait, approach
        Always return valid JSON with 'type' and 'parameters' fields for each action.
        """

    def _parse_llm_response(self, response: str) -> List[RobotAction]:
        """Parse LLM response into RobotAction objects"""
        try:
            # Extract JSON from response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                actions_data = json.loads(json_str)

                actions = []
                for action_data in actions_data:
                    action_type = ActionType(action_data['type'])
                    action = RobotAction(
                        type=action_type,
                        parameters=action_data.get('parameters', {}),
                        description=action_data.get('description', '')
                    )
                    actions.append(action)

                return actions
            else:
                # If no JSON found, try to parse as text
                return self._parse_text_response(response)
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return []

    def _parse_text_response(self, response: str) -> List[RobotAction]:
        """Parse text response as fallback"""
        # Simple parsing based on keywords
        actions = []
        lines = response.split('\n')

        for line in lines:
            if 'navigate' in line.lower() or 'move' in line.lower():
                actions.append(RobotAction(
                    type=ActionType.NAVIGATE,
                    parameters={'target_location': 'unknown'},
                    description=line
                ))
            elif 'grasp' in line.lower() or 'pick' in line.lower():
                actions.append(RobotAction(
                    type=ActionType.GRASP,
                    parameters={'target_object': 'unknown'},
                    description=line
                ))
            elif 'speak' in line.lower() or 'say' in line.lower():
                actions.append(RobotAction(
                    type=ActionType.SPEAK,
                    parameters={'text': line},
                    description=line
                ))

        return actions

    def _create_fallback_plan(self, command: str) -> List[RobotAction]:
        """Create a fallback plan if LLM fails"""
        return [RobotAction(
            type=ActionType.SPEAK,
            parameters={'text': f"I'm sorry, I couldn't understand: {command}"},
            description="Apologize for not understanding command"
        )]
```

### Advanced LLM Task Planner with Context

For more sophisticated task planning, we need to maintain context and handle complex scenarios:

```python
import openai
import json
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class TaskContext:
    """Context for task planning"""
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    current_objects: List[Dict[str, Any]] = field(default_factory=list)
    robot_capabilities: Dict[str, Any] = field(default_factory=dict)
    environment_map: Dict[str, Any] = field(default_factory=dict)
    current_task: str = ""
    task_history: List[str] = field(default_factory=list)

class LLMAdvancedTaskPlanner:
    """
    Advanced LLM-based task planner with context management
    """
    def __init__(self, api_key: str, model: str = "gpt-4"):
        openai.api_key = api_key
        self.model = model
        self.context = TaskContext()

        # Initialize robot capabilities
        self.context.robot_capabilities = {
            "navigation": {
                "max_speed": 0.5,  # m/s
                "turn_speed": 0.5,  # rad/s
                "step_height": 0.15  # m
            },
            "manipulation": {
                "reach_distance": 1.0,  # m
                "grasp_types": ["pinch", "power", "hook"],
                "max_load": 2.0  # kg
            },
            "sensors": ["camera", "lidar", "imu", "force_torque"],
            "speakers": True
        }

        # Object knowledge base
        self.object_knowledge = {
            "cup": {"grasp_type": "pinch", "weight": 0.3, "typical_location": ["kitchen", "table"]},
            "book": {"grasp_type": "power", "weight": 0.5, "typical_location": ["shelf", "table"]},
            "bottle": {"grasp_type": "power", "weight": 0.7, "typical_location": ["kitchen", "refrigerator"]},
            "chair": {"grasp_type": "power", "weight": 5.0, "typical_location": ["living_room", "dining_room"]}
        }

    def update_context(self, **kwargs):
        """Update task context"""
        for key, value in kwargs.items():
            if hasattr(self.context, key):
                setattr(self.context, key, value)

    def plan_task(self, command: str, environment_state: Dict = None) -> Dict[str, Any]:
        """
        Plan a task with full context consideration

        Returns:
            Dictionary with 'actions', 'confidence', 'reasoning', and 'next_steps'
        """
        if environment_state is None:
            environment_state = {
                "objects_in_view": ["cup", "book"],
                "room": "kitchen",
                "people_present": 1,
                "obstacles": []
            }

        # Add command to conversation history
        self.context.conversation_history.append({
            "role": "user",
            "content": command,
            "timestamp": datetime.now().isoformat()
        })

        # Create detailed prompt
        prompt = self._create_detailed_prompt(command, environment_state)

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_advanced_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,  # Lower temperature for more consistent planning
                max_tokens=800,
                functions=[
                    {
                        "name": "plan_robot_task",
                        "description": "Plan a sequence of actions for a humanoid robot",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "actions": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "type": {"type": "string", "enum": ["navigate", "grasp", "place", "speak", "detect", "wait", "approach"]},
                                            "parameters": {"type": "object"},
                                            "description": {"type": "string"}
                                        },
                                        "required": ["type", "parameters", "description"]
                                    }
                                },
                                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                                "reasoning": {"type": "string"},
                                "potential_issues": {"type": "array", "items": {"type": "string"}},
                                "next_steps": {"type": "array", "items": {"type": "string"}}
                            },
                            "required": ["actions", "confidence", "reasoning"]
                        }
                    }
                ],
                function_call={"name": "plan_robot_task"}
            )

            # Parse the function call result
            result = json.loads(response.choices[0].message.function_call.arguments)

            # Add to conversation history
            self.context.conversation_history.append({
                "role": "assistant",
                "content": json.dumps(result),
                "timestamp": datetime.now().isoformat()
            })

            # Update current task
            self.context.current_task = command

            return result

        except Exception as e:
            print(f"Error in advanced task planning: {e}")
            return self._create_error_response(command)

    def _create_detailed_prompt(self, command: str, environment_state: Dict) -> str:
        """Create detailed prompt with full context"""
        return f"""
        Human command: "{command}"

        Current context:
        - Conversation history: {json.dumps(self.context.conversation_history[-3:])}
        - Current environment: {json.dumps(environment_state)}
        - Available objects: {json.dumps(self.context.current_objects)}
        - Robot capabilities: {json.dumps(self.context.robot_capabilities)}
        - Known object properties: {json.dumps(self.object_knowledge)}

        Please generate a detailed task plan that includes:
        1. A sequence of executable actions
        2. Confidence level (0-1) in the plan
        3. Reasoning for the approach
        4. Potential issues to watch for
        5. Next steps if the plan fails

        Consider:
        - Physical constraints of the humanoid robot
        - Safety of the actions
        - Efficiency of the plan
        - Natural human-robot interaction
        """

    def _get_advanced_system_prompt(self) -> str:
        """Get advanced system prompt"""
        return """
        You are an expert task planner for a humanoid robot. Your role is to:
        1. Understand complex natural language commands
        2. Generate safe, executable action sequences
        3. Consider robot capabilities and environmental constraints
        4. Plan for contingencies and error recovery
        5. Provide confidence estimates for your plans

        Always return structured JSON with actions, confidence, reasoning, potential issues, and next steps.
        Actions should be specific and executable by a humanoid robot.
        """

    def _create_error_response(self, command: str) -> Dict[str, Any]:
        """Create error response when planning fails"""
        return {
            "actions": [
                {
                    "type": "speak",
                    "parameters": {"text": f"I'm sorry, I'm having trouble understanding '{command}'. Could you please rephrase?"},
                    "description": "Ask for clarification"
                }
            ],
            "confidence": 0.3,
            "reasoning": "Command unclear or beyond robot capabilities",
            "potential_issues": ["Misunderstanding", "Capability limitations"],
            "next_steps": ["Wait for clarification", "Ask for simpler command"]
        }

    def refine_plan(self, original_plan: Dict[str, Any], feedback: str) -> Dict[str, Any]:
        """
        Refine a plan based on feedback or new information
        """
        refinement_prompt = f"""
        Original plan: {json.dumps(original_plan)}
        Feedback received: "{feedback}"

        Please refine the plan based on this feedback while maintaining the overall goal.
        Return the updated plan in the same format.
        """

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_advanced_system_prompt()},
                    {"role": "user", "content": refinement_prompt}
                ],
                temperature=0.1,
                max_tokens=600,
                functions=[
                    {
                        "name": "plan_robot_task",
                        "description": "Plan a sequence of actions for a humanoid robot",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "actions": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "type": {"type": "string", "enum": ["navigate", "grasp", "place", "speak", "detect", "wait", "approach"]},
                                            "parameters": {"type": "object"},
                                            "description": {"type": "string"}
                                        },
                                        "required": ["type", "parameters", "description"]
                                    }
                                },
                                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                                "reasoning": {"type": "string"},
                                "potential_issues": {"type": "array", "items": {"type": "string"}},
                                "next_steps": {"type": "array", "items": {"type": "string"}}
                            },
                            "required": ["actions", "confidence", "reasoning"]
                        }
                    }
                ],
                function_call={"name": "plan_robot_task"}
            )

            refined_plan = json.loads(response.choices[0].message.function_call.arguments)
            return refined_plan

        except Exception as e:
            print(f"Error refining plan: {e}")
            return original_plan  # Return original if refinement fails
```

### Integration with ROS 2

Let's integrate the LLM task planner with ROS 2:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from action_msgs.msg import GoalStatus
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
import threading
import asyncio

class LLMTasksROSIntegration(Node):
    """
    ROS 2 integration for LLM-based task planning
    """
    def __init__(self):
        super().__init__('llm_task_planner')

        # Initialize LLM task planner
        import os
        api_key = os.getenv('OPENAI_API_KEY', 'your-api-key-here')
        self.llm_planner = LLMAdvancedTaskPlanner(api_key)

        # Publishers
        self.task_plan_pub = self.create_publisher(String, '/llm_task_plan', 10)
        self.action_pub = self.create_publisher(String, '/robot_action', 10)
        self.status_pub = self.create_publisher(String, '/llm_status', 10)

        # Subscribers
        self.command_sub = self.create_subscription(
            String, '/natural_language_command', self.command_callback, 10
        )

        self.environment_sub = self.create_subscription(
            String, '/environment_state', self.environment_callback, 10
        )

        # Action server for task execution
        self._action_server = ActionServer(
            self,
            ExecuteTask,
            'execute_task',
            self.execute_task_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=ReentrantCallbackGroup()
        )

        # Store current environment state
        self.current_environment = {}
        self.current_task_plan = None

        self.get_logger().info('LLM Task Planner ROS 2 integration initialized')

    def command_callback(self, msg):
        """Handle natural language command"""
        command = msg.data
        self.get_logger().info(f'Received command: {command}')

        # Plan the task using LLM
        task_plan = self.llm_planner.plan_task(command, self.current_environment)

        # Store the plan
        self.current_task_plan = task_plan

        # Publish the plan
        plan_msg = String()
        plan_msg.data = json.dumps(task_plan)
        self.task_plan_pub.publish(plan_msg)

        # Publish status
        status_msg = String()
        status_msg.data = f"Planned {len(task_plan.get('actions', []))} actions with {task_plan.get('confidence', 0):.2f} confidence"
        self.status_pub.publish(status_msg)

    def environment_callback(self, msg):
        """Handle environment state updates"""
        try:
            self.current_environment = json.loads(msg.data)
            self.get_logger().debug('Updated environment state')
        except json.JSONDecodeError:
            self.get_logger().error('Invalid environment state JSON')

    def goal_callback(self, goal_request):
        """Handle task execution goal"""
        self.get_logger().info('Received task execution goal')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Handle task execution cancellation"""
        self.get_logger().info('Received task cancellation')
        return CancelResponse.ACCEPT

    async def execute_task_callback(self, goal_handle):
        """Execute the planned task"""
        self.get_logger().info('Executing task...')

        # If we have a current plan, execute it
        if self.current_task_plan:
            feedback_msg = ExecuteTask.Feedback()
            result = ExecuteTask.Result()

            actions = self.current_task_plan.get('actions', [])
            total_actions = len(actions)

            for i, action in enumerate(actions):
                # Publish action for robot execution
                action_msg = String()
                action_msg.data = json.dumps(action)
                self.action_pub.publish(action_msg)

                # Update feedback
                feedback_msg.current_action = action.get('description', f'Action {i+1}')
                feedback_msg.progress = float(i + 1) / total_actions
                goal_handle.publish_feedback(feedback_msg)

                # Wait for action completion (simplified)
                await asyncio.sleep(1.0)  # Replace with actual action completion check

            result.success = True
            result.message = f"Executed {total_actions} actions successfully"
        else:
            result.success = False
            result.message = "No task plan available"

        goal_handle.succeed()
        return result

def main(args=None):
    rclpy.init(args=args)

    llm_node = LLMTasksROSIntegration()

    try:
        rclpy.spin(llm_node)
    except KeyboardInterrupt:
        pass
    finally:
        llm_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Handling Complex Multi-Step Tasks

For complex tasks, we need to handle task decomposition and execution monitoring:

```python
class ComplexTaskHandler:
    """
    Handles complex multi-step tasks with monitoring and recovery
    """
    def __init__(self, llm_planner: LLMAdvancedTaskPlanner):
        self.planner = llm_planner
        self.current_task = None
        self.task_status = "idle"
        self.execution_history = []

    def execute_complex_task(self, command: str, environment: Dict) -> Dict[str, Any]:
        """
        Execute a complex task with monitoring and recovery
        """
        # Plan the task
        plan = self.planner.plan_task(command, environment)

        # Execute step by step with monitoring
        execution_result = self._execute_step_by_step(plan, environment)

        return execution_result

    def _execute_step_by_step(self, plan: Dict, environment: Dict) -> Dict[str, Any]:
        """
        Execute plan step by step with monitoring
        """
        actions = plan.get('actions', [])
        results = []
        success = True
        error_message = ""

        for i, action in enumerate(actions):
            self.get_logger().info(f"Executing action {i+1}/{len(actions)}: {action['description']}")

            try:
                # Execute the action
                action_result = self._execute_single_action(action, environment)

                # Check if action succeeded
                if action_result.get('success', True):
                    results.append(action_result)
                    self.execution_history.append({
                        'action': action,
                        'result': action_result,
                        'status': 'success'
                    })
                else:
                    # Action failed, try recovery
                    recovery_result = self._attempt_recovery(action, action_result, environment)
                    if recovery_result.get('success', False):
                        results.append(recovery_result)
                        self.execution_history.append({
                            'action': action,
                            'result': recovery_result,
                            'status': 'recovered'
                        })
                    else:
                        success = False
                        error_message = f"Action failed and recovery unsuccessful: {action['description']}"
                        break

            except Exception as e:
                self.get_logger().error(f"Error executing action {i+1}: {e}")
                success = False
                error_message = str(e)
                break

        return {
            'success': success,
            'results': results,
            'total_actions': len(actions),
            'completed_actions': len(results),
            'error': error_message if not success else None,
            'execution_history': self.execution_history
        }

    def _execute_single_action(self, action: Dict, environment: Dict) -> Dict[str, Any]:
        """
        Execute a single action and return result
        """
        # In practice, this would interface with the robot's action execution system
        # For now, we'll simulate execution
        action_type = action['type']

        # Simulate action execution time
        import time
        time.sleep(0.5)  # Simulated execution time

        # Simulate success/failure based on action type
        success_probability = {
            'navigate': 0.95,
            'grasp': 0.85,
            'place': 0.90,
            'speak': 0.99,
            'detect': 0.90,
            'wait': 1.0,
            'approach': 0.95
        }

        import random
        success = random.random() < success_probability.get(action_type, 0.9)

        return {
            'action': action,
            'success': success,
            'execution_time': 0.5,
            'details': f"Action {action_type} executed {'successfully' if success else 'with issues'}"
        }

    def _attempt_recovery(self, failed_action: Dict, failure_result: Dict, environment: Dict) -> Dict[str, Any]:
        """
        Attempt to recover from a failed action
        """
        # Use LLM to suggest recovery strategy
        recovery_prompt = f"""
        Action failed: {json.dumps(failed_action)}
        Failure result: {json.dumps(failure_result)}
        Current environment: {json.dumps(environment)}

        Please suggest a recovery strategy or alternative approach.
        Return the recovery action in the same format as the original action.
        """

        try:
            response = openai.ChatCompletion.create(
                model=self.planner.model,
                messages=[
                    {"role": "system", "content": "You are an expert at robot task recovery. Suggest alternative approaches when actions fail."},
                    {"role": "user", "content": recovery_prompt}
                ],
                temperature=0.1,
                max_tokens=300
            )

            # Parse recovery action (simplified)
            recovery_action = failed_action.copy()  # For demo, just retry
            recovery_action['description'] = f"Recovery attempt: {recovery_action['description']}"

            # Execute recovery action
            return self._execute_single_action(recovery_action, environment)

        except Exception as e:
            return {
                'action': failed_action,
                'success': False,
                'error': f"Recovery failed: {e}",
                'details': "Could not recover from action failure"
            }
```

### Context-Aware Planning

For more intelligent planning, we need to maintain and use context:

```python
class ContextAwarePlanner:
    """
    LLM planner that maintains and uses context for better planning
    """
    def __init__(self, llm_planner: LLMAdvancedTaskPlanner):
        self.planner = llm_planner
        self.context_memory = []
        self.user_preferences = {}
        self.task_patterns = {}  # Learned patterns from previous tasks

    def plan_with_context(self, command: str, environment: Dict, user_id: str = None) -> Dict[str, Any]:
        """
        Plan task considering historical context and user preferences
        """
        # Load user context if available
        user_context = self._get_user_context(user_id) if user_id else {}

        # Get recent conversation context
        recent_context = self._get_recent_context()

        # Combine all context
        full_context = {
            'user_preferences': user_context,
            'recent_conversations': recent_context,
            'learned_patterns': self.task_patterns,
            'current_environment': environment,
            'current_command': command
        }

        # Create enhanced prompt with context
        enhanced_prompt = self._create_context_enhanced_prompt(command, full_context)

        # Plan using enhanced context
        plan = self._plan_with_enhanced_context(enhanced_prompt, environment)

        # Store this interaction for future learning
        self._store_interaction(command, plan, user_id)

        return plan

    def _get_user_context(self, user_id: str) -> Dict:
        """Get user-specific context"""
        # In practice, this would load from a database
        return self.user_preferences.get(user_id, {})

    def _get_recent_context(self) -> List[Dict]:
        """Get recent conversation context"""
        return self.context_memory[-10:]  # Last 10 interactions

    def _create_context_enhanced_prompt(self, command: str, context: Dict) -> str:
        """Create prompt enhanced with context"""
        return f"""
        Human command: "{command}"

        Context information:
        - User preferences: {json.dumps(context['user_preferences'])}
        - Recent interactions: {json.dumps(context['recent_conversations'])}
        - Learned patterns: {json.dumps(context['learned_patterns'])}
        - Current environment: {json.dumps(context['current_environment'])}

        Please generate a task plan that considers:
        1. User preferences and past behavior
        2. Similar tasks performed before
        3. Current environmental context
        4. Most efficient approach based on learned patterns

        Return in standard format.
        """

    def _plan_with_enhanced_context(self, prompt: str, environment: Dict) -> Dict[str, Any]:
        """Plan using enhanced context"""
        # This would call the LLM with the enhanced prompt
        # For now, we'll use the standard planner with some context hints
        return self.planner.plan_task(prompt, environment)

    def _store_interaction(self, command: str, plan: Dict, user_id: str = None):
        """Store interaction for learning"""
        interaction = {
            'command': command,
            'plan': plan,
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id
        }

        # Add to context memory
        self.context_memory.append(interaction)

        # Keep only recent interactions
        if len(self.context_memory) > 100:
            self.context_memory = self.context_memory[-100:]

        # Update user preferences if user ID provided
        if user_id:
            self._update_user_preferences(user_id, interaction)

    def _update_user_preferences(self, user_id: str, interaction: Dict):
        """Update user preferences based on interaction"""
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {
                'preferred_actions': [],
                'common_tasks': [],
                'interaction_style': 'formal'
            }

        # Simple preference learning (in practice, use more sophisticated methods)
        actions = [action['type'] for action in interaction['plan'].get('actions', [])]
        self.user_preferences[user_id]['preferred_actions'].extend(actions)
```

### Error Handling and Fallback Strategies

Implement robust error handling for production use:

```python
import logging
from functools import wraps
from typing import Callable, Any

def llm_retry(max_attempts: int = 3, fallback_func: Callable = None):
    """Decorator for retrying LLM calls with fallback"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logging.warning(f"LLM call failed (attempt {attempt + 1}): {e}")
                    last_exception = e

                    if attempt < max_attempts - 1:
                        import time
                        time.sleep(2 ** attempt)  # Exponential backoff

            # All attempts failed, use fallback if available
            if fallback_func:
                logging.info("Using fallback function after LLM failures")
                return fallback_func(*args, **kwargs)
            else:
                raise last_exception

        return wrapper
    return decorator

class RobustLLMPlanner:
    """
    Production-ready LLM planner with error handling
    """
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        openai.api_key = api_key
        self.model = model
        self.logger = logging.getLogger(__name__)

    @llm_retry(max_attempts=3, fallback_func=lambda cmd, env: [{"type": "speak", "parameters": {"text": f"Error processing: {cmd}"}, "description": "Error response"}])
    def plan_task_with_retry(self, command: str, environment: Dict = None) -> List[Dict]:
        """Plan task with automatic retry and fallback"""
        if environment is None:
            environment = {}

        # Call LLM with error handling
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Plan robot actions from natural language"},
                {"role": "user", "content": f"Command: {command}, Environment: {environment}"}
            ],
            temperature=0.3,
            max_tokens=500
        )

        # Parse response
        content = response.choices[0].message.content
        # Extract and return actions (simplified parsing)
        return [{"type": "speak", "parameters": {"text": content}, "description": "LLM response"}]
```

### Summary

In this chapter, we've covered:
- Basic and advanced LLM-based task planning
- Context management and conversation history
- ROS 2 integration for humanoid robots
- Complex multi-step task execution
- Context-aware planning with user preferences
- Error handling and fallback strategies

In the next chapter, we'll explore vision-guided manipulation, learning how to integrate computer vision with robot manipulation for precise object interaction.