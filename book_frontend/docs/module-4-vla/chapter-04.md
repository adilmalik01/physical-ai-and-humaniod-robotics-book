---
sidebar_position: 4
---

# Chapter 04: Vision-Guided Manipulation

## Precise Object Interaction with Computer Vision

In this chapter, we'll explore vision-guided manipulation, which combines computer vision with robotic manipulation to enable precise object interaction. This is a critical capability for humanoid robots, allowing them to perceive objects in their environment and perform dexterous manipulation tasks based on visual feedback.

### Understanding Vision-Guided Manipulation

Vision-guided manipulation integrates three key components:

1. **Computer Vision**: Perceiving and understanding objects in the environment
2. **Manipulation Planning**: Planning grasps and manipulation actions
3. **Control Systems**: Executing precise movements with visual feedback

The vision-guided manipulation pipeline:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Visual       │ -> │  Grasp &       │ -> │  Manipulation   │
│   Perception   │    │  Manipulation  │    │  Execution      │
│   (Detection,  │    │  Planning      │    │  (Control,      │
│   Pose Est.)   │    │  (IK, Grasps)  │    │  Feedback)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Computer Vision for Manipulation

For manipulation tasks, we need specialized computer vision capabilities:

#### Object Detection and Localization

```python
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from scipy.spatial.transform import Rotation as R

class ManipulationVisionSystem:
    """
    Computer vision system for manipulation tasks
    """
    def __init__(self):
        # Load object detection model (e.g., YOLO, Detectron2)
        self.detection_model = self.load_detection_model()

        # Load pose estimation model
        self.pose_model = self.load_pose_model()

        # Initialize camera parameters
        self.camera_matrix = np.array([
            [554.256, 0.0, 320.0],
            [0.0, 554.256, 240.0],
            [0.0, 0.0, 1.0]
        ])

        # Object models for pose estimation
        self.object_models = self.load_object_models()

        # Transform from camera to robot base
        self.cam_to_robot = self.get_camera_to_robot_transform()

    def load_detection_model(self):
        """
        Load object detection model
        In practice, this could be YOLO, Detectron2, or similar
        """
        # Using a pre-trained model for demonstration
        # In practice, you'd load your specific model
        return None

    def load_pose_model(self):
        """
        Load 6D pose estimation model
        """
        return None

    def load_object_models(self):
        """
        Load 3D models of objects for pose estimation
        """
        # Dictionary of object models with their properties
        return {
            'cup': {
                'dimensions': [0.08, 0.08, 0.1],  # width, depth, height
                'grasp_points': [[0, 0, 0.05], [0.04, 0, 0.05]],  # potential grasp points
                'preferred_grasp_type': 'pinch'
            },
            'bottle': {
                'dimensions': [0.06, 0.06, 0.25],
                'grasp_points': [[0, 0, 0.1], [0.03, 0, 0.1]],
                'preferred_grasp_type': 'power'
            },
            'book': {
                'dimensions': [0.2, 0.15, 0.03],
                'grasp_points': [[0.1, 0, 0.015], [-0.1, 0, 0.015]],
                'preferred_grasp_type': 'power'
            }
        }

    def detect_objects(self, image):
        """
        Detect objects in the image and return bounding boxes and classes
        """
        # In practice, this would use your detection model
        # For demonstration, returning mock detections
        detections = [
            {
                'class': 'cup',
                'bbox': [100, 150, 200, 250],  # [x1, y1, x2, y2]
                'confidence': 0.95,
                'center': [150, 200]  # center pixel coordinates
            },
            {
                'class': 'book',
                'bbox': [300, 100, 450, 200],
                'confidence': 0.89,
                'center': [375, 150]
            }
        ]
        return detections

    def estimate_object_pose(self, image, object_class, bbox):
        """
        Estimate 6D pose (position + orientation) of object
        """
        # Extract object region
        x1, y1, x2, y2 = bbox
        object_region = image[y1:y2, x1:x2]

        # Use pose estimation model (simplified)
        # In practice, this would use methods like PVNet, Pix2Pose, etc.
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # Convert 2D pixel coordinates to 3D world coordinates
        # This requires depth information or known object size
        depth = self.estimate_depth(image, object_class, bbox)

        # Convert to 3D coordinates using camera matrix
        world_pos = self.pixel_to_world([center_x, center_y], depth)

        # Estimate orientation (simplified)
        # In practice, this would use more sophisticated pose estimation
        orientation = R.from_euler('xyz', [0, 0, 0]).as_quat()

        return {
            'position': world_pos,
            'orientation': orientation,
            'bbox': bbox
        }

    def estimate_depth(self, image, object_class, bbox):
        """
        Estimate depth of object (simplified approach)
        In practice, use depth camera, stereo vision, or monocular depth estimation
        """
        # For demonstration, return a fixed depth
        # In practice, use actual depth sensor or estimation
        if object_class == 'cup':
            return 0.8  # meters
        elif object_class == 'book':
            return 0.75
        else:
            return 0.85

    def pixel_to_world(self, pixel_coords, depth):
        """
        Convert pixel coordinates to world coordinates using camera matrix
        """
        u, v = pixel_coords
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]

        # Convert to world coordinates
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth

        # Transform from camera frame to robot base frame
        world_coords = self.transform_to_robot_frame([x, y, z])
        return world_coords

    def transform_to_robot_frame(self, coords):
        """
        Transform coordinates from camera frame to robot base frame
        """
        # Apply transformation matrix
        # This should be calibrated for your specific robot setup
        transformed = np.dot(self.cam_to_robot[:3, :3], coords) + self.cam_to_robot[:3, 3]
        return transformed.tolist()

    def get_camera_to_robot_transform(self):
        """
        Get transformation from camera frame to robot base frame
        This should be calibrated for your specific setup
        """
        # Identity matrix as placeholder
        # In practice, this would be a calibrated transformation
        return np.eye(4)
```

#### 3D Object Pose Estimation

For precise manipulation, we need accurate 6D pose (position and orientation) of objects:

```python
class PoseEstimation:
    """
    Advanced pose estimation for manipulation
    """
    def __init__(self):
        # Initialize pose estimation models
        self.keypoint_detector = self.load_keypoint_detector()
        self.correspondence_matcher = self.load_correspondence_matcher()
        self.pose_refiner = self.load_pose_refiner()

    def estimate_pose_with_refinement(self, image, object_class, object_model):
        """
        Estimate pose using keypoint detection and iterative refinement
        """
        # Detect keypoints in the image
        image_keypoints = self.detect_keypoints(image, object_class)

        # Match with 3D model keypoints
        correspondences = self.match_keypoints(image_keypoints, object_model)

        # Initial pose estimation using PnP
        initial_pose = self.estimate_initial_pose(correspondences)

        # Refine pose using iterative methods
        refined_pose = self.refine_pose(image, object_model, initial_pose)

        return refined_pose

    def detect_keypoints(self, image, object_class):
        """
        Detect object keypoints in the image
        """
        # In practice, use specialized keypoint detection
        # For example, using learned descriptors or template matching
        pass

    def match_keypoints(self, image_kpts, model_kpts):
        """
        Match image keypoints with 3D model keypoints
        """
        # Match using descriptors or geometric constraints
        pass

    def estimate_initial_pose(self, correspondences):
        """
        Estimate initial pose using Perspective-n-Point algorithm
        """
        # Use OpenCV's solvePnP or similar
        pass

    def refine_pose(self, image, model, initial_pose):
        """
        Refine pose estimate using iterative methods
        """
        # Use methods like ICP, EPnP, or learning-based refinement
        pass
```

### Grasp Planning

Once we have object poses, we need to plan appropriate grasps:

```python
class GraspPlanner:
    """
    Plan grasps for manipulation based on object properties and vision data
    """
    def __init__(self):
        # Grasp types supported by the robot
        self.grasp_types = {
            'pinch': {'max_aperture': 0.08, 'min_aperture': 0.01, 'force': 20},
            'power': {'max_aperture': 0.12, 'min_aperture': 0.02, 'force': 50},
            'hook': {'max_aperture': 0.15, 'min_aperture': 0.03, 'force': 30}
        }

        # Grasp quality metrics
        self.quality_weights = {
            'stability': 0.4,
            'accessibility': 0.3,
            'safety': 0.2,
            'efficiency': 0.1
        }

    def plan_grasps(self, object_pose, object_class, robot_state):
        """
        Plan potential grasps for an object
        """
        object_info = self.get_object_info(object_class)

        # Generate candidate grasp poses
        candidate_grasps = self.generate_candidate_grasps(
            object_pose, object_info, robot_state
        )

        # Evaluate grasp quality
        evaluated_grasps = []
        for grasp in candidate_grasps:
            quality = self.evaluate_grasp_quality(grasp, object_info, robot_state)
            evaluated_grasps.append({
                'pose': grasp,
                'quality': quality,
                'type': self.select_grasp_type(object_info)
            })

        # Sort by quality
        evaluated_grasps.sort(key=lambda x: x['quality'], reverse=True)

        return evaluated_grasps

    def generate_candidate_grasps(self, object_pose, object_info, robot_state):
        """
        Generate candidate grasp poses around the object
        """
        candidate_poses = []

        # Get object dimensions
        dims = object_info['dimensions']
        pos = object_pose['position']
        orient = object_pose['orientation']

        # Generate grasps at different positions around the object
        grasp_points = self.get_grasp_points(object_info)

        for point in grasp_points:
            # Transform point to world coordinates based on object pose
            world_point = self.transform_point_to_world(point, object_pose)

            # Generate approach direction (typically from above or side)
            approach_dirs = self.generate_approach_directions()

            for approach_dir in approach_dirs:
                # Create grasp pose with approach direction
                grasp_pose = self.create_grasp_pose(world_point, approach_dir, orient)
                candidate_poses.append(grasp_pose)

        return candidate_poses

    def get_grasp_points(self, object_info):
        """
        Get potential grasp points for the object
        """
        if 'grasp_points' in object_info:
            return object_info['grasp_points']
        else:
            # Generate default grasp points based on object dimensions
            dims = object_info['dimensions']
            return [
                [dims[0]/2, 0, dims[2]/2],  # Side grasp
                [0, dims[1]/2, dims[2]/2],  # Front grasp
                [0, 0, dims[2]]             # Top grasp
            ]

    def transform_point_to_world(self, local_point, object_pose):
        """
        Transform a point from object local coordinates to world coordinates
        """
        # Apply object transformation to local point
        pos = object_pose['position']
        orient = object_pose['orientation']

        # Convert quaternion to rotation matrix
        rot = R.from_quat(orient).as_matrix()

        # Transform the point
        world_point = np.dot(rot, local_point) + pos
        return world_point

    def generate_approach_directions(self):
        """
        Generate possible approach directions for grasping
        """
        return [
            [0, 0, -1],  # From above
            [0, -1, 0],  # From front
            [-1, 0, 0],  # From side
            [1, 0, 0],   # From opposite side
            [0, 1, 0]    # From back
        ]

    def create_grasp_pose(self, position, approach_dir, object_orient):
        """
        Create a grasp pose given position and approach direction
        """
        # Normalize approach direction
        approach = np.array(approach_dir) / np.linalg.norm(approach_dir)

        # Calculate grasp orientation based on approach direction
        # This depends on the gripper type and desired grasp
        grasp_orient = self.calculate_grasp_orientation(approach, object_orient)

        return {
            'position': position.tolist(),
            'orientation': grasp_orient,
            'approach_direction': approach.tolist()
        }

    def calculate_grasp_orientation(self, approach_dir, object_orient):
        """
        Calculate appropriate grasp orientation based on approach direction
        """
        # For a simple parallel jaw gripper, align gripper fingers
        # perpendicular to the approach direction
        approach = np.array(approach_dir)

        # Choose a reference up vector (typically z-axis)
        up = np.array([0, 0, 1])

        # Calculate gripper orientation
        # This is a simplified approach - in practice, more sophisticated methods are used
        z_axis = -approach  # Approach direction is negative z for gripper
        y_axis = np.cross(up, z_axis)
        if np.linalg.norm(y_axis) < 1e-6:  # Parallel case
            y_axis = np.array([1, 0, 0])
        y_axis = y_axis / np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)

        # Create rotation matrix
        rot_matrix = np.column_stack([x_axis, y_axis, z_axis])

        # Convert to quaternion
        rot = R.from_matrix(rot_matrix)
        return rot.as_quat().tolist()

    def evaluate_grasp_quality(self, grasp, object_info, robot_state):
        """
        Evaluate the quality of a grasp based on multiple factors
        """
        stability_score = self.evaluate_stability(grasp, object_info)
        accessibility_score = self.evaluate_accessibility(grasp, robot_state)
        safety_score = self.evaluate_safety(grasp, robot_state)
        efficiency_score = self.evaluate_efficiency(grasp, object_info)

        # Weighted combination of scores
        total_score = (
            stability_score * self.quality_weights['stability'] +
            accessibility_score * self.quality_weights['accessibility'] +
            safety_score * self.quality_weights['safety'] +
            efficiency_score * self.quality_weights['efficiency']
        )

        return total_score

    def evaluate_stability(self, grasp, object_info):
        """
        Evaluate grasp stability based on object properties
        """
        # Consider object weight, center of mass, grasp type
        weight = object_info.get('weight', 0.5)  # default 0.5kg
        grasp_type = self.select_grasp_type(object_info)

        # Stability increases with appropriate grasp type for object
        if grasp_type == object_info.get('preferred_grasp_type'):
            stability = 0.9
        else:
            stability = 0.7

        # Adjust for object weight
        if weight > 1.0:  # Heavy object
            stability *= 0.8
        elif weight < 0.1:  # Very light object
            stability *= 0.9

        return stability

    def evaluate_accessibility(self, grasp, robot_state):
        """
        Evaluate if the grasp is accessible given robot configuration
        """
        # Check if the target pose is within robot's reach
        target_pos = np.array(grasp['position'])
        robot_pos = np.array(robot_state.get('position', [0, 0, 0]))

        distance = np.linalg.norm(target_pos - robot_pos)
        reach = robot_state.get('manipulation_reach', 1.0)  # default reach

        if distance > reach:
            return 0.1  # Very poor accessibility
        elif distance > reach * 0.9:
            return 0.4  # Poor accessibility
        else:
            return 0.9  # Good accessibility

    def evaluate_safety(self, grasp, robot_state):
        """
        Evaluate safety of the grasp considering environment
        """
        # Check for obstacles near the grasp position
        # This would interface with the robot's obstacle map
        obstacles = robot_state.get('obstacles', [])
        grasp_pos = np.array(grasp['position'])

        safety_score = 1.0
        for obstacle in obstacles:
            obs_pos = np.array(obstacle['position'])
            distance = np.linalg.norm(grasp_pos - obs_pos)
            if distance < 0.1:  # Very close to obstacle
                safety_score *= 0.3
            elif distance < 0.3:  # Close to obstacle
                safety_score *= 0.7

        return safety_score

    def evaluate_efficiency(self, grasp, object_info):
        """
        Evaluate the efficiency of the grasp
        """
        # Consider grasp stability, required force, etc.
        return 0.8  # Default efficiency

    def select_grasp_type(self, object_info):
        """
        Select appropriate grasp type based on object properties
        """
        preferred = object_info.get('preferred_grasp_type')
        if preferred:
            return preferred

        # Default selection based on object dimensions
        dims = object_info['dimensions']
        if dims[2] > 0.15:  # Tall object
            return 'power'
        elif dims[0] < 0.08 and dims[1] < 0.08:  # Small object
            return 'pinch'
        else:
            return 'power'
```

### Manipulation Control with Visual Feedback

Implementing closed-loop control for precise manipulation:

```python
class VisualFeedbackController:
    """
    Closed-loop controller for vision-guided manipulation
    """
    def __init__(self):
        # PID controller parameters for position control
        self.position_pid = {
            'kp': 2.0,  # Proportional gain
            'ki': 0.1,  # Integral gain
            'kd': 0.05  # Derivative gain
        }

        # PID controller parameters for orientation control
        self.orientation_pid = {
            'kp': 5.0,
            'ki': 0.2,
            'kd': 0.1
        }

        # Error tolerances
        self.position_tolerance = 0.005  # 5mm
        self.orientation_tolerance = 0.05  # 0.05 rad (~3 degrees)

        # Control loop parameters
        self.control_frequency = 100  # Hz
        self.max_velocity = 0.1  # m/s
        self.max_angular_velocity = 0.5  # rad/s

    def execute_grasp_with_feedback(self, target_grasp, vision_system):
        """
        Execute grasp with continuous visual feedback
        """
        # Initialize PID controllers
        pos_error_int = np.zeros(3)
        pos_error_prev = np.zeros(3)
        orient_error_int = np.zeros(3)
        orient_error_prev = np.zeros(3)

        start_time = time.time()
        max_time = 10.0  # Maximum execution time

        while time.time() - start_time < max_time:
            # Get current robot pose
            current_pose = self.get_robot_pose()

            # Get updated object pose from vision
            updated_object_pose = self.update_object_pose(vision_system)

            # Calculate pose errors
            pos_error = np.array(target_grasp['position']) - np.array(current_pose['position'])
            orient_error = self.calculate_orientation_error(
                target_grasp['orientation'],
                current_pose['orientation']
            )

            # Check if we're close enough
            pos_error_norm = np.linalg.norm(pos_error)
            orient_error_norm = np.linalg.norm(orient_error)

            if (pos_error_norm < self.position_tolerance and
                orient_error_norm < self.orientation_tolerance):
                print("Target pose reached!")
                break

            # Calculate control outputs using PID
            pos_control = self.calculate_pid_control(
                pos_error, pos_error_int, pos_error_prev,
                self.position_pid
            )
            orient_control = self.calculate_pid_control(
                orient_error, orient_error_int, orient_error_prev,
                self.orientation_pid
            )

            # Limit control outputs
            pos_control = np.clip(pos_control, -self.max_velocity, self.max_velocity)
            orient_control = np.clip(orient_control, -self.max_angular_velocity, self.max_angular_velocity)

            # Send control commands to robot
            self.send_control_commands(pos_control, orient_control)

            # Update error integrals and previous errors
            pos_error_int += pos_error * (1.0 / self.control_frequency)
            pos_error_prev = pos_error.copy()

            orient_error_int += orient_error * (1.0 / self.control_frequency)
            orient_error_prev = orient_error.copy()

            # Wait for next control cycle
            time.sleep(1.0 / self.control_frequency)

    def calculate_orientation_error(self, target_quat, current_quat):
        """
        Calculate orientation error as a 3D vector
        """
        # Convert quaternions to rotation vectors
        target_rot = R.from_quat(target_quat)
        current_rot = R.from_quat(current_quat)

        # Calculate relative rotation
        relative_rot = target_rot * current_rot.inv()

        # Convert to rotation vector (axis-angle representation)
        rot_vec = relative_rot.as_rotvec()

        return rot_vec

    def calculate_pid_control(self, error, error_int, error_prev, pid_params):
        """
        Calculate PID control output
        """
        p_term = pid_params['kp'] * error
        i_term = pid_params['ki'] * error_int
        d_term = pid_params['kd'] * (error - error_prev)

        control_output = p_term + i_term + d_term
        return control_output

    def get_robot_pose(self):
        """
        Get current robot end-effector pose
        In practice, this would interface with robot state
        """
        # This would be implemented based on your robot interface
        # For example, using ROS topics or robot API
        return {
            'position': [0.0, 0.0, 0.0],
            'orientation': [0.0, 0.0, 0.0, 1.0]  # quaternion
        }

    def update_object_pose(self, vision_system):
        """
        Update object pose estimate from vision system
        """
        # This would call the vision system to get updated pose
        # For demonstration, returning a fixed pose
        return {
            'position': [0.5, 0.2, 0.1],
            'orientation': [0.0, 0.0, 0.0, 1.0]
        }

    def send_control_commands(self, pos_control, orient_control):
        """
        Send control commands to robot
        """
        # This would interface with the robot's control system
        # For example, sending joint velocities or Cartesian velocities
        pass
```

### Isaac ROS Integration for Accelerated Vision

Integrating with Isaac ROS for accelerated vision processing:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from vision_msgs.msg import Detection2DArray, ObjectHypothesisWithPose
from std_msgs.msg import String

class IsaacVisionManipulationNode(Node):
    """
    ROS 2 node integrating Isaac ROS vision with manipulation
    """
    def __init__(self):
        super().__init__('isaac_vision_manipulation')

        # Initialize Isaac ROS vision components
        self.vision_system = ManipulationVisionSystem()
        self.grasp_planner = GraspPlanner()
        self.controller = VisualFeedbackController()

        # Subscribe to camera data
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera_info',
            self.camera_info_callback,
            10
        )

        # Subscribe to manipulation commands
        self.manip_cmd_sub = self.create_subscription(
            String,
            '/manipulation/command',
            self.manipulation_command_callback,
            10
        )

        # Publishers
        self.detection_pub = self.create_publisher(
            Detection2DArray,
            '/manipulation/detections',
            10
        )

        self.grasp_plan_pub = self.create_publisher(
            String,  # For now, using String to publish grasp plans as JSON
            '/manipulation/grasp_plan',
            10
        )

        self.target_pose_pub = self.create_publisher(
            PoseStamped,
            '/manipulation/target_pose',
            10
        )

        # Store camera info
        self.camera_info = None

        self.get_logger().info('Isaac Vision Manipulation Node initialized')

    def camera_info_callback(self, msg):
        """Update camera information"""
        self.camera_info = msg

        # Update camera matrix in vision system
        self.vision_system.camera_matrix = np.array(msg.k).reshape(3, 3)

    def image_callback(self, msg):
        """Process incoming image for manipulation"""
        try:
            # Convert ROS image to OpenCV format
            from cv_bridge import CvBridge
            bridge = CvBridge()
            cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")

            # Detect objects
            detections = self.vision_system.detect_objects(cv_image)

            # Process each detection for manipulation
            for detection in detections:
                object_class = detection['class']
                bbox = detection['bbox']

                # Estimate object pose
                object_pose = self.vision_system.estimate_object_pose(
                    cv_image, object_class, bbox
                )

                # Plan grasps for the object
                robot_state = self.get_robot_state()
                grasp_plans = self.grasp_planner.plan_grasps(
                    object_pose, object_class, robot_state
                )

                # Publish the best grasp plan
                if grasp_plans:
                    best_grasp = grasp_plans[0]  # Highest quality grasp
                    self.publish_grasp_plan(best_grasp, object_class)

            # Publish detections for visualization
            self.publish_detections(detections, msg.header)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def manipulation_command_callback(self, msg):
        """Handle manipulation commands"""
        command = msg.data
        self.get_logger().info(f'Received manipulation command: {command}')

        if command.startswith('grasp_'):
            # Extract object type from command
            object_type = command.split('_')[1]
            self.execute_grasp_for_object(object_type)

    def execute_grasp_for_object(self, object_type):
        """Execute grasp for a specific object type"""
        # This would involve:
        # 1. Looking for the object in the current scene
        # 2. Planning the best grasp
        # 3. Executing the grasp with visual feedback
        pass

    def get_robot_state(self):
        """Get current robot state"""
        # In practice, this would get robot state from TF, joint states, etc.
        return {
            'position': [0, 0, 0],
            'orientation': [0, 0, 0, 1],
            'manipulation_reach': 1.0,
            'obstacles': []
        }

    def publish_grasp_plan(self, grasp_plan, object_class):
        """Publish grasp plan for execution"""
        import json

        grasp_msg = String()
        grasp_data = {
            'object_class': object_class,
            'grasp_pose': grasp_plan['pose'],
            'grasp_type': grasp_plan['type'],
            'quality': grasp_plan['quality']
        }
        grasp_msg.data = json.dumps(grasp_data)

        self.grasp_plan_pub.publish(grasp_msg)

    def publish_detections(self, detections, header):
        """Publish object detections"""
        detection_msg = Detection2DArray()
        detection_msg.header = header

        for detection in detections:
            # Create detection message
            detection_2d = Detection2D()

            # Set bounding box
            x1, y1, x2, y2 = detection['bbox']
            detection_2d.bbox.center.x = (x1 + x2) / 2
            detection_2d.bbox.center.y = (y1 + y2) / 2
            detection_2d.bbox.size_x = x2 - x1
            detection_2d.bbox.size_y = y2 - y1

            # Set classification
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = detection['class']
            hypothesis.hypothesis.score = detection['confidence']

            detection_2d.results.append(hypothesis)
            detection_msg.detections.append(detection_2d)

        self.detection_pub.publish(detection_msg)

def main(args=None):
    rclpy.init(args=args)

    vision_manip_node = IsaacVisionManipulationNode()

    try:
        rclpy.spin(vision_manip_node)
    except KeyboardInterrupt:
        pass
    finally:
        vision_manip_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Advanced Manipulation Strategies

Implementing advanced manipulation strategies for complex tasks:

```python
class AdvancedManipulationStrategies:
    """
    Advanced manipulation strategies for complex tasks
    """
    def __init__(self):
        self.reinforcement_learning_agent = self.initialize_rl_agent()
        self.tactile_feedback_handler = TactileFeedbackHandler()

    def plan_complex_manipulation(self, task_description, scene_state):
        """
        Plan complex manipulation tasks involving multiple objects and steps
        """
        # Decompose complex task into subtasks
        subtasks = self.decompose_task(task_description)

        # Plan each subtask considering the scene state
        manipulation_sequence = []

        for subtask in subtasks:
            plan = self.plan_subtask(subtask, scene_state)
            manipulation_sequence.append(plan)

            # Update scene state after each subtask
            scene_state = self.update_scene_state(scene_state, plan)

        return manipulation_sequence

    def decompose_task(self, task_description):
        """
        Decompose high-level task into manipulation subtasks
        """
        # Example: "Pour water from bottle to cup" ->
        # 1. Grasp bottle
        # 2. Move to above cup
        # 3. Tilt bottle
        # 4. Return to neutral position

        task_subtasks = {
            'pour': ['grasp_source', 'move_above_target', 'tilt', 'return'],
            'stack': ['grasp_object', 'move_above_stack', 'place', 'return'],
            'assemble': ['grasp_part1', 'move_to_assembly', 'align', 'connect', 'return']
        }

        task_type = task_description.split()[0].lower()
        return task_subtasks.get(task_type, ['grasp', 'move', 'manipulate', 'release'])

    def plan_subtask(self, subtask, scene_state):
        """
        Plan a specific manipulation subtask
        """
        if subtask == 'grasp_source':
            return self.plan_grasp_subtask(scene_state['source_object'])
        elif subtask == 'move_above_target':
            return self.plan_move_subtask(scene_state['target_object'], offset=[0, 0, 0.1])
        elif subtask == 'tilt':
            return self.plan_tilt_subtask()
        elif subtask == 'place':
            return self.plan_place_subtask(scene_state['target_location'])
        else:
            return self.plan_generic_manipulation_subtask(subtask, scene_state)

    def plan_grasp_subtask(self, object_info):
        """
        Plan grasping subtask
        """
        grasp_planner = GraspPlanner()
        robot_state = self.get_current_robot_state()

        grasp_options = grasp_planner.plan_grasps(
            object_info['pose'],
            object_info['class'],
            robot_state
        )

        # Select best grasp based on task requirements
        best_grasp = grasp_options[0] if grasp_options else None

        return {
            'action': 'grasp',
            'target_pose': best_grasp['pose'] if best_grasp else None,
            'grasp_type': best_grasp['type'] if best_grasp else 'power',
            'approach_vector': self.calculate_approach_vector(best_grasp)
        }

    def plan_move_subtask(self, target_object, offset=None):
        """
        Plan movement subtask with offset from target
        """
        if offset is None:
            offset = [0, 0, 0.1]  # Default 10cm above target

        target_pos = target_object['pose']['position']
        move_target = [
            target_pos[0] + offset[0],
            target_pos[1] + offset[1],
            target_pos[2] + offset[2]
        ]

        return {
            'action': 'move',
            'target_pose': {
                'position': move_target,
                'orientation': target_object['pose']['orientation']
            },
            'motion_type': 'cartesian'
        }

    def plan_tilt_subtask(self):
        """
        Plan tilting/rotation subtask
        """
        return {
            'action': 'rotate',
            'axis': [0, 1, 0],  # Rotate about y-axis
            'angle': 1.57,      # 90 degrees
            'speed': 0.1        # rad/s
        }

    def execute_with_tactile_feedback(self, manipulation_plan):
        """
        Execute manipulation with tactile feedback for fine control
        """
        for step in manipulation_plan:
            if step['action'] == 'grasp':
                # Execute grasp with tactile feedback
                self.execute_grasp_with_tactile(step)
            elif step['action'] == 'move':
                # Execute move with visual feedback
                self.execute_move_with_visual(step)
            elif step['action'] == 'rotate':
                # Execute rotation with tactile feedback
                self.execute_rotation_with_tactile(step)

    def execute_grasp_with_tactile(self, grasp_step):
        """
        Execute grasp with tactile feedback for precise control
        """
        # Move to pre-grasp position
        pre_grasp_pos = self.calculate_pre_grasp_position(grasp_step['target_pose'])
        self.move_to_position(pre_grasp_pos)

        # Approach with tactile monitoring
        contact_force_threshold = 2.0  # Newtons
        while True:
            # Move slowly toward object
            self.move_along_approach_vector(grasp_step['approach_vector'], step_size=0.001)

            # Check tactile sensors
            tactile_data = self.tactile_feedback_handler.get_tactile_data()
            if self.detect_contact(tactile_data, contact_force_threshold):
                break

        # Close gripper with force control
        self.close_gripper_with_force_control(grasp_step['grasp_type'])

    def execute_move_with_visual(self, move_step):
        """
        Execute move with visual feedback
        """
        controller = VisualFeedbackController()
        controller.execute_grasp_with_feedback(move_step['target_pose'], self.vision_system)

    def execute_rotation_with_tactile(self, rotate_step):
        """
        Execute rotation with tactile feedback
        """
        # Rotate slowly while monitoring tactile sensors
        # Stop if excessive force is detected
        pass

    def calculate_pre_grasp_position(self, target_pose):
        """
        Calculate pre-grasp position (slightly away from target)
        """
        approach_vec = target_pose['approach_direction']
        offset_distance = 0.05  # 5cm offset

        pre_grasp_pos = [
            target_pose['position'][0] - approach_vec[0] * offset_distance,
            target_pose['position'][1] - approach_vec[1] * offset_distance,
            target_pose['position'][2] - approach_vec[2] * offset_distance
        ]

        return {
            'position': pre_grasp_pos,
            'orientation': target_pose['orientation']
        }

    def detect_contact(self, tactile_data, threshold):
        """
        Detect contact based on tactile sensor data
        """
        # Check if any tactile sensor exceeds threshold
        for sensor_force in tactile_data:
            if sensor_force > threshold:
                return True
        return False

    def close_gripper_with_force_control(self, grasp_type):
        """
        Close gripper with appropriate force for the grasp type
        """
        grasp_params = self.grasp_planner.grasp_types[grasp_type]
        target_force = grasp_params['force']

        # Implement force-controlled grasping
        # This would interface with the gripper controller
        pass

    def get_current_robot_state(self):
        """
        Get current robot state including joint positions, end-effector pose, etc.
        """
        # This would interface with the robot's state publisher
        return {
            'position': [0, 0, 0],
            'orientation': [0, 0, 0, 1],
            'joint_positions': [],
            'joint_velocities': [],
            'manipulation_reach': 1.0,
            'obstacles': []
        }

class TactileFeedbackHandler:
    """
    Handle tactile feedback for manipulation
    """
    def __init__(self):
        # Initialize tactile sensor interface
        pass

    def get_tactile_data(self):
        """
        Get current tactile sensor readings
        """
        # This would interface with tactile sensors
        # Return force/torque readings from tactile sensors
        return [0.0, 0.0, 0.0, 0.0]  # Example: 4 sensor readings
```

### Safety Considerations

Implementing safety measures for vision-guided manipulation:

```python
class SafeManipulationController:
    """
    Safe manipulation controller with safety checks and emergency procedures
    """
    def __init__(self):
        self.safety_thresholds = {
            'force_limit': 50.0,      # Maximum force (Newtons)
            'velocity_limit': 0.2,    # Maximum velocity (m/s)
            'acceleration_limit': 1.0, # Maximum acceleration (m/s²)
            'collision_distance': 0.05 # Minimum distance to obstacles (m)
        }

        self.emergency_stop_active = False
        self.safety_monitor = SafetyMonitor()

    def execute_safe_manipulation(self, manipulation_plan, scene_state):
        """
        Execute manipulation with safety monitoring
        """
        for step in manipulation_plan:
            # Check safety before executing each step
            if not self.is_safe_to_execute(step, scene_state):
                self.trigger_emergency_stop()
                return False

            # Execute step with safety monitoring
            success = self.execute_step_with_monitoring(step)

            if not success:
                return False

        return True

    def is_safe_to_execute(self, step, scene_state):
        """
        Check if it's safe to execute a manipulation step
        """
        # Check force limits
        if self.would_exceed_force_limits(step):
            return False

        # Check collision avoidance
        if self.detect_potential_collision(step, scene_state):
            return False

        # Check velocity limits
        if self.would_exceed_velocity_limits(step):
            return False

        return True

    def would_exceed_force_limits(self, step):
        """
        Check if step would exceed force limits
        """
        # This would check planned forces against limits
        return False  # Placeholder

    def detect_potential_collision(self, step, scene_state):
        """
        Detect potential collisions with obstacles
        """
        # Check planned path against obstacle map
        return False  # Placeholder

    def would_exceed_velocity_limits(self, step):
        """
        Check if step would exceed velocity limits
        """
        return False  # Placeholder

    def execute_step_with_monitoring(self, step):
        """
        Execute manipulation step with real-time safety monitoring
        """
        # Start safety monitoring thread
        monitoring_thread = threading.Thread(target=self.monitor_safety)
        monitoring_thread.start()

        try:
            # Execute the step
            if step['action'] == 'grasp':
                self.execute_grasp_with_safety(step)
            elif step['action'] == 'move':
                self.execute_move_with_safety(step)
            # Add other action types as needed

            # Stop monitoring
            self.safety_monitor.stop_monitoring()
            monitoring_thread.join()

            return True

        except Exception as e:
            self.get_logger().error(f'Safety error during execution: {e}')
            self.trigger_emergency_stop()
            return False

    def trigger_emergency_stop(self):
        """
        Trigger emergency stop for safety
        """
        self.emergency_stop_active = True
        # Send emergency stop command to robot
        self.send_emergency_stop_command()

        # Log safety event
        self.log_safety_event('EMERGENCY_STOP', 'Safety threshold exceeded')

    def send_emergency_stop_command(self):
        """
        Send emergency stop command to robot
        """
        # This would send an emergency stop signal to the robot controller
        pass

    def log_safety_event(self, event_type, description):
        """
        Log safety events for analysis
        """
        # Log to file or database
        pass

class SafetyMonitor:
    """
    Continuous safety monitoring during manipulation
    """
    def __init__(self):
        self.monitoring_active = False
        self.safety_violations = []
        self.monitoring_thread = None

    def start_monitoring(self):
        """
        Start safety monitoring
        """
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self.monitoring_loop)
        self.monitoring_thread.start()

    def stop_monitoring(self):
        """
        Stop safety monitoring
        """
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join()

    def monitoring_loop(self):
        """
        Continuous monitoring loop
        """
        while self.monitoring_active:
            # Check various safety parameters
            self.check_force_sensors()
            self.check_collision_sensors()
            self.check_joint_limits()
            self.check_velocity_limits()

            time.sleep(0.01)  # 100 Hz monitoring

    def check_force_sensors(self):
        """
        Check force/torque sensors for excessive forces
        """
        # Read force/torque sensor values
        # Compare against safety thresholds
        pass

    def check_collision_sensors(self):
        """
        Check proximity/collision sensors
        """
        # Check distance to obstacles
        # Trigger warning if too close
        pass

    def check_joint_limits(self):
        """
        Check joint position, velocity, and torque limits
        """
        # Monitor joint states
        # Ensure within safe operating range
        pass

    def check_velocity_limits(self):
        """
        Check for excessive velocities
        """
        # Monitor joint and Cartesian velocities
        # Ensure within safe limits
        pass
```

### Summary

Vision-guided manipulation is a critical capability for humanoid robots, enabling them to interact with objects in their environment based on visual feedback. This chapter covered:

- Computer vision techniques for object detection and pose estimation
- Grasp planning algorithms for different object types
- Closed-loop control with visual feedback
- Isaac ROS integration for accelerated vision processing
- Advanced manipulation strategies for complex tasks
- Safety considerations and monitoring

In the next chapter, we'll explore end-to-end system integration, bringing together all the components we've developed into a complete humanoid robot system.