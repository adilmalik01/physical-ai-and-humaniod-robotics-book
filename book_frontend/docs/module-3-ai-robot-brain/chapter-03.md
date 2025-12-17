---
sidebar_position: 3
---

# Chapter 03: Isaac ROS for Accelerated Perception

## Leveraging Hardware Acceleration for Real-Time Perception

In this chapter, we'll explore Isaac ROS packages that provide hardware-accelerated perception capabilities for humanoid robots. Isaac ROS leverages NVIDIA's GPU computing platform to deliver high-performance perception algorithms that are essential for real-time humanoid robot operation.

### Understanding Isaac ROS Architecture

Isaac ROS packages are designed to provide accelerated versions of common ROS 2 perception tasks:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Raw Sensors   │ -> │ Isaac ROS       │ -> │  Accelerated    │
│   (Camera,      │    │   Processing    │    │  Perception     │
│   LiDAR, IMU)   │    │   Nodes         │    │  Output         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
       │                        │                        │
       v                        v                        v
   Sensor Data         GPU-Accelerated        Real-time Perception
   Processing          Algorithms             Results
```

### Isaac ROS Package Ecosystem

Isaac ROS provides several specialized packages:

#### Isaac ROS Image Pipeline
```bash
# Isaac ROS image pipeline for accelerated image processing
ros2 run isaac_ros_image_pipeline image_pipeline_node
```

Key features:
- Hardware-accelerated image rectification
- Color conversion and scaling
- Image compression/decompression
- Stereo rectification

#### Isaac ROS AprilTag
```bash
# Accelerated AprilTag detection for precise localization
ros2 run isaac_ros_apriltag apriltag_node
```

Key features:
- GPU-accelerated tag detection
- Sub-pixel corner refinement
- Multi-tag tracking
- Pose estimation

#### Isaac ROS DNN Inference
```bash
# Hardware-accelerated deep learning inference
ros2 run isaac_ros_dnn_inference dnn_inference_node
```

Key features:
- TensorRT optimization
- INT8 and FP16 quantization
- Batch processing
- Multiple model formats support

#### Isaac ROS Visual SLAM
```bash
# Accelerated visual SLAM
ros2 run isaac_ros_visual_slam visual_slam_node
```

Key features:
- Real-time pose estimation
- Map building
- Loop closure detection
- GPU-accelerated feature extraction

### Installing Isaac ROS

To install Isaac ROS packages:

```bash
# Add NVIDIA package repository
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt update

# Install Isaac ROS packages
sudo apt install ros-humble-isaac-ros-common
sudo apt install ros-humble-isaac-ros-apriltag
sudo apt install ros-humble-isaac-ros-dnn-inference
sudo apt install ros-humble-isaac-ros-image-pipeline
sudo apt install ros-humble-isaac-ros-visual-slam
sudo apt install ros-humble-isaac-ros-pointcloud-utils
```

### Isaac ROS Image Pipeline

The image pipeline accelerates common image processing tasks:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np

class IsaacImageProcessor(Node):
    def __init__(self):
        super().__init__('isaac_image_processor')

        self.cv_bridge = CvBridge()

        # Subscribe to raw camera feed
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        # Subscribe to camera info
        self.info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera_info',
            self.info_callback,
            10
        )

        # Publishers for processed images
        self.rectified_pub = self.create_publisher(
            Image,
            '/camera/image_rectified',
            10
        )

        self.processed_pub = self.create_publisher(
            Image,
            '/camera/image_processed',
            10
        )

        # Store camera parameters
        self.camera_matrix = None
        self.distortion_coeffs = None

        self.get_logger().info('Isaac Image Processor initialized')

    def info_callback(self, msg):
        """Receive camera calibration parameters"""
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.distortion_coeffs = np.array(msg.d)

    def image_callback(self, msg):
        """Process incoming image using Isaac-accelerated methods"""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")

            # Rectify image if camera parameters are available
            if self.camera_matrix is not None and self.distortion_coeffs is not None:
                rectified_image = self.rectify_image(
                    cv_image,
                    self.camera_matrix,
                    self.distortion_coeffs
                )

                # Publish rectified image
                rectified_msg = self.cv_bridge.cv2_to_imgmsg(rectified_image, "bgr8")
                rectified_msg.header = msg.header
                self.rectified_pub.publish(rectified_msg)

            # Apply additional processing
            processed_image = self.apply_isaac_processing(cv_image)

            # Publish processed image
            processed_msg = self.cv_bridge.cv2_to_imgmsg(processed_image, "bgr8")
            processed_msg.header = msg.header
            self.processed_pub.publish(processed_msg)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def rectify_image(self, image, camera_matrix, dist_coeffs):
        """Apply camera rectification (in practice, use Isaac ROS node)"""
        # In real implementation, this would be handled by Isaac ROS rectification node
        # For demonstration, using OpenCV
        h, w = image.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), 1, (w, h)
        )
        rectified = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)
        return rectified

    def apply_isaac_processing(self, image):
        """Apply Isaac-accelerated image processing"""
        # Placeholder for Isaac-accelerated processing
        # In practice, this would use Isaac ROS nodes for:
        # - Hardware-accelerated filtering
        # - Color space conversion
        # - Image scaling
        # - Noise reduction

        # Example: Apply hardware-accelerated bilateral filter
        processed = cv2.bilateralFilter(image, 9, 75, 75)
        return processed

def main(args=None):
    rclpy.init(args=args)
    node = IsaacImageProcessor()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Isaac ROS DNN Inference

Accelerated deep learning inference for humanoid perception:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, ObjectHypothesisWithPose
from cv_bridge import CvBridge
import numpy as np
import torch
import torchvision.transforms as transforms

class IsaacDNNInference(Node):
    def __init__(self):
        super().__init__('isaac_dnn_inference')

        self.cv_bridge = CvBridge()

        # Subscribe to camera feed
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        # Publisher for detections
        self.detection_pub = self.create_publisher(
            Detection2DArray,
            '/detections',
            10
        )

        # Load pre-trained model (in practice, use TensorRT optimized model)
        self.load_model()

        # Preprocessing transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((640, 640)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        self.get_logger().info('Isaac DNN Inference node initialized')

    def load_model(self):
        """Load and optimize model for inference"""
        # In practice, load TensorRT optimized model
        # For demonstration, using a placeholder
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.model.eval()

        # Move model to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.get_logger().info('Model loaded on GPU')
        else:
            self.get_logger().warn('CUDA not available, running on CPU')

    def image_callback(self, msg):
        """Process image and run inference"""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")

            # Preprocess image
            input_tensor = self.preprocess_image(cv_image)

            # Run inference
            with torch.no_grad():
                if torch.cuda.is_available():
                    input_tensor = input_tensor.cuda()

                results = self.model(input_tensor)

            # Process results
            detections = self.process_detections(results, cv_image.shape, msg.header)

            # Publish detections
            self.detection_pub.publish(detections)

        except Exception as e:
            self.get_logger().error(f'DNN inference error: {e}')

    def preprocess_image(self, image):
        """Preprocess image for inference"""
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert to tensor and normalize
        input_tensor = self.transform(image_rgb).unsqueeze(0)

        return input_tensor

    def process_detections(self, results, image_shape, header):
        """Process inference results into ROS messages"""
        detections_msg = Detection2DArray()
        detections_msg.header = header

        # Extract detections from YOLOv5 results
        # In practice, this would use Isaac ROS DNN Inference output format
        if hasattr(results, 'xyxy'):
            for detection in results.xyxy[0]:  # xyxy format: [x1, y1, x2, y2, conf, class]
                if detection[4] > 0.5:  # Confidence threshold
                    detection_2d = Detection2D()

                    # Set bounding box
                    bbox = detection[:4].cpu().numpy()
                    detection_2d.bbox.size_x = bbox[2] - bbox[0]
                    detection_2d.bbox.size_y = bbox[3] - bbox[1]
                    detection_2d.bbox.center.x = (bbox[0] + bbox[2]) / 2
                    detection_2d.bbox.center.y = (bbox[1] + bbox[3]) / 2

                    # Set class and confidence
                    hypothesis = ObjectHypothesisWithPose()
                    hypothesis.hypothesis.class_id = str(int(detection[5]))
                    hypothesis.hypothesis.score = float(detection[4])

                    detection_2d.results.append(hypothesis)
                    detections_msg.detections.append(detection_2d)

        return detections_msg

def main(args=None):
    rclpy.init(args=args)
    node = IsaacDNNInference()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Isaac ROS AprilTag Detection

For precise localization and calibration:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import MarkerArray, Marker
from cv_bridge import CvBridge
import numpy as np

class IsaacAprilTagDetector(Node):
    def __init__(self):
        super().__init__('isaac_april_tag_detector')

        self.cv_bridge = CvBridge()

        # Subscribe to camera data
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera_info',
            self.info_callback,
            10
        )

        # Publishers
        self.pose_pub = self.create_publisher(
            PoseStamped,
            '/tag_pose',
            10
        )

        self.marker_pub = self.create_publisher(
            MarkerArray,
            '/tag_markers',
            10
        )

        # Camera parameters
        self.camera_matrix = None
        self.distortion_coeffs = None

        # Tag size (in meters)
        self.tag_size = 0.16  # 16cm tag

        self.get_logger().info('Isaac AprilTag Detector initialized')

    def info_callback(self, msg):
        """Receive camera calibration parameters"""
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.distortion_coeffs = np.array(msg.d)

    def image_callback(self, msg):
        """Process image to detect AprilTags"""
        try:
            if self.camera_matrix is None:
                return  # Wait for camera info

            # Convert ROS image to OpenCV
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "mono8")

            # In practice, use Isaac ROS AprilTag node
            # For demonstration, using a placeholder detection function
            tags = self.detect_april_tags(cv_image)

            if tags:
                # Process each detected tag
                markers = MarkerArray()

                for i, tag in enumerate(tags):
                    # Calculate pose (in practice, Isaac ROS does this)
                    pose = self.calculate_tag_pose(tag)

                    # Publish pose
                    pose_msg = PoseStamped()
                    pose_msg.header = msg.header
                    pose_msg.pose = pose
                    self.pose_pub.publish(pose_msg)

                    # Create visualization marker
                    marker = self.create_marker(tag, msg.header, i)
                    markers.markers.append(marker)

                self.marker_pub.publish(markers)

        except Exception as e:
            self.get_logger().error(f'AprilTag detection error: {e}')

    def detect_april_tags(self, image):
        """Detect AprilTags in image (placeholder - use Isaac ROS node)"""
        # In real implementation, subscribe to Isaac ROS AprilTag output
        # This is a simplified placeholder
        return []  # Return list of detected tags

    def calculate_tag_pose(self, tag):
        """Calculate 3D pose of tag (placeholder)"""
        # In practice, this is done by Isaac ROS AprilTag node
        # using camera matrix, distortion coefficients, and tag size
        from geometry_msgs.msg import Pose
        return Pose()  # Placeholder

    def create_marker(self, tag, header, id_num):
        """Create visualization marker for tag"""
        marker = Marker()
        marker.header = header
        marker.ns = "april_tags"
        marker.id = id_num
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        # Set position and orientation (from pose)
        marker.pose = self.calculate_tag_pose(tag)
        marker.scale.x = self.tag_size
        marker.scale.y = self.tag_size
        marker.scale.z = 0.01
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0

        return marker

def main(args=None):
    rclpy.init(args=args)
    node = IsaacAprilTagDetector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Isaac ROS Visual SLAM

For localization and mapping:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
import numpy as np

class IsaacVisualSLAM(Node):
    def __init__(self):
        super().__init__('isaac_visual_slam')

        # TF broadcaster for pose publishing
        self.tf_broadcaster = TransformBroadcaster(self)

        # Subscribe to camera and IMU data
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )

        # Publishers
        self.odom_pub = self.create_publisher(
            Odometry,
            '/visual_odom',
            10
        )

        self.pose_pub = self.create_publisher(
            PoseStamped,
            '/visual_pose',
            10
        )

        # Store pose state
        self.current_pose = np.eye(4)  # 4x4 transformation matrix
        self.prev_image = None

        self.get_logger().info('Isaac Visual SLAM initialized')

    def image_callback(self, msg):
        """Process image for visual SLAM"""
        try:
            # In practice, use Isaac ROS Visual SLAM node
            # For demonstration, using a simplified approach

            # Process visual features and update pose
            pose_update = self.process_visual_features(msg)

            if pose_update is not None:
                # Update current pose
                self.current_pose = self.current_pose @ pose_update

                # Publish odometry
                self.publish_odometry(msg.header)

                # Broadcast transform
                self.broadcast_transform(msg.header)

        except Exception as e:
            self.get_logger().error(f'Visual SLAM error: {e}')

    def imu_callback(self, msg):
        """Process IMU data to aid visual SLAM"""
        # In practice, Isaac ROS Visual SLAM uses IMU for better pose estimation
        # For demonstration, storing IMU data
        pass

    def process_visual_features(self, image_msg):
        """Process visual features for pose estimation (placeholder)"""
        # In real implementation, this would interface with Isaac ROS Visual SLAM
        # which provides hardware-accelerated feature detection and tracking
        return np.eye(4)  # Identity matrix as placeholder

    def publish_odometry(self, header):
        """Publish odometry message"""
        odom = Odometry()
        odom.header = header
        odom.child_frame_id = "base_link"

        # Convert transformation matrix to pose
        position = self.current_pose[:3, 3]
        odom.pose.pose.position.x = position[0]
        odom.pose.pose.position.y = position[1]
        odom.pose.pose.position.z = position[2]

        # Convert rotation matrix to quaternion
        rotation_matrix = self.current_pose[:3, :3]
        quat = self.rotation_matrix_to_quaternion(rotation_matrix)
        odom.pose.pose.orientation.w = quat[0]
        odom.pose.pose.orientation.x = quat[1]
        odom.pose.pose.orientation.y = quat[2]
        odom.pose.pose.orientation.z = quat[3]

        self.odom_pub.publish(odom)

    def broadcast_transform(self, header):
        """Broadcast transform to TF tree"""
        t = TransformStamped()

        t.header.stamp = header.stamp
        t.header.frame_id = "map"
        t.child_frame_id = "base_link"

        position = self.current_pose[:3, 3]
        t.transform.translation.x = position[0]
        t.transform.translation.y = position[1]
        t.transform.translation.z = position[2]

        rotation_matrix = self.current_pose[:3, :3]
        quat = self.rotation_matrix_to_quaternion(rotation_matrix)
        t.transform.rotation.w = quat[0]
        t.transform.rotation.x = quat[1]
        t.transform.rotation.y = quat[2]
        t.transform.rotation.z = quat[3]

        self.tf_broadcaster.sendTransform(t)

    def rotation_matrix_to_quaternion(self, R):
        """Convert rotation matrix to quaternion"""
        # Using the algorithm from:
        # https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
        trace = np.trace(R)

        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2  # S=4*qw
            qw = 0.25 * s
            qx = (R[2, 1] - R[1, 2]) / s
            qy = (R[0, 2] - R[2, 0]) / s
            qz = (R[1, 0] - R[0, 1]) / s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # S=4*qx
                qw = (R[2, 1] - R[1, 2]) / s
                qx = 0.25 * s
                qy = (R[0, 1] + R[1, 0]) / s
                qz = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # S=4*qy
                qw = (R[0, 2] - R[2, 0]) / s
                qx = (R[0, 1] + R[1, 0]) / s
                qy = 0.25 * s
                qz = (R[1, 2] + R[2, 1]) / s
            else:
                s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # S=4*qz
                qw = (R[1, 0] - R[0, 1]) / s
                qx = (R[0, 2] + R[2, 0]) / s
                qy = (R[1, 2] + R[2, 1]) / s
                qz = 0.25 * s

        return np.array([qw, qx, qy, qz])

def main(args=None):
    rclpy.init(args=args)
    node = IsaacVisualSLAM()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Isaac ROS Performance Optimization

To maximize performance with Isaac ROS:

#### GPU Memory Management
```python
import torch

class IsaacPerceptionOptimizer:
    def __init__(self):
        # Enable TensorRT optimization
        self.enable_tensorrt_optimization()

        # Optimize CUDA memory usage
        self.optimize_cuda_memory()

        # Set appropriate batch sizes
        self.set_optimal_batch_sizes()

    def enable_tensorrt_optimization(self):
        """Enable TensorRT optimization for inference"""
        try:
            import tensorrt as trt
            self.get_logger().info('TensorRT optimization enabled')
        except ImportError:
            self.get_logger().warn('TensorRT not available, using standard inference')

    def optimize_cuda_memory(self):
        """Optimize CUDA memory usage"""
        if torch.cuda.is_available():
            # Enable memory fraction
            torch.cuda.set_per_process_memory_fraction(0.8)

            # Enable memory caching
            torch.cuda.empty_cache()

    def set_optimal_batch_sizes(self):
        """Set batch sizes based on available GPU memory"""
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory
            if total_memory > 10e9:  # >10GB
                self.batch_size = 8
            elif total_memory > 6e9:  # >6GB
                self.batch_size = 4
            else:
                self.batch_size = 2
```

#### Pipeline Optimization
```python
# Example launch file for optimized Isaac ROS pipeline
# launch/optimized_perception_pipeline.launch.py
from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

def generate_launch_description():
    """Create optimized Isaac ROS perception pipeline"""

    # Create composable container for better performance
    perception_container = ComposableNodeContainer(
        name='perception_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',  # Multi-threaded
        composable_node_descriptions=[
            ComposableNode(
                package='isaac_ros_image_pipeline',
                plugin='nvidia::isaac_ros::image_pipeline::RectifyNode',
                name='rectify_node',
                parameters=[{
                    'output_width': 640,
                    'output_height': 480,
                }],
                remappings=[
                    ('image_raw', '/camera/image_raw'),
                    ('camera_info', '/camera/camera_info'),
                    ('image_rect', '/camera/image_rect'),
                ]
            ),
            ComposableNode(
                package='isaac_ros_apriltag',
                plugin='nvidia::isaac_ros::apriltag::AprilTagNode',
                name='apriltag_node',
                parameters=[{
                    'family': 'tag36h11',
                    'max_hamming': 0,
                    'quad_decimate': 2.0,
                    'quad_sigma': 0.0,
                    'nthreads': 4,
                }],
                remappings=[
                    ('image', '/camera/image_rect'),
                    ('detections', '/apriltag_detections'),
                ]
            ),
            ComposableNode(
                package='isaac_ros_dnn_inference',
                plugin='nvidia::isaac_ros::dnn_inference::DNNInferenceNode',
                name='dnn_inference_node',
                parameters=[{
                    'model_path': '/path/to/model.plan',  # TensorRT model
                    'input_tensor_names': ['input'],
                    'input_tensor_formats': ['nitros_tensor_list_nchw_rgb_f32'],
                    'output_tensor_names': ['output'],
                    'output_tensor_formats': ['nitros_tensor_list_nhwc_rgb_f32'],
                    'tensorrt_engine_file_path': '/path/to/engine.plan',
                }],
                remappings=[
                    ('tensor_sub', '/camera/image_rect'),
                    ('tensor_pub', '/dnn_result'),
                ]
            ),
        ],
        output='screen',
    )

    return LaunchDescription([perception_container])
```

### Troubleshooting Isaac ROS Issues

Common issues and solutions:

#### Performance Issues
```bash
# Check GPU utilization
nvidia-smi

# Monitor memory usage
ros2 run rqt_plot rqt_plot

# Check for bottlenecks
ros2 topic hz /camera/image_raw
```

#### Installation Issues
```bash
# Verify CUDA installation
nvidia-smi
nvcc --version

# Check Isaac ROS packages
dpkg -l | grep isaac-ros

# Verify GPU compatibility
ros2 run isaac_ros_apriltag check_gpu_compatibility
```

### Best Practices for Isaac ROS

1. **Use Composable Nodes**: Combine related nodes in containers for better performance
2. **Optimize Memory**: Monitor and optimize GPU memory usage
3. **Batch Processing**: Use appropriate batch sizes for inference
4. **Pipeline Design**: Design efficient processing pipelines with minimal data copying
5. **Hardware Matching**: Match algorithms to available hardware capabilities

### Summary

In this chapter, we've covered:
- Isaac ROS architecture and package ecosystem
- Installation and setup of Isaac ROS packages
- Implementation of key perception nodes (image processing, DNN inference, AprilTag, Visual SLAM)
- Performance optimization techniques
- Best practices for Isaac ROS development

In the next chapter, we'll explore Navigation2 (Nav2) integration with Isaac tools for humanoid robot navigation.