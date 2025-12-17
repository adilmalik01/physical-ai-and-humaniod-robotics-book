---
sidebar_position: 2
---

# Chapter 02: Isaac Sim for Synthetic Data Generation

## Creating Training Data for AI Perception Systems

In this chapter, we'll explore how to use Isaac Sim for generating synthetic data to train AI perception systems for humanoid robots. Synthetic data generation is a powerful technique that allows us to create large, diverse, and perfectly annotated datasets without the need for expensive and time-consuming real-world data collection.

### Understanding Synthetic Data Generation

Synthetic data generation in Isaac Sim involves creating photorealistic virtual environments and using them to generate training data for AI models. This approach offers several advantages:

1. **Perfect Annotations**: Ground truth data is available for every frame
2. **Variety**: Easy to create diverse scenarios and edge cases
3. **Safety**: No risk to physical robots or humans
4. **Cost-Effective**: Faster and cheaper than real-world data collection
5. **Control**: Complete control over lighting, objects, and scenarios

### Isaac Sim Synthetic Data Pipeline

The synthetic data generation pipeline in Isaac Sim typically involves:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   3D Scenes     │ -> │  Rendering &    │ -> │  Synthetic      │
│   & Objects     │    │  Simulation     │    │  Datasets       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Setting Up Isaac Sim for Data Generation

To use Isaac Sim for synthetic data generation, you'll need to configure several components:

#### USD Scene Creation
Universal Scene Description (USD) is the format used by Isaac Sim for scenes:

```python
# Example of creating a USD stage for synthetic data generation
import omni
from pxr import Usd, UsdGeom, Gf
import carb

def create_synthetic_scene():
    """Create a USD stage for synthetic data generation"""
    # Create a new USD stage
    stage = Usd.Stage.CreateNew("synthetic_scene.usd")

    # Set up the default prim
    default_prim = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(default_prim.GetPrim())

    # Add ground plane
    ground_plane = UsdGeom.Mesh.Define(stage, "/World/GroundPlane")
    # Configure ground plane geometry and materials

    # Add lighting
    dome_light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome_light.CreateIntensityAttr(1000)

    # Add camera
    camera = UsdGeom.Camera.Define(stage, "/World/Camera")
    camera.GetPrim().CreateAttribute("focalLength", 24.0)

    # Save the stage
    stage.GetRootLayer().Save()

    return stage
```

#### Domain Randomization

Domain randomization helps bridge the sim-to-real gap by varying environmental parameters:

```python
import random
import numpy as np

class DomainRandomizer:
    def __init__(self):
        self.lighting_conditions = [
            "sunny", "overcast", "indoor", "evening"
        ]
        self.material_properties = {
            "albedo": (0.1, 1.0),
            "roughness": (0.0, 1.0),
            "metallic": (0.0, 1.0)
        }
        self.camera_parameters = {
            "fov": (30, 90),
            "position_variance": (0.1, 0.1, 0.1)
        }

    def randomize_lighting(self):
        """Randomize lighting conditions"""
        lighting_type = random.choice(self.lighting_conditions)

        if lighting_type == "sunny":
            intensity = random.uniform(800, 1200)
            color = (1.0, 0.9, 0.8)
        elif lighting_type == "overcast":
            intensity = random.uniform(400, 800)
            color = (0.8, 0.8, 1.0)
        elif lighting_type == "indoor":
            intensity = random.uniform(300, 600)
            color = (1.0, 0.9, 0.8)
        else:  # evening
            intensity = random.uniform(200, 400)
            color = (1.0, 0.6, 0.3)

        return intensity, color

    def randomize_materials(self, material_path):
        """Randomize material properties"""
        albedo = random.uniform(*self.material_properties["albedo"])
        roughness = random.uniform(*self.material_properties["roughness"])
        metallic = random.uniform(*self.material_properties["metallic"])

        return {
            "albedo": albedo,
            "roughness": roughness,
            "metallic": metallic
        }

    def randomize_camera(self):
        """Randomize camera parameters"""
        fov = random.uniform(*self.camera_parameters["fov"])
        position_variance = [
            random.uniform(-v, v) for v in self.camera_parameters["position_variance"]
        ]

        return {
            "fov": fov,
            "position_variance": position_variance
        }
```

### Isaac Sim Extensions for Data Generation

Isaac Sim provides several extensions for synthetic data generation:

#### Isaac ROS Bridge Extension
```python
# Example of using Isaac Sim extensions
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.synthetic_utils import SyntheticDataHelper

def setup_synthetic_data_collection():
    """Set up Isaac Sim for synthetic data collection"""
    # Initialize Isaac Sim world
    world = World(stage_units_in_meters=1.0)

    # Add robot to the stage
    add_reference_to_stage(
        usd_path="/path/to/robot.usd",
        prim_path="/World/Robot"
    )

    # Configure synthetic data helper
    sd_helper = SyntheticDataHelper()
    sd_helper.initialize(
        camera_paths=["/World/Camera"],
        rgb=True,
        depth=True,
        semantic_segmentation=True,
        instance_segmentation=True
    )

    return world, sd_helper
```

#### Synthetic Data Capture

```python
import numpy as np
from PIL import Image
import json
import os

class SyntheticDataCapture:
    def __init__(self, output_dir="synthetic_data"):
        self.output_dir = output_dir
        self.frame_counter = 0

        # Create output directories
        os.makedirs(f"{output_dir}/rgb", exist_ok=True)
        os.makedirs(f"{output_dir}/depth", exist_ok=True)
        os.makedirs(f"{output_dir}/labels", exist_ok=True)

    def capture_frame(self, rgb_image, depth_image, segmentation, metadata):
        """Capture and save a synthetic frame with annotations"""
        # Save RGB image
        rgb_pil = Image.fromarray(rgb_image)
        rgb_path = f"{self.output_dir}/rgb/frame_{self.frame_counter:06d}.png"
        rgb_pil.save(rgb_path)

        # Save depth image
        depth_pil = Image.fromarray(depth_image)
        depth_path = f"{self.output_dir}/depth/frame_{self.frame_counter:06d}.png"
        depth_pil.save(depth_path)

        # Save segmentation labels
        seg_pil = Image.fromarray(segmentation)
        seg_path = f"{self.output_dir}/labels/frame_{self.frame_counter:06d}.png"
        seg_pil.save(seg_path)

        # Save metadata
        metadata_path = f"{self.output_dir}/labels/frame_{self.frame_counter:06d}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)

        self.frame_counter += 1

        return {
            "rgb_path": rgb_path,
            "depth_path": depth_path,
            "seg_path": seg_path,
            "metadata_path": metadata_path
        }
```

### Object Detection Dataset Generation

For training object detection models, Isaac Sim can generate perfectly annotated bounding boxes:

```python
def generate_object_detection_data(world, sd_helper, capture_helper):
    """Generate synthetic data for object detection"""
    # Spawn objects randomly in the scene
    objects = spawn_random_objects(world)

    # Get camera pose and capture data
    camera_poses = get_camera_poses()

    for pose in camera_poses:
        # Move camera to new pose
        set_camera_pose(pose)

        # Step the physics simulation
        world.step(render=True)

        # Capture synthetic data
        rgb, depth, seg = sd_helper.get_data()

        # Generate bounding box annotations
        bboxes = generate_bounding_boxes(objects, seg)

        # Create metadata with annotations
        metadata = {
            "frame_id": capture_helper.frame_counter,
            "timestamp": world.current_time,
            "camera_pose": pose,
            "objects": bboxes
        }

        # Save the frame
        capture_helper.capture_frame(rgb, depth, seg, metadata)

def generate_bounding_boxes(objects, segmentation):
    """Generate bounding box annotations from segmentation"""
    bboxes = []

    for obj in objects:
        # Find pixels belonging to this object in segmentation
        obj_mask = (segmentation == obj.id)

        if np.any(obj_mask):
            # Get bounding box coordinates
            y_coords, x_coords = np.where(obj_mask)
            x_min, x_max = x_coords.min(), x_coords.max()
            y_min, y_max = y_coords.min(), y_coords.max()

            bbox = {
                "label": obj.label,
                "x_min": int(x_min),
                "y_min": int(y_min),
                "x_max": int(x_max),
                "y_max": int(y_max),
                "confidence": 1.0  # Perfect annotation
            }
            bboxes.append(bbox)

    return bboxes
```

### Semantic Segmentation Dataset

For semantic segmentation tasks:

```python
def generate_semantic_segmentation_data(world, sd_helper, capture_helper):
    """Generate synthetic data for semantic segmentation"""
    # Define semantic classes
    semantic_classes = {
        0: "background",
        1: "robot",
        2: "person",
        3: "table",
        4: "chair",
        5: "floor",
        6: "wall"
    }

    # Randomize scene configuration
    randomize_scene(world)

    for i in range(1000):  # Generate 1000 frames
        # Randomize lighting and materials
        randomize_environment(world)

        # Step simulation
        world.step(render=True)

        # Capture data
        rgb, depth, seg = sd_helper.get_data()

        # Create metadata
        metadata = {
            "frame_id": capture_helper.frame_counter,
            "semantic_classes": semantic_classes,
            "scene_config": get_scene_config(world)
        }

        # Save frame
        capture_helper.capture_frame(rgb, depth, seg, metadata)
```

### Humanoid-Specific Synthetic Scenarios

For humanoid robotics, create specific scenarios:

#### Human-Robot Interaction Scenarios
```python
def create_interaction_scenarios(world):
    """Create synthetic scenarios for human-robot interaction"""
    scenarios = [
        {
            "name": "object_handover",
            "description": "Human handing object to robot",
            "objects": ["mug", "box"],
            "humans": ["person_1"],
            "actions": ["approach", "grasp", "handover"]
        },
        {
            "name": "navigation_with_people",
            "description": "Robot navigating around humans",
            "objects": ["furniture"],
            "humans": ["person_1", "person_2", "person_3"],
            "actions": ["follow_path", "avoid_collision"]
        },
        {
            "name": "manipulation_in_clutter",
            "description": "Robot manipulating objects in cluttered environment",
            "objects": ["cups", "books", "boxes"],
            "humans": [],
            "actions": ["pick", "place", "avoid"]
        }
    ]

    return scenarios

def generate_interaction_data(scenario, world, sd_helper, capture_helper):
    """Generate synthetic data for human-robot interaction"""
    # Set up the scenario in Isaac Sim
    setup_scenario(world, scenario)

    # Record the interaction sequence
    for step in range(scenario["duration"]):
        # Execute interaction step
        execute_interaction_step(world, scenario, step)

        # Capture synthetic data
        rgb, depth, seg = sd_helper.get_data()

        # Generate annotations
        annotations = generate_interaction_annotations(scenario, step)

        # Save frame
        metadata = {
            "frame_id": capture_helper.frame_counter,
            "scenario": scenario["name"],
            "step": step,
            "annotations": annotations
        }

        capture_helper.capture_frame(rgb, depth, seg, metadata)
```

### Data Quality Validation

Validate the quality of synthetic data:

```python
class DataQualityValidator:
    def __init__(self):
        self.quality_metrics = {
            "image_clarity": [],
            "annotation_accuracy": [],
            "domain_diversity": []
        }

    def validate_image_quality(self, image):
        """Validate image quality metrics"""
        # Check for blur
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        blur_score = laplacian.var()

        # Check for proper exposure
        mean_intensity = image.mean()

        return {
            "blur_score": blur_score,
            "mean_intensity": mean_intensity,
            "is_valid": blur_score > 100 and 50 < mean_intensity < 200
        }

    def validate_annotations(self, image, annotations):
        """Validate annotation quality"""
        # Check if annotations are within image bounds
        height, width = image.shape[:2]
        valid_annotations = []

        for ann in annotations:
            if (0 <= ann['x_min'] < width and 0 <= ann['x_max'] < width and
                0 <= ann['y_min'] < height and 0 <= ann['y_max'] < height and
                ann['x_max'] > ann['x_min'] and ann['y_max'] > ann['y_min']):
                valid_annotations.append(ann)

        return {
            "valid_count": len(valid_annotations),
            "total_count": len(annotations),
            "valid_ratio": len(valid_annotations) / len(annotations) if annotations else 0
        }

    def assess_domain_coverage(self, dataset_stats):
        """Assess how well the synthetic data covers the target domain"""
        # Compare synthetic data distribution to real-world expectations
        coverage_score = 0.8  # Placeholder - implement based on specific domain

        return coverage_score
```

### Synthetic Data Pipeline Integration

Integrate synthetic data generation into your development workflow:

```python
def run_synthetic_data_pipeline(config):
    """Run the complete synthetic data generation pipeline"""
    # Initialize Isaac Sim
    world = initialize_isaac_sim(config)

    # Set up synthetic data helper
    sd_helper = setup_synthetic_data_collection()

    # Set up data capture
    capture_helper = SyntheticDataCapture(config["output_dir"])

    # Set up domain randomizer
    randomizer = DomainRandomizer()

    # Generate data for each scenario
    for scenario in config["scenarios"]:
        print(f"Generating data for scenario: {scenario['name']}")

        for variation in range(config["variations_per_scenario"]):
            # Randomize the scene
            randomizer.randomize_scene(world, scenario)

            # Generate frames for this variation
            generate_scenario_data(scenario, world, sd_helper, capture_helper)

    # Validate generated data
    validator = DataQualityValidator()
    validation_results = validator.validate_dataset(config["output_dir"])

    print(f"Data generation completed. Validation results: {validation_results}")

    return config["output_dir"]
```

### Best Practices for Synthetic Data Generation

1. **Start Simple**: Begin with basic scenarios and gradually increase complexity
2. **Domain Randomization**: Vary lighting, materials, and camera parameters
3. **Quality Validation**: Implement validation checks to ensure data quality
4. **Real-World Alignment**: Ensure synthetic data covers expected real-world scenarios
5. **Annotation Accuracy**: Verify that annotations match the intended ground truth
6. **Performance Monitoring**: Monitor generation speed and resource usage

### Troubleshooting Common Issues

1. **Performance**: Reduce scene complexity if generation is too slow
2. **Memory**: Monitor GPU and system memory usage during generation
3. **Artifacts**: Check for rendering artifacts that might affect training
4. **Inconsistencies**: Ensure consistent labeling across the dataset

### Summary

In this chapter, we've covered:
- The concept and benefits of synthetic data generation
- Isaac Sim's capabilities for creating synthetic datasets
- Domain randomization techniques to improve sim-to-real transfer
- Specific techniques for generating different types of synthetic data (object detection, segmentation, etc.)
- Humanoid-specific scenarios for synthetic data generation
- Quality validation approaches
- Best practices for synthetic data generation

In the next chapter, we'll explore Isaac ROS packages for accelerated perception, learning how to leverage NVIDIA's hardware acceleration for real-time perception tasks.