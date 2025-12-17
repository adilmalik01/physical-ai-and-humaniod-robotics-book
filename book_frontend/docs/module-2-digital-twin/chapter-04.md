---
sidebar_position: 4
---

# Chapter 04: Physics-Based Environment Design

## Creating Realistic Simulation Worlds

In this chapter, we'll explore how to design physics-based environments that accurately represent real-world scenarios for humanoid robot testing and validation. Creating realistic environments is crucial for the sim-to-real transfer of robot behaviors.

### Understanding Environment Design Principles

Effective environment design for humanoid robots requires consideration of several factors:

1. **Physics Accuracy**: Environments must simulate real-world physics correctly
2. **Interaction Complexity**: Include elements that challenge robot capabilities
3. **Safety**: Design environments that allow safe testing of robot behaviors
4. **Variability**: Create diverse scenarios for comprehensive testing
5. **Scalability**: Design modular environments that can be extended

### SDF World Design Fundamentals

Gazebo worlds are defined using SDF (Simulation Description Format). Here's a comprehensive example of a humanoid testing environment:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="humanoid_test_world">
    <!-- Physics engine configuration -->
    <physics name="1ms" type="ignored">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000.0</real_time_update_rate>
      <gravity>0 0 -9.8</gravity>
    </physics>

    <!-- Include default models -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Add fog for realistic outdoor simulation -->
    <scene>
      <ambient>0.4 0.4 0.4 1</ambient>
      <background>0.7 0.7 0.7 1</background>
      <shadows>false</shadows>
      <fog>
        <type>linear</type>
        <color>0.8 0.8 0.8 1</color>
        <density>0.01</density>
        <start>10</start>
        <end>100</end>
      </fog>
    </scene>

    <!-- Simple indoor environment -->
    <model name="room_walls">
      <!-- Back wall -->
      <link name="back_wall">
        <collision name="collision">
          <geometry>
            <box>
              <size>10 0.2 3</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>10 0.2 3</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
        <pose>0 -5 1.5 0 0 0</pose>
        <inertial>
          <mass>100</mass>
          <inertia>
            <ixx>1</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>1</iyy>
            <iyz>0</iyz>
            <izz>1</izz>
          </inertial>
        </inertial>
      </link>
    </model>

    <!-- Left wall -->
    <model name="left_wall">
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>0.2 10 3</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.2 10 3</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
        <pose>-5 0 1.5 0 0 0</pose>
        <inertial>
          <mass>100</mass>
          <inertia>
            <ixx>1</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>1</iyy>
            <iyz>0</iyz>
            <izz>1</izz>
          </inertial>
        </inertial>
      </link>
    </model>

    <!-- Furniture for humanoid interaction -->
    <model name="table">
      <pose>2 0 0 0 0 0</pose>
      <link name="table_top">
        <collision name="collision">
          <geometry>
            <box>
              <size>1.2 0.8 0.02</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1.2 0.8 0.02</size>
            </box>
          </geometry>
          <material>
            <ambient>0.6 0.4 0.2 1</ambient>
            <diffuse>0.6 0.4 0.2 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>20</mass>
          <inertia>
            <ixx>1</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>1</iyy>
            <iyz>0</iyz>
            <izz>1</izz>
          </inertial>
        </inertial>
      </link>
      <link name="leg1">
        <pose>-0.5 -0.3 -0.4 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <box>
              <size>0.05 0.05 0.8</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.05 0.05 0.8</size>
            </box>
          </geometry>
          <material>
            <ambient>0.6 0.4 0.2 1</ambient>
            <diffuse>0.6 0.4 0.2 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>2</mass>
          <inertia>
            <ixx>0.1</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.1</iyy>
            <iyz>0</iyz>
            <izz>0.1</izz>
          </inertial>
        </inertial>
      </link>
      <link name="leg2">
        <pose>0.5 -0.3 -0.4 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <box>
              <size>0.05 0.05 0.8</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.05 0.05 0.8</size>
            </box>
          </geometry>
          <material>
            <ambient>0.6 0.4 0.2 1</ambient>
            <diffuse>0.6 0.4 0.2 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>2</mass>
          <inertia>
            <ixx>0.1</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.1</iyy>
            <iyz>0</iyz>
            <izz>0.1</izz>
          </inertial>
        </inertial>
      </link>
      <link name="leg3">
        <pose>-0.5 0.3 -0.4 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <box>
              <size>0.05 0.05 0.8</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.05 0.05 0.8</size>
            </box>
          </geometry>
          <material>
            <ambient>0.6 0.4 0.2 1</ambient>
            <diffuse>0.6 0.4 0.2 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>2</mass>
          <inertia>
            <ixx>0.1</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.1</iyy>
            <iyz>0</iyz>
            <izz>0.1</izz>
          </inertial>
        </inertial>
      </link>
      <link name="leg4">
        <pose>0.5 0.3 -0.4 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <box>
              <size>0.05 0.05 0.8</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.05 0.05 0.8</size>
            </box>
          </geometry>
          <material>
            <ambient>0.6 0.4 0.2 1</ambient>
            <diffuse>0.6 0.4 0.2 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>2</mass>
          <inertia>
            <ixx>0.1</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.1</iyy>
            <iyz>0</iyz>
            <izz>0.1</izz>
          </inertial>
        </inertial>
      </link>
      <joint name="top_to_leg1" type="fixed">
        <parent>table_top</parent>
        <child>leg1</child>
      </joint>
      <joint name="top_to_leg2" type="fixed">
        <parent>table_top</parent>
        <child>leg2</child>
      </joint>
      <joint name="top_to_leg3" type="fixed">
        <parent>table_top</parent>
        <child>leg3</child>
      </joint>
      <joint name="top_to_leg4" type="fixed">
        <parent>table_top</parent>
        <child>leg4</child>
      </joint>
    </model>

    <!-- Objects for manipulation -->
    <model name="cup">
      <pose>2.1 0.1 0.8 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <cylinder>
              <radius>0.04</radius>
              <length>0.1</length>
            </cylinder>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <cylinder>
              <radius>0.04</radius>
              <length>0.1</length>
            </cylinder>
          </geometry>
          <material>
            <ambient>0.2 0.6 0.8 1</ambient>
            <diffuse>0.2 0.6 0.8 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>0.1</mass>
          <inertia>
            <ixx>0.0001</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.0001</iyy>
            <iyz>0</iyz>
            <izz>0.0001</izz>
          </inertial>
        </inertial>
      </link>
    </model>

    <!-- Stairs for locomotion testing -->
    <model name="stairs">
      <pose>-3 0 0 0 0 0</pose>
      <link name="step1">
        <collision name="collision">
          <geometry>
            <box>
              <size>1 1 0.2</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1 1 0.2</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.5 0.5 0.5 1</diffuse>
          </material>
        </visual>
        <pose>0 0 0.1 0 0 0</pose>
        <inertial>
          <mass>10</mass>
          <inertia>
            <ixx>1</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>1</iyy>
            <iyz>0</iyz>
            <izz>1</izz>
          </inertial>
        </inertial>
      </link>
      <link name="step2">
        <collision name="collision">
          <geometry>
            <box>
              <size>1 1 0.2</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1 1 0.2</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.5 0.5 0.5 1</diffuse>
          </material>
        </visual>
        <pose>0 0 0.3 0 0 0</pose>
        <inertial>
          <mass>10</mass>
          <inertia>
            <ixx>1</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>1</iyy>
            <iyz>0</iyz>
            <izz>1</izz>
          </inertial>
        </inertial>
      </link>
      <link name="step3">
        <collision name="collision">
          <geometry>
            <box>
              <size>1 1 0.2</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1 1 0.2</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.5 0.5 0.5 1</diffuse>
          </material>
        </visual>
        <pose>0 0 0.5 0 0 0</pose>
        <inertial>
          <mass>10</mass>
          <inertia>
            <ixx>1</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>1</iyy>
            <iyz>0</iyz>
            <izz>1</izz>
          </inertial>
        </inertial>
      </link>
    </model>

    <!-- Obstacles for navigation -->
    <model name="obstacle1">
      <pose>0 2 0.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>0.5 0.5 1</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.5 0.5 1</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.4 0.1 1</ambient>
            <diffuse>0.8 0.4 0.1 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>5</mass>
          <inertia>
            <ixx>0.5</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.5</iyy>
            <iyz>0</iyz>
            <izz>0.5</izz>
          </inertial>
        </inertial>
      </link>
    </model>

    <model name="obstacle2">
      <pose>0 -2 0.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <cylinder>
              <radius>0.3</radius>
              <length>1</length>
            </cylinder>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <cylinder>
              <radius>0.3</radius>
              <length>1</length>
            </cylinder>
          </geometry>
          <material>
            <ambient>0.1 0.4 0.8 1</ambient>
            <diffuse>0.1 0.4 0.8 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>5</mass>
          <inertia>
            <ixx>0.5</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.5</iyy>
            <iyz>0</iyz>
            <izz>0.5</izz>
          </inertial>
        </inertial>
      </link>
    </model>

    <!-- Add a humanoid robot spawn point -->
    <state world_name="humanoid_test_world">
      <model name="humanoid_robot">
        <pose>0 0 1.0 0 0 0</pose>
        <link name="torso">
          <pose>0 0 1.0 0 0 0</pose>
        </link>
      </model>
    </state>
  </world>
</sdf>
```

### Physics Engine Configuration

For humanoid robot simulation, careful physics configuration is essential:

```xml
<physics name="humanoid_physics" type="ode">
  <!-- Time step - smaller is more accurate but slower -->
  <max_step_size>0.001</max_step_size>

  <!-- Real-time update rate -->
  <real_time_update_rate>1000</real_time_update_rate>

  <!-- Real-time factor (1.0 = real-time, >1 = faster than real-time) -->
  <real_time_factor>1.0</real_time_factor>

  <!-- Gravity -->
  <gravity>0 0 -9.8</gravity>

  <!-- ODE-specific parameters -->
  <ode>
    <!-- Solver parameters -->
    <solver>
      <type>quick</type>
      <iters>100</iters>
      <sor>1.3</sor>
    </solver>

    <!-- Constraints parameters -->
    <constraints>
      <cfm>0.000001</cfm>
      <erp>0.2</erp>
      <contact_max_correcting_vel>100</contact_max_correcting_vel>
      <contact_surface_layer>0.001</contact_surface_layer>
    </constraints>
  </ode>
</physics>
```

### Creating Modular Environments

For complex humanoid testing, it's useful to create modular environments that can be combined:

#### Indoor Environment Module
```xml
<!-- indoor_room.sdf -->
<sdf version="1.7">
  <model name="indoor_room">
    <!-- Room structure -->
    <link name="room_structure">
      <visual name="walls">
        <geometry>
          <mesh>
            <uri>model://indoor_room/meshes/room.dae</uri>
          </mesh>
        </geometry>
      </visual>
      <collision name="walls_collision">
        <geometry>
          <mesh>
            <uri>model://indoor_room/meshes/room_collision.dae</uri>
          </mesh>
        </geometry>
      </collision>
      <inertial>
        <mass>1000</mass>
        <inertia>
          <ixx>1000</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>1000</iyy>
          <iyz>0</iyz>
          <izz>1000</izz>
        </inertia>
      </inertial>
    </link>

    <!-- Furniture -->
    <include>
      <uri>model://table</uri>
      <pose>2 0 0 0 0 0</pose>
    </include>

    <include>
      <uri>model://chair</uri>
      <pose>1.5 1 0 0 0 1.57</pose>
    </include>
  </model>
</sdf>
```

#### Outdoor Environment Module
```xml
<!-- outdoor_park.sdf -->
<sdf version="1.7">
  <model name="outdoor_park">
    <!-- Ground terrain -->
    <link name="terrain">
      <visual name="ground_visual">
        <geometry>
          <heightmap>
            <uri>model://outdoor_park/materials/textures/terrain.png</uri>
            <size>20 20 3</size>
            <pos>0 0 0</pos>
          </heightmap>
        </geometry>
      </visual>
      <collision name="ground_collision">
        <geometry>
          <heightmap>
            <uri>model://outdoor_park/materials/textures/terrain.png</uri>
            <size>20 20 3</size>
            <pos>0 0 0</pos>
          </heightmap>
        </geometry>
      </collision>
      <inertial>
        <mass>10000</mass>
        <inertia>
          <ixx>10000</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>10000</iyy>
          <iyz>0</iyz>
          <izz>10000</izz>
        </inertia>
      </inertial>
    </link>

    <!-- Trees and obstacles -->
    <include>
      <uri>model://tree</uri>
      <pose>-5 5 0 0 0 0</pose>
    </include>

    <include>
      <uri>model://tree</uri>
      <pose>5 -5 0 0 0 0</pose>
    </include>

    <!-- Path for navigation -->
    <include>
      <uri>model://path_segment</uri>
      <pose>0 0 0 0 0 0</pose>
    </include>
  </model>
</sdf>
```

### Advanced Physics Properties

For realistic humanoid interaction with environments:

#### Surface Properties
```xml
<gazebo reference="floor_surface">
  <collision>
    <surface>
      <friction>
        <ode>
          <mu>0.8</mu>    <!-- Static friction coefficient -->
          <mu2>0.8</mu2>  <!-- Dynamic friction coefficient -->
          <fdir1>1 0 0</fdir1>  <!-- Friction direction -->
        </ode>
      </friction>
      <bounce>
        <restitution_coefficient>0.1</restitution_coefficient>  <!-- Bounciness -->
        <threshold>100000</threshold>  <!-- Velocity threshold for bounce -->
      </bounce>
      <contact>
        <ode>
          <soft_cfm>0.000001</soft_cfm>    <!-- Soft constraint force mixing -->
          <soft_erp>0.2</soft_erp>         <!-- Soft error reduction parameter -->
          <kp>1000000000000.0</kp>         <!-- Contact stiffness -->
          <kd>1.0</kd>                     <!-- Contact damping -->
          <max_vel>100.0</max_vel>          <!-- Maximum contact correction velocity -->
          <min_depth>0.0001</min_depth>     <!-- Depth of interpenetration -->
        </ode>
      </contact>
    </surface>
  </collision>
</gazebo>
```

#### Material Properties for Different Surfaces
```xml
<!-- Different surfaces for various walking conditions -->
<gazebo reference="wood_floor">
  <mu1>0.6</mu1>
  <mu2>0.6</mu2>
  <kp>1000000000000.0</kp>
  <kd>1.0</kd>
</gazebo>

<gazebo reference="carpet">
  <mu1>0.9</mu1>
  <mu2>0.9</mu2>
  <kp>500000000000.0</kp>
  <kd>2.0</kd>
</gazebo>

<gazebo reference="tile">
  <mu1>0.4</mu1>
  <mu2>0.4</mu2>
  <kp>2000000000000.0</kp>
  <kd>0.5</kd>
</gazebo>
```

### Dynamic Environments

For advanced testing, you can create environments with dynamic elements:

```xml
<!-- Moving platform for balance testing -->
<model name="moving_platform">
  <link name="platform">
    <collision name="collision">
      <geometry>
        <box>
          <size>2 2 0.1</size>
        </box>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <box>
          <size>2 2 0.1</size>
        </box>
      </geometry>
      <material>
        <ambient>0.7 0.7 0.7 1</ambient>
        <diffuse>0.7 0.7 0.7 1</diffuse>
      </material>
    </visual>
    <inertial>
      <mass>50</mass>
      <inertia>
        <ixx>10</ixx>
        <ixy>0</ixy>
        <ixz>0</ixz>
        <iyy>10</iyy>
        <iyz>0</iyz>
        <izz>20</izz>
      </inertia>
    </inertial>
  </link>

  <!-- Joint to make it oscillate -->
  <joint name="platform_oscillation" type="prismatic">
    <parent>world</parent>
    <child>platform</child>
    <axis>
      <xyz>0 1 0</xyz>
    </axis>
  </joint>

  <!-- Plugin to control the motion -->
  <plugin filename="gz-sim-joint-position-controller-system" name="gz::sim::systems::JointPositionController">
    <joint_name>platform_oscillation</joint_name>
  </plugin>
</model>
```

### Environment Testing Scenarios

Create specific scenarios for different aspects of humanoid robot testing:

#### Balance Testing Environment
```xml
<!-- balance_test.sdf -->
<world name="balance_test">
  <!-- Physics configuration optimized for balance -->
  <physics name="balance_physics" type="ode">
    <max_step_size>0.001</max_step_size>
    <real_time_update_rate>1000</real_time_update_rate>
    <real_time_factor>1.0</real_time_factor>
    <gravity>0 0 -9.8</gravity>
  </physics>

  <!-- Include ground plane -->
  <include>
    <uri>model://ground_plane</uri>
  </include>

  <!-- Slightly unstable platform -->
  <model name="unstable_platform">
    <pose>0 0 0 0 0 0</pose>
    <link name="platform">
      <collision name="collision">
        <geometry>
          <cylinder>
            <radius>1.0</radius>
            <length>0.1</length>
          </cylinder>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <cylinder>
            <radius>1.0</radius>
            <length>0.1</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>0.5 0.5 0.5 1</ambient>
          <diffuse>0.5 0.5 0.5 1</diffuse>
        </material>
      </visual>
      <inertial>
        <mass>100</mass>
        <inertia>
          <ixx>10</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>10</iyy>
          <iyz>0</iyz>
          <izz>20</izz>
        </inertia>
      </inertial>
    </link>
  </model>

  <!-- Add some obstacles for balance challenges -->
  <model name="balance_obstacle">
    <pose>1.5 0 0.5 0 0 0</pose>
    <link name="link">
      <collision name="collision">
        <geometry>
          <box>
            <size>0.2 0.2 1</size>
          </box>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <box>
            <size>0.2 0.2 1</size>
          </box>
        </geometry>
        <material>
          <ambient>0.8 0.2 0.2 1</ambient>
          <diffuse>0.8 0.2 0.2 1</diffuse>
        </material>
      </visual>
      <inertial>
        <mass>1</mass>
        <inertia>
          <ixx>0.1</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>0.1</iyy>
          <iyz>0</iyz>
          <izz>0.1</izz>
        </inertia>
      </inertial>
    </link>
  </model>
</world>
```

### Loading and Using Environments

To load your custom environments:

```bash
# Launch with your custom world
gz sim -r custom_world.sdf

# Or through ROS 2
ros2 launch ros_gz_sim gz_sim.launch.py world:=/path/to/your/world.sdf
```

### Environment Validation

Before using your environment for testing:

1. **Physics Stability**: Ensure objects don't exhibit unrealistic behavior
2. **Collision Detection**: Verify all objects have proper collision geometry
3. **Performance**: Check that simulation runs at acceptable speeds
4. **Realism**: Compare to real-world physics and behaviors

### Troubleshooting Common Issues

1. **Simulation Instability**: Reduce time step or adjust physics parameters
2. **Objects Falling Through**: Check collision geometry and surface properties
3. **Performance Issues**: Simplify collision geometry or reduce complexity
4. **Jittery Movement**: Adjust solver parameters and contact properties

### Summary

In this chapter, we've covered:
- SDF world design fundamentals for humanoid robots
- Physics engine configuration for realistic simulation
- Creating modular environments for different testing scenarios
- Advanced physics properties for realistic interaction
- Dynamic environments for advanced testing
- Specific scenarios for different aspects of humanoid robot testing
- Environment validation and troubleshooting

In the next chapter, we'll explore sensor simulation and integration, which is crucial for creating realistic perception capabilities in your simulated humanoid robots.