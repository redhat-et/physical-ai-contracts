# Descriptor Survey: Embodiment & Robot Interface

**Date**: 2026-09-21
**Scope**: How different Physical AI communities represent robot embodiment metadata (EmbodimentDescriptor) and robot interface metadata (RobotInterfaceDescriptor).

**EmbodimentDescriptor** = mechanical/sensor properties: joints (name, type, limits, ordering, control modes), sensors (cameras, IMU, F/T), actuators, reference frames, kinematics.
**RobotInterfaceDescriptor** = what a robot accepts as commands and provides as sensor data: accepted action format, provided observation format, embodiment reference.

---

## 1. URDF (Unified Robot Description Format)

### What they call the concept

"Robot description." A single `<robot>` element describes the full kinematic/dynamic structure.

### Explicitly declared properties

- **Joints**: `<joint>` elements with attributes `name`, `type` (revolute, continuous, prismatic, fixed, floating, planar), `<axis>`, `<limit>` (lower, upper, effort, velocity), `<dynamics>` (damping, friction), parent/child link references.
- **Links**: `<link>` elements with `<inertial>` (mass, inertia tensor, origin), `<visual>` (geometry, material), `<collision>` (geometry).
- **Transmissions**: `<transmission>` elements linking joints to actuators (type, mechanical reduction, hardware interface type).
- **Geometry**: box, cylinder, sphere, mesh (STL/DAE/glTF).

### Implicit or missing properties

- **Joint ordering**: No canonical ordering. The tree structure implies a traversal order, but URDF does not declare which joint maps to action dimension N. Each consumer (MuJoCo, Isaac Lab, ros2_control) assigns its own ordering when parsing.
- **Sensors**: No native sensor elements. Cameras, IMUs, LiDAR require simulator-specific plugin tags (e.g., Gazebo `<sensor>` extensions within `<gazebo>` blocks). Not portable across simulators.
- **Actuator models**: Only transmission interfaces. No motor dynamics, harmonic drives, or series-elastic actuator properties.
- **Action space mapping**: Nothing about ML policy interfaces, observation vectors, or action spaces.
- **Camera roles**: No concept of "wrist camera" vs "world camera" for ML.
- **Named configurations**: Cannot express HOME, READY, or default joint positions.
- **Contact parameters**: Friction, stiffness, damping unspecified.
- **Closed kinematic chains**: Tree structures only.

### Format and location

XML file, typically `<robot_name>.urdf` or generated from `<robot_name>.urdf.xacro` (Xacro macro preprocessor). Shipped in ROS packages under `urdf/` or `description/` directories. Parsed by `urdfdom` (C++/Python), available across ROS distributions.

### Key gaps relative to our model

| Descriptor concept | URDF coverage |
| --- | --- |
| Joint names, types, limits | Complete |
| Joint ordering for ML | Absent -- no canonical index assignment |
| Control modes (position/velocity/effort) | Partial -- in `<transmission>` hardware interface type |
| Sensors (cameras, IMU, F/T) | Absent without simulator-specific extensions |
| Actuator dynamics | Absent |
| Reference frames | Implicit in link tree (base_link convention) |
| Action space / observation space | Absent |
| Camera roles for ML | Absent |

Sources: [ROS 2 URDF docs](https://ros-industrial.github.io/ros2_i_training/_source/urdf/introduction.html), [URDF 2.0 proposal](https://sachinchitta.github.io/urdf2/)

---

## 2. MJCF (MuJoCo XML Format)

### What they call the concept

"Model" or "simulation specification." An MJCF file describes the complete simulation: bodies, joints, actuators, sensors, contacts, tendons, and solver settings.

### Explicitly declared properties

- **Joints**: `<joint>` elements within `<body>` hierarchy. Attributes: `name`, `type` (hinge, slide, ball, free), `axis`, `range`, `damping`, `stiffness`, `ref` (reference position). Multiple joints per body allowed (composite joints without dummy bodies).
- **Actuators**: 7 actuator types including `<motor>`, `<position>`, `<velocity>`, `<general>`. Each actuator references a joint by name and has its own dynamics (gain, bias, force range, control range). This is where control modes are explicitly declared.
- **Sensors**: 33 sensor types including `<accelerometer>`, `<gyro>`, `<force>`, `<torque>`, `<camera>`, `<rangefinder>`, `<touch>`. Each sensor is a named element with a site reference.
- **Contact parameters**: `<contact>` element with `<pair>` sub-elements specifying friction, condim, margin between body pairs.
- **Tendons**: Routing through sites and joints. Equality constraints.
- **Defaults**: `<default>` element allows setting properties on all elements of a type (class-based defaults).

### Implicit or missing properties

- **Action space mapping for ML**: Like URDF, no conventions for policy action/observation spaces. The actuator list implicitly defines the action vector (action dimension i maps to actuator i), but there is no declarative metadata saying "these actuators correspond to the left arm joints."
- **Camera roles for ML**: Cameras are defined as sensors with physical properties but without semantic roles (wrist, world, head).
- **Joint ordering**: Determined by definition order in the XML file. Stable within a single MJCF file but may differ from URDF ordering for the same robot.
- **Angle units**: Default to degrees (URDF uses radians) -- a common conversion pitfall.

### Format and location

XML file, typically `<model>.xml`. In MuJoCo Menagerie: per-model directory with `<model>.xml` (MJCF), `scene.xml` (model + ground plane + lighting), meshes. Installable via `pip install mujoco-menagerie`, loadable via `mm.load('unitree_go2')`.

### Key gaps relative to our model

| Descriptor concept | MJCF coverage |
| --- | --- |
| Joint names, types, limits | Complete |
| Joint ordering for ML | Implicit (definition order = action order via actuators) |
| Control modes | Explicit -- actuator type declares control mode per joint |
| Sensors (cameras, IMU, F/T) | Complete -- 33 sensor types natively defined |
| Actuator dynamics | Complete -- motor models, gain, bias, force range |
| Reference frames | Implicit in body hierarchy |
| Action space / observation space | Absent -- derived at runtime by Gymnasium wrappers |
| Camera roles for ML | Absent |

Sources: [MuJoCo XML reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html), [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)

---

## 3. OpenUSD + SimReady + REP-0158

### What they call the concept

OpenUSD: "Scene description" with physics schemas. SimReady: "Asset validation profiles." REP-0158: "Simulation asset interoperability conventions."

### Explicitly declared properties

**OpenUSD physics schemas:**

- `UsdPhysicsJoint` -- joint types: revolute, prismatic, spherical, distance, fixed, custom. Properties: axis, lower/upper limits.
- `PhysicsArticulationRootAPI` -- marks subtrees for reduced-coordinate articulation solvers.
- `PhysicsDriveAPI` -- drive properties: type (force/acceleration), target position/velocity, stiffness, damping, max force. Applied to joints to control them.
- `UsdGeomCamera` -- projection, clipping planes, focal length.
- Mass, inertia, collision shapes on rigid body prims.

**REP-0158 additions:**

- `ros:joint:name` custom property on `UsdPhysicsJoint` prims -- "the single source of truth for ROS-facing joint identity." Governs `FollowJointTrajectory` mapping, `JointState` field names, ros2_control integration.
- Camera optical frame convention: child `UsdGeomXform` rotated 180 deg around X-axis (USD -Z to ROS +Z).
- Vendor isolation: neutral `physics.usda` layer + vendor-prefixed layers (`physx.usda`, `mujoco.usda`).
- ROS integration schemas: `RosTopicAPI`, `RosServiceAPI`, `RosActionAPI`, `RosFrameAPI` as applied API schemas.
- MKS units, Z-up, quaternion-only orientation.

**SimReady Foundation:**

- Requirements -> features -> profiles hierarchy for asset validation.
- Robot-specific features: `_ROBOT_CORE` (joint names, articulation), `_ROBOT_DRIVEN` (driven joints), `_ROBOT_BASE_ARTICULATION`.
- Machine-checkable rules (e.g., `RB.COL.001`: every rigid shape needs `CollisionAPI`).

### Implicit or missing properties

- **No ML-facing metadata**: No action spaces, observation semantics, normalization, policy interfaces.
- **No camera roles for ML**: REP-0158 defines optical frame conventions for ROS TF, not semantic roles (wrist, world, head) for VLAs.
- **No action space semantics**: `PhysicsDriveAPI` describes how to drive a joint, not how a policy's action vector maps to joints.
- **No cross-artifact compatibility**: SimReady validates individual assets, not pairs (model-robot, dataset-robot).

### Format and location

USD files (`.usda` ASCII, `.usdc` binary). Layered composition: `geometries.usdc`, `materials.usda`, `physics.usda`, `ros.usda`, vendor layers. Entry point references sub-layers via USD composition arcs.

### Key gaps relative to our model

| Descriptor concept | OpenUSD + SimReady + REP-0158 coverage |
| --- | --- |
| Joint names, types, limits | Complete (UsdPhysicsJoint + ros:joint:name) |
| Joint ordering for ML | Absent -- ros:joint:name provides identity, not index |
| Control modes | Partial -- PhysicsDriveAPI declares drive type per joint |
| Sensors (cameras, IMU, F/T) | Camera geometry only. No IMU, F/T natively. |
| Actuator dynamics | Partial -- PhysicsDriveAPI stiffness/damping |
| Reference frames | Well-defined (USD + REP-0158 optical frame convention) |
| Action space / observation space | Absent |
| Camera roles for ML | Absent |

Sources: [OpenUSD physics schema](https://openusd.org/dev/api/usd_physics_page_front.html), [REP-0158](https://reps.openrobotics.org/rep-0158-2006/), [SimReady Foundation](https://nvidia.github.io/simready-foundation/)

---

## 4. ros2_control

### What they call the concept

"Hardware interface description" embedded in URDF via `<ros2_control>` tags. Declares what command and state interfaces a robot provides to the ROS 2 control framework.

### Explicitly declared properties

- **Command interfaces**: Per-joint declaration of what commands the hardware accepts: `position`, `velocity`, `effort`. Declared as `<command_interface name="position">` within `<joint>`.
- **State interfaces**: Per-joint declaration of what state the hardware reports: `position`, `velocity`, `effort`, plus custom interfaces. Declared as `<state_interface name="position">`.
- **Hardware plugin**: `<hardware>` element with `<plugin>` specifying which system interface to load (e.g., `MujocoSystemInterface`, `GazeboSystem`, `TopicBasedSystem`).
- **Joint names**: Inherited from the URDF `<joint name="...">` that contains the `<ros2_control>` tag.
- **Joint ordering**: Defined by the order of joints in the `<ros2_control>` block. This is the canonical ordering for ros2_control controllers.

**Controller configuration (YAML):**

- Controller type (e.g., `joint_trajectory_controller/JointTrajectoryController`)
- `joints` list -- explicit ordered list of joint names this controller manages
- `command_interfaces` -- list of interface types (position, velocity)
- `state_interfaces` -- list of interface types the controller reads
- Gains, limits, tolerances per joint

### Implicit or missing properties

- **Control-level abstraction only**: Joint-level position/velocity/effort commands. No concept of task-space commands (EE delta, Cartesian velocity), which are what VLA policies typically produce.
- **No ML policy interfaces**: No action spaces in the RL/VLA sense, no observation vectors, no normalization.
- **No camera/sensor integration**: ros2_control handles joint-level hardware. Cameras are separate ROS 2 nodes publishing on topics.
- **No mapping from policy action dimensions to joints**: The controller YAML lists joints by name, but there is no mechanism to declare "action dimension 0 = joint X."

### Format and location

`<ros2_control>` XML tags embedded in URDF files. Controller configs in YAML files, typically `config/controllers.yaml` in ROS 2 packages. Hardware plugins as shared libraries loaded at runtime.

### Key gaps relative to our model

| Descriptor concept | ros2_control coverage |
| --- | --- |
| Joint names, types, limits | Complete (from URDF) |
| Joint ordering for ML | Partial -- ordering defined for controllers but not for ML policies |
| Control modes | Complete -- command_interfaces declare accepted modes |
| Sensors | Absent -- cameras/IMU handled outside ros2_control |
| Actuator dynamics | Absent -- delegated to hardware plugin |
| Reference frames | Inherited from URDF |
| Action space / observation space | Absent |
| Camera roles for ML | Absent |

Sources: [ros2_control docs](https://control.ros.org/rolling/), [mujoco_ros2_control](https://github.com/ros-controls/mujoco_ros2_control)

---

## 5. Gymnasium (Farama Foundation)

### What they call the concept

"Environment" with `action_space` and `observation_space`. The environment wraps a robot simulator and exposes a standardized Python API.

### Explicitly declared properties

- **`action_space`**: A `gymnasium.spaces.Space` object. Common types:
  - `Box(low, high, shape, dtype)` -- continuous actions with bounds
  - `Discrete(n)` -- discrete actions
  - `Dict({name: Space, ...})` -- named sub-spaces
  - `Tuple((Space, ...))` -- ordered sub-spaces
  - `MultiBinary(n)`, `MultiDiscrete(nvec)`
- **`observation_space`**: Same `Space` types. For robotics, typically `Dict` with keys like `observation`, `achieved_goal`, `desired_goal` (via `GoalEnv`).
- **`step(action) -> (obs, reward, terminated, truncated, info)`**: Standard RL loop.
- **`env.spec`**: `EnvSpec` with `id`, `entry_point`, `max_episode_steps`, `reward_threshold`, `kwargs`.
- **`metadata`**: Dict with `render_modes`, `render_fps`.

### Implicit or missing properties

- **Runtime API, not metadata format**: Spaces are defined in Python code (the environment's `__init__` method), not in a declarative schema file. You must instantiate the environment to discover its spaces.
- **No per-dimension semantics**: A `Box(-1, 1, (7,))` action space carries shape and range but not "dimension 0 is dx, dimension 1 is dy, ..., dimension 6 is gripper."
- **No joint mapping**: No link between action dimensions and robot joints by name.
- **No camera roles**: Image observations are keyed by arbitrary strings in `Dict` spaces.
- **No normalization metadata**: No information about how observations/actions should be normalized for training.
- **No embodiment reference**: No link to a URDF, MJCF, or USD file.
- **No control frequency**: The `step()` rate is determined by the environment's internal timestep, not declared in metadata.

### Format and location

Python classes inheriting from `gymnasium.Env`. Registered via `gymnasium.register()` with an ID string (e.g., `Ant-v5`). Spaces are `gymnasium.spaces.*` objects. No standalone metadata file.

### Key gaps relative to our model

| Descriptor concept | Gymnasium coverage |
| --- | --- |
| Joint names, types, limits | Absent -- Box bounds only, no named dimensions |
| Joint ordering for ML | Implicit (positional in array) -- no named mapping |
| Control modes | Absent |
| Sensors | Absent -- observation shape only, no sensor semantics |
| Actuator dynamics | Absent |
| Reference frames | Absent |
| Action space shape and bounds | Complete |
| Observation space shape and bounds | Complete |
| Camera roles for ML | Absent |

Sources: [Gymnasium docs](https://gymnasium.farama.org/), [Gymnasium-Robotics](https://robotics.farama.org/)

---

## 6. LEAPP (NVIDIA)

### What they call the concept

"Embodiment tag" and "joint config" at export time. "Tensor semantics" for annotating model I/O with per-dimension metadata.

### Explicitly declared properties

**Export-time parameters:**

- `--embodiment_tag`: String identifier for the target robot (e.g., `unitree_g1`, `franka_panda`). Selects the appropriate embodiment-specific preprocessing and action head.
- `--joint_config`: Path to a joint configuration file specifying which joints are active, their ordering, and control properties.

**Tensor semantics (the most ML-facing robot metadata in any ecosystem):**

- `kind`: Semantic category per tensor dimension -- `JOINT_POSITION`, `JOINT_VELOCITY`, `GRIPPER_STATE`, `EE_POSITION`, `EE_ORIENTATION`, etc.
- `element_names`: Per-dimension names (e.g., `hip_l`, `knee_l`, `ankle_l`).
- `extra`: Extensible dict for additional metadata per dimension -- documented fields include `frame` (reference frame) and `units` (e.g., radians, meters).

**ONNX bundle:**

- Full pipeline (preprocessing + backbone + action head) traced as ONNX. Normalization baked into the graph.
- Model input/output tensor names and shapes in ONNX metadata.

### Implicit or missing properties

- **No standardized schema for `extra` fields**: The `frame` and `units` fields are documented conventions, not enforced schema.
- **Camera/observation semantics not covered**: Tensor semantics annotate action outputs, not observation inputs.
- **Embodiment tag is an opaque string**: No machine-readable definition of what `unitree_g1` means -- the mapping from tag to joint config is internal to the LEAPP codebase.
- **Joint config format not publicly documented**: The structure of the joint configuration file is not part of the public API.
- **No cross-artifact compatibility**: Cannot check whether a LEAPP-exported model is compatible with a particular robot configuration without running the export.

### Format and location

Python API for tensor semantics. ONNX files for exported models. CLI parameters for export (`leapp export --embodiment_tag ... --joint_config ...`). Deployed via Triton Inference Server with `isaac_ros_deploy` ROS 2 bridge.

### Key gaps relative to our model

| Descriptor concept | LEAPP coverage |
| --- | --- |
| Joint names | Partial -- `element_names` in tensor semantics |
| Joint types, limits | Absent |
| Joint ordering for ML | Present -- baked into ONNX at export via `--joint_config` |
| Control modes | Partial -- `kind` annotation (`JOINT_POSITION` vs `JOINT_VELOCITY`) |
| Sensors | Absent from tensor semantics |
| Actuator dynamics | Absent |
| Reference frames | Partial -- `extra.frame` field (convention, not enforced) |
| Action space semantics | Best coverage of any ecosystem -- `kind`, `element_names`, `extra` |
| Observation space semantics | Absent |
| Camera roles for ML | Absent |

Sources: [LEAPP docs](https://nvidia-isaac.github.io/leapp/), [LEAPP tensor semantics](https://nvidia-isaac.github.io/leapp/semantics/usage.html)

---

## 7. Isaac Lab (NVIDIA)

### What they call the concept

"Articulation config" (`ArticulationCfg`) for the robot, "Actions config" (`ActionsCfg`) and "Observations config" (`ObservationsCfg`) for the environment interface.

### Explicitly declared properties

**ArticulationCfg:**

- `spawn`: USD asset path or URDF path (auto-converted to USD). Specifies the robot model to load.
- `init_state`: Initial joint positions and velocities (`ArticulationCfg.InitialStateCfg`). Named joint positions can be specified as a dict mapping joint name regex patterns to values.
- `actuators`: Dict of `ActuatorCfg` groups, each with:
  - `joint_names_expr`: Regex pattern matching joint names from the USD/URDF
  - `effort_limit`, `velocity_limit`, `stiffness`, `damping`
  - Actuator model class (e.g., `ImplicitActuatorCfg`, `DCMotorCfg`, `IdealPDActuatorCfg`, `ActuatorNetMLPCfg`)
  - `saturation_effort` for motor models
- `soft_joint_pos_limit_factor`: Safety scaling on position limits

**ActionsCfg (per environment):**

- Action term classes (e.g., `JointPositionActionCfg`, `JointVelocityActionCfg`, `JointEffortActionCfg`, `EEPoseActionCfg`)
- Each action term references an `asset_name` and `joint_names` (list or regex)
- `scale` factor per action term
- Defines which joints are controlled and in what mode -- this is where action space dimensions are determined

**ObservationsCfg (per environment):**

- Observation groups (e.g., `policy`, `critic`) each containing observation term configs
- Each term is a callable (function reference) that computes observations from env state
- Terms have `scale`, `clip`, `noise` parameters
- Common terms: `joint_pos`, `joint_vel`, `body_pose_w`, camera images
- `enable_corruption` flag per group (for domain randomization)

**Joint ordering:**

- Determined by USD articulation order when the asset is loaded. This can differ from URDF definition order -- confirmed as a known issue (Isaac Lab [#7750](https://github.com/isaac-sim/IsaacLab/issues/7750)).
- `Articulation.data.joint_names` provides the runtime joint name list.
- Action terms select joints by `joint_names` regex, but the mapping from action dimension index to specific joint is implicit in the regex match order.

### Implicit or missing properties

- **Action space shape derived at runtime**: The total action space is the concatenation of all action terms' joint selections. No declarative metadata file describes the resulting `Box` space.
- **Observation space derived at runtime**: Same -- observation group shapes are determined by which terms are configured, computed at `env.reset()`.
- **Camera roles**: Cameras are scene objects configured in USD, accessed by name. No semantic role annotation.
- **No standalone robot metadata**: The robot's properties are split across USD asset + `ArticulationCfg` + `ActionsCfg` + `ObservationsCfg`. No single file describes "what this robot is" for ML purposes.
- **Quaternion convention**: `(w,x,y,z)`, differs from ROS 2's `(x,y,z,w)`.

### Format and location

Python dataclass configs (`@configclass` decorator). Typically in `envs/<task>/<task>_env_cfg.py`. Robot configs in `assets/` directories. No standalone metadata files -- everything is Python code.

### Key gaps relative to our model

| Descriptor concept | Isaac Lab coverage |
| --- | --- |
| Joint names, types, limits | Complete (from USD/URDF asset + ActuatorCfg overrides) |
| Joint ordering for ML | Implicit -- USD articulation order, known to differ from URDF |
| Control modes | Complete -- action term classes declare mode per joint group |
| Sensors (cameras, IMU, F/T) | Partial -- camera via USD scene objects; IMU/F/T via sensor configs |
| Actuator dynamics | Complete -- actuator model classes with gains, limits, saturation |
| Reference frames | From USD + explicit `body_pose_w` observation terms |
| Action space | Implicit -- derived from ActionsCfg at runtime, not declared |
| Observation space | Implicit -- derived from ObservationsCfg at runtime, not declared |
| Camera roles for ML | Absent |

Sources: [Isaac Lab articulation API](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.html), [Isaac Lab migration guide](https://isaac-sim.github.io/IsaacLab/main/source/migration/migrating_from_isaacgymenvs.html), [Isaac Lab #7750](https://github.com/isaac-sim/IsaacLab/issues/7750)

---

## 8. GR00T (NVIDIA)

### What they call the concept

"Embodiment config" and "modality config." Robot identity is a string tag; data format mapping is via `modality.json`.

### Explicitly declared properties

**Embodiment identification:**

- String-based embodiment tags (e.g., `unitree_g1`, `fourier_gr1`, `franka_panda`). Used to select robot-specific preprocessing, action heads, and normalization.
- The mapping from embodiment tag to specific robot configuration is internal to the GR00T/LEAPP codebase.

**`modality.json`:**

- Maps index ranges in flat state/action arrays to named semantic fields: e.g., `"joint_position": {"start": 0, "end": 6}`, `"gripper": {"start": 6, "end": 7}`.
- `video_key` remapping: maps dataset camera names to model input names.
- Declares which modalities the model expects (images, state, language).

**Robot configs (in Isaac-GR00T repo):**

- Per-robot Python configs defining: active joint names, joint limits, default positions, control mode (position/velocity).
- Joint ordering is hardcoded in the config as an ordered list.
- Normalization statistics reference (per-embodiment stats).

**Two-tier architecture for humanoids:**

- GR00T VLA at ~10 Hz produces latent action tokens or high-level commands.
- SONIC whole-body controller at 50-120 Hz decodes into motor commands.
- Each tier has its own embodiment configuration.

### Implicit or missing properties

- **Embodiment tag is an opaque string**: No machine-readable definition. You cannot discover what joints `unitree_g1` has without reading the Python code.
- **No standardized format**: Robot configs are Python code, not declarative metadata files.
- **Camera semantics limited**: `video_key` remapping handles name translation but does not declare camera roles (wrist, world), intrinsics, or extrinsics.
- **`modality.json` splits arrays but lacks semantics**: Index ranges map to field names, but no representation type (absolute vs delta), units, or coordinate frame declarations.
- **No standalone robot description**: Robot properties are scattered across embodiment configs, modality configs, and LEAPP export parameters.

### Format and location

Python config files in `Isaac-GR00T` repo. `modality.json` as JSON alongside datasets. LEAPP CLI parameters at export time. No standalone robot description format.

### Key gaps relative to our model

| Descriptor concept | GR00T coverage |
| --- | --- |
| Joint names | Present -- in per-robot Python configs |
| Joint types, limits | Present -- in per-robot Python configs |
| Joint ordering for ML | Present -- hardcoded ordered list in config |
| Control modes | Present -- declared in config |
| Sensors | Absent from robot config; `video_key` handles camera name mapping only |
| Actuator dynamics | Absent |
| Reference frames | Absent |
| Action space semantics | Partial -- `modality.json` index ranges, no representation/units |
| Observation space semantics | Partial -- modality declarations, no per-dimension semantics |
| Camera roles for ML | Absent |

Sources: [GR00T data preparation](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/data_preparation.md), [GR00T E2E workflow](https://docs.nvidia.com/learning/physical-ai/gr00t-e2e-workflow/latest/)

---

## Cross-Community Analysis

### Embodiment identity: how each community identifies a robot

| Community | Robot identity mechanism | Machine-readable? |
| --- | --- | --- |
| URDF | `<robot name="...">` in XML file | Yes -- but name is free-form |
| MJCF | `<mujoco model="...">` in XML file | Yes -- but name is free-form |
| OpenUSD/REP-0158 | USD prim path + `ros:joint:name` per joint | Yes |
| ros2_control | URDF with `<ros2_control>` tags | Yes |
| Gymnasium | Environment ID string (e.g., `Ant-v5`) | No -- ID maps to Python code |
| LEAPP | `--embodiment_tag` (opaque string) | No -- maps to internal code |
| Isaac Lab | USD asset path + `ArticulationCfg` Python | No -- Python code |
| GR00T | Embodiment tag string | No -- maps to internal code |

### Joint ordering: the fundamental interoperability problem

| Community | How joint order is determined | Explicit or implicit? |
| --- | --- | --- |
| URDF | Tree traversal order (implementation-dependent) | Implicit |
| MJCF | XML definition order | Implicit |
| OpenUSD | Articulation solver's internal ordering | Implicit |
| ros2_control | Order in `<ros2_control>` block | Explicit (for controllers) |
| Gymnasium | Positional in `action_space` array | Implicit |
| LEAPP | Baked into ONNX at export time | Explicit (frozen at export) |
| Isaac Lab | USD articulation order | Implicit (known to differ from URDF) |
| GR00T | Hardcoded ordered list in Python config | Explicit (in code) |

Confirmed: URDF order != USD articulation order != MJCF order for the same robot (Isaac Lab [#7750](https://github.com/isaac-sim/IsaacLab/issues/7750)).

### Action semantics: who declares what each action dimension means

| Community | Per-dimension semantics? | Representation type? | Units? |
| --- | --- | --- | --- |
| URDF | No | No | Radians/meters (by convention) |
| MJCF | Actuator names link to joints | Actuator type implies mode | Degrees default |
| OpenUSD | No | PhysicsDriveAPI type | SI |
| ros2_control | Joint names in controller YAML | command_interface type | From URDF |
| Gymnasium | No | No | No |
| LEAPP | `kind` + `element_names` | `kind` (JOINT_POSITION etc.) | `extra.units` (convention) |
| Isaac Lab | Via ActionsCfg joint regex | Action term class | From USD |
| GR00T | `modality.json` field names | No | No |

**LEAPP tensor semantics is the only system with explicit per-dimension action annotation.** All others derive action semantics implicitly from the simulator/controller configuration or hardcode them in Python.

### What would a complete EmbodimentDescriptor need that no single community provides?

1. **Joint ordering for ML** -- a canonical mapping from joint names to action-space indices, independent of simulator.
2. **Camera roles** -- semantic role vocabulary (wrist, world, head, overhead) linked to camera identifiers.
3. **Per-dimension action semantics** -- for each action dimension: joint name, representation type (position/velocity/delta/absolute), units, coordinate frame.
4. **Cross-format identity** -- mapping between URDF joint name, MJCF joint name, and USD joint prim for the same physical joint.
5. **Sensor inventory** -- declarative list of available sensors with types, not scattered across simulator-specific extensions.

### What would a complete RobotInterfaceDescriptor need?

1. **Accepted action format** -- shape, per-dimension semantics (from EmbodimentDescriptor), control frequency, data type.
2. **Provided observation format** -- per-modality (cameras with roles, proprioception with joint semantics, additional sensors).
3. **Normalization expectations** -- what normalization the robot controller expects on incoming actions (raw, normalized, denormalized).
4. **Embodiment reference** -- link to the underlying EmbodimentDescriptor (URDF/MJCF/USD file).
5. **Transport** -- how to connect (ROS 2 topics, WebSocket endpoint, shared memory).
