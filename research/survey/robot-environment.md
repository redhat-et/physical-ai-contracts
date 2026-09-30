# Prior Art Survey: Robot & Environment Description Standards

**Date**: 2026-09-18
**Scope**: URDF, SDF, MJCF, OpenUSD, REP-0158, SimReady Foundation, ROS 2 interfaces, MuJoCo Menagerie, URDD, simulation_interfaces, ros2_control, Gymnasium/Farama

---

## 1. URDF (Unified Robot Description Format)

### What it describes

XML format for robot kinematic and dynamic structure. Core elements are `<link>` (rigid body with mass, inertia, visual/collision geometry) and `<joint>` (connects parent-child links). Six joint types: revolute, continuous, prismatic, fixed, floating, planar. Joint properties include limits (position, velocity, effort) and dynamics (damping, friction). Geometry references external mesh files (STL, DAE, glTF).

Xacro is a macro preprocessor for URDF that enables parametric robot descriptions — repeated structures written once and expanded at build time.

Source: [ROS 2 URDF documentation](https://ros-industrial.github.io/ros2_i_training/_source/urdf/introduction.html), [Articulated Robotics tutorial](https://articulatedrobotics.xyz/tutorials/ready-for-ros/urdf/)

### What it doesn't

- **No closed kinematic chains** — tree structures only; parallel linkages impossible
- **No sensor descriptions** — cameras, IMUs, LiDAR require simulator-specific plugin tags (e.g., Gazebo `<sensor>` extensions)
- **No actuator models** — only transmission interfaces; no motor dynamics, harmonic drives, or series-elastic actuators
- **No contact parameters** — friction, stiffness, damping unspecified
- **No named states** — cannot express HOME, READY, or default configurations
- **No multi-robot** — single `<robot>` element per file
- **No action/observation semantics** — nothing about ML policy interfaces, observation vectors, or action spaces
- **No camera roles** — no concept of "wrist camera" vs "world camera" for ML

### Type system & extensibility

Minimal type system: joint types (6 fixed kinds), geometry types (box, cylinder, sphere, mesh). No formal extension mechanism — extensions are done via Gazebo plugin tags or Xacro macros. This lack of extensibility is a widely recognized limitation.

### Community adoption

De facto standard for robot description in ROS. Virtually every ROS-based robot ships with a URDF. URDF parsers exist in C++, Python, Rust. The URDF→MJCF and URDF→SDF conversion paths are well-established but lossy.

**Governance**: Maintained by Open Robotics / OSRA. The `urdfdom` parser package is actively maintained (ROS Rolling, Jazzy, Lyrical distributions). No formal versioning scheme.

### Trajectory

URDF has been essentially frozen for years. The **URDF 2.0 proposal** by Sachin Chitta (2021) identified key gaps (sensors, named states, motors, vacuum joints) but was never formally adopted or implemented. The proposal remains a document with no implementation path.

Two recent academic efforts attempt to address URDF's limitations without replacing it:

- **URDF+** adds `<loop>` and `<coupling>` elements for closed kinematic chains, maintaining backward compatibility via a supplementary YAML file
- **URDD** (Universal Robot Description Directory, Dec 2025) adds a preprocessing layer that generates structured JSON/YAML modules from URDF, providing derived data (DOF mappings, kinematic chains, joint bounds, collision metadata). See Section 8 below.

Source: [URDF 2.0 proposal](https://sachinchitta.github.io/urdf2/), [URDD paper](https://arxiv.org/abs/2512.23135)

### Already solving our problems?

**Partially, but not the ML-facing ones.** URDF provides the joint names, types, and limits that our observed joint-ordering problem requires — but it provides no convention for how these map to policy action spaces or observation vectors. The joint-ordering mismatch between IsaacLab and MuJoCo exists precisely because each simulator loads the URDF and assigns its own internal ordering. URDF doesn't define a canonical ordering for ML purposes.

---

## 2. SDF (Simulation Description Format)

### What it describes

XML format originally developed for Gazebo. Superset of URDF's scope: supports multi-robot scenes, world descriptions (lights, ground planes, physics solver settings), native sensor definitions, and closed kinematic chains. Structured as `<world>` containing `<model>` elements.

Source: [SDFormat specification](https://sdformat.org/), [Gazebo SDF documentation](https://gazebosim.org/libs/sdformat/)

### What it doesn't

Same ML-facing gaps as URDF: no action/observation semantics, no camera role conventions for ML, no normalization metadata, no policy-level interface definitions.

### Type system & extensibility

Richer than URDF: native sensor types, physics solver configuration, light sources. Has a versioning scheme (spec versions 1.0–1.12+). Backward compatibility via version attributes.

### Community adoption

Standard format for Gazebo. All Gazebo worlds and models use SDF. However, outside the Gazebo ecosystem, adoption is limited. Most robot manufacturers ship URDF, not SDF. Drake also parses SDF.

### Trajectory

Active development continues under OSRA/Gazebo. Conversion from URDF to SDF is automatic (Gazebo does this internally) but lossy — fixed joint lumping can affect sensor poses.

### Already solving our problems?

**No more than URDF for ML workflows.** SDF solves the multi-robot and sensor description gaps in URDF, but doesn't address the ML-facing problems (action spaces, observation semantics, camera mapping, normalization).

---

## 3. MJCF (MuJoCo XML Format)

### What it describes

XML format for MuJoCo physics engine. Describes the full simulation specification: bodies, joints, actuators, sensors, contact parameters, tendons, equality constraints, and solver settings. Uses a nested `<body>` hierarchy rather than separate link/joint elements. A body with no joints is welded to its parent. Multiple joints per body are allowed (composite joints without dummy bodies).

Key design difference from URDF: MJCF defines the **physics contract** — how the simulation should behave — not just the robot's geometry and kinematics. Includes actuator dynamics (motors, position servos, velocity servos), tendon routing, and contact model parameters.

Source: [MuJoCo modeling documentation](https://mujoco.readthedocs.io/en/stable/modeling.html), [MuJoCo XML reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html)

### What it doesn't

- **MuJoCo-specific** — no other simulator reads MJCF natively (though Drake now parses a subset)
- **No ROS integration** — no concept of topics, services, or frames
- **No ML-facing metadata** — like URDF, no conventions for policy action/observation spaces
- **No camera roles for ML** — cameras are defined as sensors but without semantic roles (wrist, world, head)
- Angle units default to degrees (URDF uses radians) — a common conversion pitfall

### Type system & extensibility

Rich type system: 7 actuator types, 33 sensor types (including touch, accelerometer, gyro, force-torque, rangefinder, camera), tendon types, equality constraint types. Extensible via custom sensor plugins and user-defined parameters (`<custom>` element). Defaults system allows setting properties on all elements of a type.

### Community adoption

Standard format for MuJoCo, which is now open-source (Google DeepMind, Apache 2.0 since 2022). Widely used in RL research (Gymnasium/MuJoCo environments, robotics benchmarks). Drake parses MJCF. Growing adoption with MuJoCo's integration into Gazebo (announced ROSCon 2026).

### Trajectory

Active development by Google DeepMind. Key 2026 developments:

- **MJX** (MuJoCo on JAX/XLA) enables massively parallel simulation on GPU/TPU
- **mujoco_ros2_control** (by ros-controls and NASA-JSC) bridges MuJoCo into the ros2_control ecosystem, providing a hardware interface plugin. Supported across Humble, Jazzy, Kilted, Lyrical, Rolling.
- **ROSCon 2026**: PAL Robotics presenting `mujoco_ros2_control`, Intrinsic's challenge featuring cross-simulator support for MuJoCo and Isaac Lab with LeRobot integration
- MuJoCo physics integration into Gazebo announced — enabling MuJoCo as an alternative physics backend within Gazebo

Source: [mujoco_ros2_control (ros-controls)](https://github.com/ros-controls/mujoco_ros2_control), [ROSCon 2026](https://robottoday.com/article/ros-con-26), [mujoco_ros2_control docs](https://control.ros.org/rolling/doc/mujoco_ros2_control/doc/index.html)

### Already solving our problems?

**The ros2_control abstraction partially addresses the simulator-backend interoperability problem.** The `ros2_control` hardware interface pattern means the same ROS 2 controller stack works against MuJoCo, Gazebo, Isaac Sim (via topic bridge), or real hardware. Joint ordering is defined by the URDF's ros2_control `<hardware>` tag, providing a canonical mapping. However, this solves the *control-level* interoperability, not the *ML-level* (action spaces for policy training, observation semantics for VLAs, camera mapping for datasets).

---

## 4. OpenUSD (Universal Scene Description)

### What it describes

Scene description framework (originally Pixar, now under Alliance for OpenUSD / Linux Foundation). Describes 3D scenes with composition, layering, referencing, and variant systems. Key schemas relevant to simulation:

- **UsdPhysics**: Rigid body dynamics, joints (revolute, prismatic, spherical, distance, fixed, custom), collision shapes, articulation roots, mass/inertia
- **UsdGeomCamera**: Camera definitions with projection, clipping, focal length
- **UsdShade**: Materials and shading
- **PhysicsArticulationRootAPI**: Marks subtrees for reduced-coordinate articulation solvers

Composition engine allows layering — multiple USD files compose into a single scene via references, payloads, and sublayers. This enables separating geometry, physics, materials, and vendor-specific settings into independent files.

Source: [OpenUSD physics schema](https://openusd.org/dev/api/usd_physics_page_front.html), [AOUSD announcement](https://www.linuxfoundation.org/press/aousd_prmarch2026)

### What it doesn't

- No ML-facing metadata (models, datasets, normalization, action spaces)
- No ROS integration schemas (topics, services, frames) — that's REP-0158's addition
- Physics schemas cover rigid body dynamics but higher-level concepts (contact models, actuator dynamics) require vendor extensions
- Camera schemas describe projection geometry, not semantic roles for ML (wrist vs. world)

### Type system & extensibility

Highly extensible via API schemas (applied to existing prims without changing type) and custom properties with namespace prefixes. Variant sets allow multiple representations of the same prim (e.g., different LODs, different physics backends). Formal extension mechanism via registered schemas.

### Community adoption

AOUSD members include Pixar, Apple, NVIDIA, Adobe, Autodesk, Epic Games, and others. Core Specification 1.0 ratified. Rapidly growing in robotics via NVIDIA Omniverse/Isaac Sim. New working groups: Materials, Geometry, Physics, and Characters/Motion/Interactivity (formed March 2026).

### Trajectory

Active standardization push. Physics working group advancing deformable physics, B-rep geometry, FMI co-simulation mapping. Newton physics engine (NVIDIA) uses custom USD schemas. URDF/MJCF/CAD → USD conversion paths exist. Increasing convergence as the "universal" scene format.

### Already solving our problems?

**USD solves scene/environment description and asset layering, but not ML-facing interoperability.** The vendor-isolation layering pattern (neutral physics layer + vendor-specific tuning layers) is architecturally relevant to our problem — the same pattern could apply to ML metadata layers on top of physics assets. But USD itself has no concept of datasets, models, normalization, or policy interfaces.

---

## 5. REP-0158: OpenUSD Conventions for Simulation Asset Interoperability

### What it describes

A Draft ROS Enhancement Proposal defining conventions for using OpenUSD so that simulation assets work across Gazebo, Isaac Sim, Newton, Genesis, MuJoCo, and O3DE. Key contributions:

**Joint naming**: `ros:joint:name` custom property on `UsdPhysicsJoint` prims — "the single source of truth for ROS-facing joint identity." Governs `FollowJointTrajectory` mapping, `JointState` field names, ros2_control integration, and format conversion (e.g., MJCF `<joint name="">`).

**Camera conventions**: Decouples physical sensor from optical interface via a child `UsdGeomXform` rotated 180° around X-axis (USD's -Z to ROS's +Z optical frame). All `RosTopicAPI` and `RosFrameAPI` schemas attach to this optical frame prim.

**Vendor isolation**: Neutral layer (`physics.usda`) uses only core `UsdPhysics` schemas. Engine-specific parameters isolated in vendor-prefixed layers (`physx.usda`, `mujoco.usda`). `physics:` namespace reserved for AOUSD-ratified schemas only.

**Asset layering (ETL pipeline)**: Functional decomposition into base layers (`geometries.usdc`, `materials.usda`, `base.usda`), feature layers (`physics.usda`, `ros.usda`), and a lightweight entry point. Schema layers use ASCII `.usda`; heavy data uses binary `.usdc`.

**ROS integration schemas**: `RosContextAPI` (namespace, domain ID), `RosTopicAPI` (publishers/subscribers with QoS), `RosServiceAPI`, `RosActionAPI`, `RosFrameAPI` (TF publishing). Declarative, engine-agnostic.

**Units/conventions**: MKS units, Z-up, quaternion-only orientation, explicit mass/inertia in eigendecomposed form.

Authors: Robotec.ai (Adam Dabrowski, Mateusz Zak, Michal Pelka), NVIDIA (Ayush Ghosh, Renato Gasoto), Ekumen (Franco Cipollone). Posted March 2026, updated July 2026.

Source: [REP-0158 full text](https://reps.openrobotics.org/rep-0158-2006/), [ROS Discourse discussion](https://discourse.openrobotics.org/t/draft-rep-158-openusd-conventions-for-simulation-asset-interoperability-in-open-source-robotics/55526)

### What it doesn't

- No ML-facing metadata (models, datasets, normalization, action spaces, observation semantics)
- No concept of camera **roles** for ML policies (wrist camera, head camera, world camera as used by VLAs) — only optical frame conventions for ROS TF
- No dataset or training metadata
- No compatibility checking between artifacts
- No action space semantics (7-DOF EE delta, 29-DOF whole-body, etc.)

### Type system & extensibility

Leverages USD's extensibility (API schemas, namespace prefixes). Vendor prefixes (`mujoco:`, `isaac:`, `physx:`) provide clean extension points. New ROS interface types (`RosTopicAPI`, etc.) are applied API schemas that can be added to any prim.

### Community adoption

**Draft status** — not yet accepted as an official REP. Led by Robotec.ai (the company behind O3DE ROS integration and significant Gazebo contributions) with NVIDIA and Ekumen involvement. The three supporting simulators (Gazebo, Isaac Sim, O3DE) represent significant coverage but MuJoCo and Genesis support is aspirational.

### Trajectory

The most ambitious cross-simulator standardization effort in the ROS ecosystem. If adopted, it would establish USD as the canonical asset format for ROS-facing simulation. However, Draft status means it could change significantly. The vendor-isolation and layering patterns are architecturally mature.

### Already solving our problems?

**Partially — for joint naming and cross-simulator asset portability, but not for ML workflows.** `ros:joint:name` as the single source of truth for joint identity addresses part of the joint-ordering problem: if all simulators use this convention, the joint naming becomes consistent. But the mapping from joint names to policy action-space dimensions (which joints map to which action vector indices) remains unspecified. REP-0158 defines how to describe a robot for simulation; it doesn't define how a policy interacts with that robot.

---

## 6. NVIDIA SimReady Foundation

### What it describes

Open specification (Apache 2.0) layered on OpenUSD defining what properties simulation assets must have for different runtimes. Three-layer architecture:

1. **Requirements** — atomic, machine-checkable rules on USD prims (e.g., `RB.COL.001`: every rigid shape needs `CollisionAPI`)
2. **Features** — bundles of requirements for a use case, with dependency resolution and runtime variants (`_BASE_NEUTRAL`, `_BASE_PHYSX`, `_ROBOT_CORE_ISAAC`)
3. **Profiles** — bundles of features for complete scenarios (e.g., `Robot-Body-Isaac`)

Profiles as of 2026.06.0: Prop-Robotics-{Neutral,Physx,Isaac}, Robot-Body-{Neutral,Runnable,Isaac}.

Features cover: core structure, physics (posable bodies, RBD, multi-body, grasp), materials, semantic labels, Isaac Sim composition, robot-specific (core robot, driven joints, base articulation), packaging.

Tooling: `simready-validate` CLI, Python validation API, feature adapters for cross-profile conversion (`workspace upgrade --output_profile <target>`), CI/CD integration.

Source: [SimReady Foundation docs](https://nvidia.github.io/simready-foundation/), [GitHub repo](https://github.com/nvidia/simready-foundation)

### What it doesn't

Same ML-facing gaps as other asset-level standards:

- No model manifests, checkpoint formats, normalization strategies
- No dataset manifests, feature schemas, stats availability
- No benchmark/evaluation definitions
- No cross-artifact compatibility checking (validates single assets, not pairs)
- No action/observation semantics, camera roles for ML
- No workflow composition

### Type system & extensibility

Well-designed: requirements → features → profiles composition. Vendor namespace prefixes (`simready:`, `physx:`, `isaac:`, third-party prefixes). Versioned features and immutable published profiles. `@register_rule()` decorator for custom validators. Feature adapters for cross-profile transformation.

### Community adoption

NVIDIA-initiated, Apache 2.0. GitHub repo created March 2026. Recent agent skills integration (2026) suggests active development. AOUSD governance connection via OpenUSD. No external contributors documented. Several features still at version 0.1.0 — early.

### Trajectory

Expanding: AIF 0.1.0 Preview in latest docs. Agent skills for automated asset conformance. Newton runtime support planned. Evolving from internal NVIDIA tool (`omni:simready:*` → `simready:*` migration) to open spec.

### Already solving our problems?

**No — SimReady operates at the asset validation layer, not the ML workflow layer.** Its three-layer architecture (requirements → features → profiles) is a well-designed pattern worth studying for our own validation approach, but the actual content is about physics/geometry correctness, not ML compatibility.

---

## 7. ROS 2 Interface Definitions (.msg, .srv, .action)

### What it describes

ROS 2's Interface Definition Language (IDL) defines message types for inter-process communication. Three kinds:

- **.msg** — data structures for pub/sub topics
- **.srv** — request/response pairs for synchronous services
- **.action** — goal/result/feedback triples for long-running operations

Common interface packages provide standardized message types:

- **sensor_msgs**: `Image`, `PointCloud2`, `Imu`, `JointState`, `CameraInfo`, `LaserScan`, `Range`, `CompressedImage`, `MagneticField`
- **geometry_msgs**: `Pose`, `Twist`, `Wrench`, `Transform`, `Point`, `Quaternion`, `Vector3`
- **trajectory_msgs**: `JointTrajectory`, `JointTrajectoryPoint`
- **control_msgs**: `FollowJointTrajectory` (action), `JointJog`

Type system supports: primitives (bool, byte, float32/64, int8–64, string), fixed-size arrays, bounded arrays (`int32[<=5]`), bounded strings (`string<=5`), default values, constants.

Source: [ROS 2 IDL design](https://design.ros2.org/articles/interface_definition.html), [common_interfaces repo](https://github.com/ros2/common_interfaces), [ROS 2 interface concepts](https://docs.ros.org/en/lyrical/Concepts/Basic/About-Interfaces.html)

### What it doesn't

- **Control-plane only** — defines how data flows between ROS nodes, not ML policy interfaces
- No concept of action spaces in the RL/VLA sense (policy input/output dimensions, semantics)
- No dataset descriptions, normalization metadata, or model metadata
- No camera roles for ML (wrist vs. world) — `sensor_msgs/CameraInfo` describes intrinsics, not semantic role
- No compatibility checking between different message types

### Type system & extensibility

Well-defined IDL with bounded arrays/strings (added in ROS 2), default values, and nesting. Custom message packages are the standard extension mechanism. However, ROS 2 deprecated `std_msgs` in favor of "semantically meaningful message types" — this push toward semantic typing is philosophically aligned with what Physical AI contracts need.

### Community adoption

Universal in ROS. Every ROS 2 node uses these interface types. The `sensor_msgs` and `geometry_msgs` packages are the vocabulary of robotics data exchange in ROS.

### Already solving our problems?

**ROS 2 interfaces provide a reusable vocabulary for sensor/actuator data types, but not for ML policy semantics.** `sensor_msgs/JointState` defines joint positions/velocities/efforts; `sensor_msgs/Image` defines image data. These could be referenced as building blocks for describing what a model expects as input — e.g., "this model takes `sensor_msgs/Image` from two cameras and `sensor_msgs/JointState` for 7 joints." But ROS interfaces don't address normalization, action chunking, camera role mapping, or cross-artifact compatibility.

---

## 8. URDD (Universal Robot Description Directory)

### What it describes

Research project (Dec 2025, arXiv:2512.23135) that generates a directory of derived, structured JSON/YAML modules from URDF (or other formats). 15 modules including: DOF mappings, kinematic chains, joint bounds, mesh approximations (convex hulls, bounding boxes), collision skip matrices, link distance statistics, and learned collision representations.

Operates **orthogonally** to base formats — sits on top of URDF/SDF/MJCF/USD. A Rust converter automatically generates URDDs from URDFs.

Key feature: eliminates redundant preprocessing. Forward kinematics from URDD requires **0 lines** of preprocessing code (vs. 315–3,784 lines in frameworks like Klampt, Drake, MuJoCo).

Source: [URDD paper](https://arxiv.org/html/2512.23135v2)

### What it doesn't

- **No ML-facing metadata** — the paper doesn't discuss action spaces, observation semantics, or policy interfaces
- No camera role definitions
- No normalization metadata
- No cross-artifact compatibility (model-dataset, model-robot)
- Early research — no community adoption, no governance, no ecosystem

### Relevance

The DOF module (joint indices, DOF mappings, bounds) and chain module (kinematic hierarchies) contain exactly the structural information needed to define a robot's action space. The modularity pattern (independent JSON/YAML modules with versioning) is architecturally relevant. But URDD focuses on geometry/kinematics preprocessing, not ML workflow integration.

---

## 9. simulation_interfaces (ROS 2 Package)

### What it describes

ROS 2 package providing standardized interfaces for interacting with simulators. Lives under `ros-simulation` GitHub organization. Defines messages, services, and actions for common simulation tasks:

- **Feature discovery**: `GetSimulatorFeatures` (recommended to implement first)
- **Entity management**: `GetEntityBounds`, `SetEntityInfo`, `GetSpawnables`
- **World management**: `LoadWorld`, `UnloadWorld`, `GetAvailableWorlds`, `GetCurrentWorld`
- **Pose management**: `GetNamedPoses`, `GetNamedPoseBounds`

Three simulators implement it: Gazebo Jetty, NVIDIA Isaac Sim 5.0, O3DE 2505. A shared warehouse scene from Isaac Sim runs across all three, demonstrating interoperability.

Source: [simulation_interfaces repo](https://github.com/ros-simulation/simulation_interfaces), [Robotec.ai blog](https://www.robotec.ai/blog/robotics/standardizing-simulation-interfaces-for-open-source-robotics)

### What it doesn't

- **Simulation control only** — spawn, reset, step. No ML-facing metadata.
- No dataset/model/benchmark concepts
- No action space or observation space semantics
- No camera role conventions for ML
- Still evolving (e.g., `Result.msg` noted as temporary)

### Already solving our problems?

**No — this solves a different problem (simulator API interoperability, not ML artifact compatibility).** However, the multi-simulator implementation pattern (one interface definition, three independent implementations) is evidence that cross-simulator standardization is actively happening in the ROS community.

---

## 10. ros2_control Hardware Interface Abstraction

### What it describes

ROS 2 framework that provides a common API for controlling robots across different backends (real hardware, Gazebo, MuJoCo, Isaac Sim). Key pattern: a `<hardware>` tag in the URDF specifies which plugin to use, and controllers (position, velocity, effort) work identically regardless of backend.

Two integration patterns:

- **Direct plugin**: `MujocoSystemInterface` (ros-controls + NASA-JSC), `GazeboSystem` — native integration, low latency
- **Topic-based bridge**: `TopicBasedSystem` for simulators without native plugins (e.g., Isaac Sim via `/isaac_joint_states` and `/isaac_joint_commands` topics)

Source: [mujoco_ros2_control](https://control.ros.org/rolling/doc/mujoco_ros2_control/doc/index.html), [ros-controls/mujoco_ros2_control](https://github.com/ros-controls/mujoco_ros2_control)

### What it doesn't

- **Control-level abstraction only** — joint-level position/velocity/effort commands
- No ML policy interfaces (action spaces, observation vectors, VLA inputs)
- No dataset or model metadata
- No normalization, no camera mapping

### Already solving our problems?

**Partially — ros2_control solves the simulator-backend interoperability problem for control.** The joint ordering is defined by the URDF's `<ros2_control>` hardware tag, providing a canonical mapping. This is exactly the kind of abstraction that prevents the joint-ordering problem — but it operates at the control level, not the ML training level. An RL policy training in Isaac Lab doesn't go through ros2_control; it uses Isaac Lab's internal joint ordering. The gap is between the ros2_control world (deployment) and the ML training world (Isaac Lab, LeRobot, MuJoCo Gymnasium environments).

---

## 11. MuJoCo Menagerie

### What it describes

Curated collection of high-quality robot models for MuJoCo (Google DeepMind, Apache 2.0). 63+ models across 11 categories: humanoids, quadrupeds, arms, end-effectors, mobile manipulators, drones, etc.

Per-model contents: `<model>.xml` (MJCF), `scene.xml` (model + ground plane + lighting), meshes, README with provenance, license, preview image. Optional MJX variants for GPU-accelerated simulation.

Python API (`pip install mujoco-menagerie`): `mm.load('unitree_go2')` downloads, caches, and compiles; `mm.get('unitree_go2').spec()` returns editable `MjSpec`. CLI viewer: `uvx mujoco-menagerie view <model>`.

Quality grading system defined (A+ to C based on system identification rigor) but not yet applied.

Source: [MuJoCo Menagerie GitHub](https://github.com/google-deepmind/mujoco_menagerie)

### What it doesn't

- **No ML-facing metadata** — no action space definitions, observation conventions, or policy interface descriptions per model
- **No JSON/YAML metadata** beyond MJCF XML — actuator and sensor configurations are embedded in XML files
- Action space and observation space details are derived **at runtime** by Gymnasium environment wrappers, not declared in the model
- No camera role conventions

### Already solving our problems?

**No — Menagerie provides high-quality MJCF models but no ML-facing metadata.** When you load a Menagerie model into Gymnasium (e.g., `Ant-v5` with a Menagerie XML), the action/observation spaces are defined by the Gymnasium wrapper, not by the model itself. This means the same robot model produces different action/observation interfaces depending on which wrapper is used — exactly the kind of implicit convention that makes interoperability fragile.

---

## 12. Gymnasium / Farama Foundation

### What it describes

Gymnasium (successor to OpenAI Gym) defines the standard Python API for RL environments. Key concepts:

- **`action_space`** and **`observation_space`** — `Space` objects (Box, Discrete, Dict, Tuple, etc.) with `dtype` and `shape`
- **`GoalEnv`** extension (Gymnasium-Robotics) — observation dict with keys `observation`, `achieved_goal`, `desired_goal`
- **`step(action) → (obs, reward, terminated, truncated, info)`** — standard RL loop
- **Environment specification** — can serialize/recreate environment config

Source: [Gymnasium docs](https://gymnasium.farama.org/), [Gymnasium-Robotics docs](https://robotics.farama.org/)

### What it doesn't

- **Runtime API, not metadata format** — spaces are defined in Python code, not in a declarative schema
- No dataset descriptions, model metadata, or normalization specs
- No cross-environment compatibility checking
- No declarative description of what a policy expects as input (it's implicit in the environment code)

### Already solving our problems?

**Gymnasium defines the de facto vocabulary for action/observation spaces in RL, but only at the Python API level.** There's no declarative metadata format that says "this robot's action space is Box(-1, 1, (7,)) representing [dx, dy, dz, ax, ay, az, gripper]." The semantics are implicit in the wrapper code. This is a concrete gap: if we want to check whether a trained policy is compatible with a robot, we need a declarative version of what Gymnasium provides programmatically.

---

## Cross-Cutting Analysis

### The joint-ordering problem specifically

Our observed problem: IsaacLab and MuJoCo use different joint orderings for the same robot, requiring hand-coded remapping arrays.

**What exists today:**

- URDF defines joint names and tree structure, but not a canonical ordering
- `ros:joint:name` (REP-0158) defines a single source of truth for ROS-facing joint identity
- `ros2_control` defines joint ordering via URDF `<hardware>` tags — but only for the control stack, not for ML training
- MuJoCo Menagerie provides high-quality MJCF models but no canonical joint ordering convention
- Gymnasium wrappers define action/observation spaces at runtime, derived from the loaded model

**The gap:** There is no standard that maps *joint names* to *action-space indices* for ML policies. REP-0158's `ros:joint:name` gets us joint identity; what's missing is the convention for "joint X maps to action dimension 3." This is inherently a policy-level concern, not an asset-level one.

### The camera-mapping problem specifically

Our observed problem: dataset camera names don't match model input slot names (CAMERA_MAP).

**What exists today:**

- USD cameras define projection geometry but not semantic roles
- REP-0158 defines optical frame conventions but not ML camera roles (wrist vs. world vs. head)
- `sensor_msgs/CameraInfo` describes intrinsics, not semantic role
- LeRobot's `info.json` names cameras by dataset-specific strings
- pi0.5's `config.json` names cameras by model input slot names

**The gap:** There is no standard vocabulary for camera roles in ML policy context. "wrist_camera", "agentview", "observation.images.wrist" — these are ad hoc names used by different projects with no alignment. This is a genuine gap that falls between communities: ROS defines camera intrinsics, USD defines camera geometry, but neither defines "this camera provides the wrist view for a manipulation policy."

### The cross-simulator interoperability trend

The ROS community is actively working on cross-simulator interoperability, but focused on the **simulation/control layer**, not the **ML training layer**:

1. **REP-0158** — asset format standardization across 6+ simulators
2. **simulation_interfaces** — common API for spawning, resetting, stepping across Gazebo/Isaac Sim/O3DE
3. **ros2_control** — hardware abstraction across Gazebo/MuJoCo/Isaac Sim/real hardware
4. **MuJoCo in Gazebo** — MuJoCo as physics backend within Gazebo (ROSCon 2026)

These are significant convergence efforts, but they solve the "same robot, different simulator" problem at the control level. The "same model, different training framework" problem (LeRobot vs. Isaac Lab vs. custom training loops) remains unaddressed.

### Where existing standards could be extended vs. where new spec is needed

| Problem | Could extend existing | Needs new spec | Notes |
| --- | --- | --- | --- |
| Joint ordering for ML | REP-0158 or URDD | Possibly | Add `ml:action_index` property to joints? |
| Camera roles for ML | REP-0158 | Possibly | Add camera role vocabulary (wrist, world, head) |
| Action space semantics | Gymnasium | Likely | Declarative version of Gymnasium's Space objects |
| Observation semantics | Gymnasium | Likely | Same — declarative, not programmatic |
| Model-dataset compatibility | — | Yes | No existing standard comes close |
| Normalization metadata | — | Yes | Specific to ML training workflows |
| Checkpoint loading semantics | — | Yes | Highly framework-specific |

---

## Summary of Standards Landscape

| Standard | Scope | ML-Facing? | Momentum | Governance |
| --- | --- | --- | --- | --- |
| URDF | Robot kinematics/geometry | No | Frozen (de facto standard) | OSRA |
| SDF | Robot + world + sensors | No | Active (Gazebo) | OSRA |
| MJCF | Full simulation spec | No | Active (DeepMind) | Google DeepMind |
| OpenUSD | Scene description | No | High (AOUSD/Linux Foundation) | AOUSD |
| REP-0158 | USD conventions for sim interop | No | Moderate (Draft) | OSRA (proposed REP) |
| SimReady | USD asset validation profiles | No | Moderate (NVIDIA) | NVIDIA + AOUSD |
| ROS 2 interfaces | Inter-process communication | No | High | OSRA |
| simulation_interfaces | Simulator API standardization | No | Growing | ros-simulation |
| ros2_control | Hardware abstraction | No | High | ros-controls |
| MuJoCo Menagerie | Robot model collection | No | High | Google DeepMind |
| URDD | Derived robot metadata | No | Research only | Academic |
| Gymnasium | RL environment API | Partial (runtime only) | High (de facto standard) | Farama Foundation |

**Key finding: No existing standard addresses the ML-facing interoperability layer.** The robot/environment description space is well-covered and actively converging. The gap is between the asset/control layer (where standards exist) and the ML training layer (where conventions are ad hoc and framework-specific).
