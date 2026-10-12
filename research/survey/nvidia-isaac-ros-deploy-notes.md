# NVIDIA Isaac ROS Deploy & LEAPP: Prior Art Survey Notes

## Overview

Isaac ROS Deploy is a set of ROS 2 packages (released April 2026) that deploys neural-network policies trained in NVIDIA Isaac Lab on real and simulated robots. It bridges the gap between a Python training stack and a robot control system by loading LEAPP bundles, running ONNX inference through NVIDIA Triton, mapping policy terms onto ROS topics or `ros2_control` interfaces, and optionally gating outputs through a safety controller.

LEAPP (Lightweight Export Annotations for Policy Pipelines) is the complementary export library that packages trained PyTorch policies — including all surrounding pre/post-processing logic — into deployment-ready bundles with a declarative YAML pipeline specification.

Together they form a two-stage architecture: **LEAPP export** (training-side) → **Isaac ROS Deploy** (deployment-side), with the LEAPP bundle as the interface artifact between them.

Sources:

- [Isaac ROS Deploy GitHub](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_deploy)
- [LEAPP documentation](https://nvidia-isaac.github.io/leapp/)
- [LEAPP GitHub](https://github.com/nvidia-isaac/leapp)
- [Isaac ROS Deploy docs](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_deploy/index.html)
- [GR00T E2E blog post](https://developer.nvidia.com/blog/develop-humanoid-robot-policies-end-to-end-with-nvidia-isaac-gr00t/)

## LEAPP Bundle Format

### What Gets Exported

LEAPP traces real execution of a PyTorch policy pipeline and captures the full inference path: preprocessing (normalization, feature transforms), model inference, postprocessing (action chunking, unit conversion, clipping), recurrent state management, and constant tensors. Each annotated stage is exported as an independent compiled model artifact.

This is significant because it eliminates the common need to reimplement pre/post-processing logic in a different language or runtime for deployment. The bundle is self-contained.

### Bundle Directory Layout

```
sample_pipeline/
  sample_pipeline.yaml       # Pipeline specification (the "contract")
  obs_processor.pt           # Per-node compiled model (TorchScript)
  policy.pt                  # Per-node compiled model
  sample_pipeline.png        # Visualization (optional)
  sample_pipeline_initial_values.safetensors  # Feedback state init (optional)
```

### Pipeline YAML Schema (pipeline.yaml)

The YAML file contains three top-level sections:

**`models`** — Per-node model contracts with tensor port definitions:

```yaml
models:
  obs_processor:
    inputs:
    - name: joint_pos
      dtype: float32
      shape: [6]
      type: tensor
      kind: state/joint/position                    # semantic annotation
      element_names: [left_hip, left_knee, left_ankle,
                      right_hip, right_knee, right_ankle]  # joint identity & ordering
    outputs:
    - name: obs_features
      dtype: float32
      shape: [18]
      type: tensor
    parameters:
      backend: jit              # loading mechanism (jit, pt2, onnx)
      model_path: obs_processor.pt
      sha256sum: ...
  policy:
    inputs:
    - name: obs_features
      dtype: float32
      shape: [18]
      type: tensor
    outputs:
    - name: joint_targets
      dtype: float32
      shape: [6]
      type: tensor
      kind: target/joint/position
      element_names: [left_hip, left_knee, left_ankle,
                      right_hip, right_knee, right_ankle]
    parameters:
      backend: jit
      model_path: policy.pt
```

**`pipeline`** — Graph wiring and runtime metadata:

```yaml
pipeline:
  data_flow:
    obs_processor/obs_features: [policy/obs_features]   # forward edges
  feedback_flow: {}           # cross-tick state edges (cyclic)
  inputs:
    obs_processor: [joint_pos, joint_vel, orientation, cmd_vel]  # external inputs
  outputs:
    policy: [joint_targets]   # external outputs to publish
  initial_values: sample_pipeline_initial_values.safetensors  # optional
  configs:
    frequency: 50             # execution rate in Hz
```

**`system information`** — Export provenance:

```yaml
system information:
  leapp version: 0.7.2
  leapp config version: '1.3'
  torch version: 2.9.1+cu128
  python version: 3.12.9
  cuda version: '12.8'
  os: Ubuntu 24.04
```

Source: [LEAPP Generated Configs](https://nvidia-isaac.github.io/leapp/generated_configs.html)

### Tensor Semantic Annotations

LEAPP defines three optional semantic annotation fields on tensor ports:

| Field | Purpose | Example |
| ------- | --------- | --------- |
| `kind` | Physical role of the tensor | `state/joint/position`, `target/joint/position` |
| `element_names` | Identity and ordering of elements along tensor axes | `[left_hip, left_knee, ...]` |
| `temporal_period_ms` | Sample spacing for temporal axes | `100` (ms) |
| `__temporal_axis__` | Marks an axis as time-like within `element_names` | Used in action-chunked outputs |

**`kind`** is used "to route values to the right subscriber, publisher, command interface, or adapter." It follows a hierarchical naming convention (`state/joint/position`, `target/joint/position`).

**`element_names`** enables "reorder robot messages into model order, verify that expected joints are present." This directly addresses the joint-ordering mismatch problem identified in our research.

**Temporal semantics** example for action-chunked outputs:

```yaml
outputs:
- name: actions
  dtype: float32
  shape: [4, 3]
  type: tensor
  element_names:
  - __temporal_axis__
  - [hip, knee, ankle]
  temporal_period_ms: 100
```

### Observations on LEAPP's Data Model

The LEAPP pipeline.yaml is essentially a **typed, semantically-annotated dataflow graph contract**. It declares:

1. What inputs the policy needs (tensor shapes, dtypes, semantic roles)
2. What outputs it produces (with the same metadata)
3. How internal processing stages connect
4. Runtime parameters (frequency, initial state)
5. Provenance (versions, checksums)

This is the closest thing to a "policy contract" in the NVIDIA ecosystem. It serves exactly the role our research has been exploring: machine-readable metadata that enables automatic compatibility checking between policies and robot configurations.

## Isaac ROS Deploy Architecture

### Core Components

1. **LEAPP Bundle Loader** — Ingests the exported bundle (YAML + model files)
2. **Triton Inference Server** — Executes ONNX models
3. **Converter Subsystem** — Translates between policy tensor terms and robot-specific ROS data formats. Users can add custom converters via an extension interface. A converter catalog exists (detailed docs behind auth wall).
4. **Safety Controller** — Optional output gating layer before commands reach the robot
5. **InferenceController** — Core controller with debug topics for flattened model inputs and selected output tensors (added September 2026)

### Deployment Modes

Two runtime modes abstract over how the policy interfaces with the robot:

| Mode | Integration Point | Use Case |
|------|-------------------|----------|
| **ROS 2 Node Graph** (`ros2_nodes_runtime`) | Publishes/subscribes to ROS 2 topics | Standard ROS 2 architecture |
| **ROS 2 Control** (`ros2_control_runtime`) | Embedded in `ros2_control` loop via hardware interfaces | Real-time control loops |

### Supported Backends

| Backend | Status | Notes |
| --------- | -------- | ------- |
| Real hardware | Supported (July 2026) | GR00T workflow on Unitree G1 |
| MuJoCo | Supported (primary tutorial target) | AGILE whole-body policy on simulated G1 |
| Isaac Sim 6.0 | Added August 2026, then removed in September 2026 release | Volatile support status |

### How Real vs. Sim Abstraction Works

The abstraction is achieved through **ROS 2's own abstraction layer**, not through Isaac ROS Deploy itself:

- For **real robots**: ROS 2 hardware drivers publish sensor data to topics and subscribe to command topics. Isaac ROS Deploy publishes/subscribes to the same topic namespace.
- For **MuJoCo**: NVIDIA provides a `ros2_control` hardware interface plugin for MuJoCo that publishes simulated sensor data and accepts commands as if they were hardware interfaces.
- For **Isaac Sim**: Isaac Sim has ROS 2 Bridge Extensions that connect simulated sensors/actuators to the ROS 2 topic graph.

**The sim/real abstraction is not a feature of Isaac ROS Deploy itself — it delegates to ROS 2's existing hardware abstraction.** Isaac ROS Deploy is backend-agnostic because it only speaks ROS 2 topics and `ros2_control` interfaces.

Source: [Isaac ROS Deploy docs](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_deploy/index.html)

### Supported Platforms

| Platform | Hardware |
| ---------- | ---------- |
| Jetson Thor (T5000/T4000) | JetPack 7.2 |
| x86_64 | Ampere+ GPU, 8 GB+ VRAM |
| DGX Spark | DGX OS 7.2.3 |

## The GR00T E2E Workflow Data Pipeline

The end-to-end workflow from the GR00T tutorial illustrates the data format conversions involved:

| Stage | Tool | Input Format | Output Format |
| ------- | ------ | ------------- | --------------- |
| Environment Setup | Isaac Lab-Arena | — | USD scene, task config |
| Data Collection | Isaac Teleop + CloudXR | — | HDF5 (joint pos, actions, camera RGB) |
| Data Conversion | `convert_hdf5_to_lerobot.py` | HDF5 | LeRobot (parquet + MP4 + metadata JSON) |
| Post-Training | GR00T 1.7 fine-tuning | LeRobot dataset | Checkpoint (PT) |
| LEAPP Export | `gr00t-leapp-export` | Checkpoint | LEAPP bundle (YAML + ONNX/PT) |
| Deployment | Isaac ROS Deploy | LEAPP bundle | ROS 2 topics / ros2_control |

The conversion step uses a YAML config to map field names:

```yaml
state_name_sim: "robot_joint_pos"
action_name_sim: "processed_actions"
pov_cam_name_sim: "robot_head_cam_rgb"
```

This mapping config is **per-embodiment and per-task**, and the `--embodiment-tag` flag registers custom embodiments. The design note from the tutorial: "the AGILE WBC and PinkIK produce the joint-space targets that become the policy's training signal," meaning the whole-body controller choice during data collection directly determines the action distribution the policy learns.

Source: [GR00T E2E blog](https://developer.nvidia.com/blog/develop-humanoid-robot-policies-end-to-end-with-nvidia-isaac-gr00t/)

## Related: RoboVerse/MetaSim (Simulation Abstraction)

While Isaac ROS Deploy handles the **deployment** abstraction (policy → robot), the **simulation** abstraction (same task across simulators) is addressed by a separate effort: RoboVerse's MetaSim.

MetaSim provides a unified interface across MuJoCo, Isaac Sim/Isaac Lab, SAPIEN, Genesis, PyBullet, and Newton. Its architecture has three layers:

1. **Scenario Config** — Simulator-agnostic specification of scene contents (robots, objects, lighting, physics)
2. **Handlers** — Backend-specific implementations that compile configs into native formats (MJCF for MuJoCo, USD for Isaac Sim, URDF for others)
3. **Task Env** — Gym-style API wrapper

**Important caveat**: The cross-simulator equivalence claim has known limitations. Different physics engines violate conservation laws differently, so identical initial states do not produce identical trajectories. This is acknowledged in the RoboVerse paper itself.

MetaSim is a community project (30+ authors, Apache 2.0, RSS 2025) — not an NVIDIA product. NVIDIA's Newton physics engine is one of the supported backends.

Sources:

- [RoboVerse/MetaSim GitHub](https://github.com/RoboVerseOrg/MetaSim)
- [RoboVerse paper (arXiv)](https://arxiv.org/html/2504.18904v1)
- [MetaSim docs](https://roboverse.wiki/metasim/)

## Relevance to Contracts Research

### What LEAPP's pipeline.yaml Validates

The LEAPP pipeline YAML is strong evidence that a **machine-readable policy contract** delivers value. It captures exactly the metadata needed to deploy a policy on a robot without manual glue code:

1. **Tensor shape and dtype** — What the model expects as input and produces as output
2. **Semantic role (`kind`)** — Whether a tensor represents joint positions, targets, camera images, etc.
3. **Element identity and ordering (`element_names`)** — Which joints, in which order — directly solving the joint-ordering mismatch problem
4. **Temporal structure** — Action chunk size and sample period
5. **Graph topology** — How pre/post-processing connects to inference
6. **Runtime requirements** — Execution frequency, initial state

### What LEAPP Does Not Cover

LEAPP is policy-centric. It does not describe:

- **The robot's capabilities** — What joints exist, their limits, kinematics. This comes from URDF/MJCF, not LEAPP.
- **Dataset compatibility** — Whether a dataset was collected on a compatible embodiment with compatible action conventions.
- **Normalization strategy provenance** — The normalization is *baked into the exported model*, but which strategy was used (quantile, mean/std, identity) and what statistics it was computed from are not declared.
- **Camera intrinsics/extrinsics** — Image tensors have shape and semantic kind but no camera calibration metadata.
- **Cross-embodiment compatibility** — Whether a policy trained on embodiment A can run on embodiment B with joint remapping.

### Hypothesis: The Contract Gap

LEAPP solves the **policy → deployment** interface well. The remaining contract gaps are:

1. **Dataset ↔ Policy compatibility** — Is a dataset suitable for training/fine-tuning a given policy architecture for a given embodiment? (This is where the HDF5-to-LeRobot conversion configs live, but they're ad-hoc YAML, not a standardized contract.)
2. **Robot description ↔ Policy compatibility** — Does the robot's joint set, ordering, and limits match what the policy expects? (LEAPP's `element_names` partially addresses this, but the robot side has no matching contract.)
3. **Simulation ↔ Real equivalence** — Is the sim environment sufficiently faithful to transfer? (MetaSim's Scenario Config attempts this for the sim side, but there's no real-world counterpart.)

These gaps align with the interoperability friction points identified elsewhere in our survey.

## Release Timeline

| Date | Milestone |
| ------ | ----------- |
| April 2026 | Isaac ROS Deploy initial release |
| July 2026 | Bring-Your-Own Policy/Embodiment, GR00T workflow on real Unitree G1 |
| August 2026 | Isaac Sim 6.0 deployment support for G1 AGILE |
| September 2026 | InferenceController debug topics, troubleshooting guides, LEAPP runtime |

## See Also

- [nvidia-e2e-workflow-notes.md](nvidia-e2e-workflow-notes.md) — Detailed analysis of the GR00T E2E tutorial's data models, conversion points, and implicit contracts at each stage boundary. Covers the full pipeline from teleoperation through deployment with specific format schemas and interoperability observations.
