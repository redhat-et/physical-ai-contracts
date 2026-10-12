# NVIDIA GR00T E2E Workflow: Data Models & Conversion Points

## Overview

This document analyzes the [NVIDIA GR00T + Unitree G1 End-to-End Tutorial](https://docs.nvidia.com/learning/physical-ai/gr00t-e2e-workflow/latest/) as a case study in Physical AI workflow interoperability. The tutorial walks through a complete sim-first-to-deployment pipeline for a humanoid pick-and-place task, making explicit the data formats, conversion steps, configuration schemas, and implicit contracts at each stage boundary.

For LEAPP bundle format and Isaac ROS Deploy architecture details, see [nvidia-isaac-ros-deploy-notes.md](nvidia-isaac-ros-deploy-notes.md).

## Workflow Structure

The tutorial implements two parallel paths through five stages:

```
                    ┌─ Isaac Lab-Arena ─── HDF5 ──┐
Teleoperation ──────┤                              ├── LeRobot ── GR00T Fine-Tuning ── Checkpoint
                    └─ Real G1 + ROS 2 ── MCAP ──┘                                       │
                                                                                          │
                    ┌─ Isaac Lab-Arena (closed-loop eval) ◄───────────────────────────────┤
Evaluation ─────────┤                                                                     │
                    └─ Real G1 ◄── Isaac ROS Deploy ◄── LEAPP Export ◄───────────────────┘
```

Each arrow represents a format boundary where data models change and conversion logic is required.

Sources: [Tutorial index](https://docs.nvidia.com/learning/physical-ai/gr00t-e2e-workflow/latest/), [Concepts overview](https://docs.nvidia.com/learning/physical-ai/gr00t-e2e-workflow/latest/getting-started/concepts-overview.html)

## Stage 1: Teleoperation & Data Collection

### Simulation Path

Isaac Lab-Arena records demonstrations as **HDF5** files. The operator controls a simulated Unitree G1 via XR headset (Meta Quest 3 or PICO 4 Ultra), streamed through CloudXR. The data flow is:

```
XR controller poses → PinkIK (inverse kinematics) → 43-DOF joint-space targets → HDF5 recording
```

Key fields recorded in HDF5:

| HDF5 Path | Content | Notes |
| --- | --- | --- |
| `observations/camera_obs/robot_head_cam_rgb` | Ego RGB camera frames | Head-mounted camera |
| `robot_joint_pos` | Joint position state | Proprioceptive observation |
| `processed_actions` | 43-DOF joint-space action targets | PinkIK output, not raw EE poses |

The embodiment used during recording is `g1_wbc_agile_pink` (AGILE whole-body controller + PinkIK inverse kinematics). This is a critical design choice: PinkIK converts end-effector pose targets from the XR controller into joint-space targets. The policy trains on these joint-space targets, not the EE poses themselves.

The merge script (`merge_demos.py`) validates that inputs share the same `format_version`, `action shape`, `observation keys`, and `camera geometry` before merging — an implicit schema compatibility check.

Source: [Sim Teleop](https://docs.nvidia.com/learning/physical-ai/gr00t-e2e-workflow/latest/simulation-workflow/sim-teleop-and-wbc.html), [Sim Data Export](https://docs.nvidia.com/learning/physical-ai/gr00t-e2e-workflow/latest/simulation-workflow/sim-data-export.html)

### Real Robot Path

Real-robot demonstrations are recorded as **MCAP** files via ROS 2 bag recording. MCAP is an open-source container format for multimodal robotics data. The data arrives asynchronously across multiple ROS 2 topics at different frequencies:

- Camera feeds (RealSense D435, topic: `/realsense_d435_rgb/color/image_raw/Image`)
- Joint states
- Wrist force-torque readings
- Teleoperation commands

The same XR-to-robot pipeline applies: AGILE handles lower-body balance while PinkIK converts upper-body XR controller poses to joint-space targets.

Source: [Real Teleop](https://docs.nvidia.com/learning/physical-ai/gr00t-e2e-workflow/latest/real-robot-workflow/real-teleop.html), [Real Record](https://docs.nvidia.com/learning/physical-ai/gr00t-e2e-workflow/latest/real-robot-workflow/real-record.html)

## Stage 2: Data Conversion to LeRobot Format

Both paths converge on the **LeRobot v3 dataset format** as the training input. Two different converters handle the two source formats.

### HDF5 → LeRobot (Simulation)

The converter is driven by a YAML config file:

```yaml
data_root: /datasets/isaaclab_arena/static_apple_tutorial
hdf5_name: "arena_g1_static_apple_dataset_recorded.hdf5"
language_instruction: "move the apple to the plate"
task_index: 3
state_name_sim: "robot_joint_pos"
action_name_sim: "processed_actions"
pov_cam_name_sim: "robot_head_cam_rgb"
fps: 50
chunks_size: 1000
```

This config is the **field-name mapping contract** between HDF5 data and LeRobot schema. The keys `state_name_sim`, `action_name_sim`, and `pov_cam_name_sim` map source HDF5 field paths to the LeRobot schema's `observation.state`, `action`, and `observation.images` respectively.

The output is a directory containing Parquet files (states/actions), MP4 video recordings (camera), and dataset metadata.

Source: [Sim Data Export](https://docs.nvidia.com/learning/physical-ai/gr00t-e2e-workflow/latest/simulation-workflow/sim-data-export.html)

### MCAP → LeRobot (Real Robot)

The `mcap-to-lerobot` converter from the `isaac_ros_mcap_lerobot_converter` package handles synchronization of asynchronous ROS 2 topics into the time-aligned LeRobot format.

```bash
mcap-to-lerobot \
    --rosbags-dir ${ISAAC_ROS_WS}/recordings/<session_name> \
    --output-dir ${ISAAC_ROS_WS}/recordings/lerobot_output \
    --task "move the apple to the plate" \
    --fps 30 \
    --robot-type unitree_g1
```

The converter also generates a `new_embodiment_config_defaults.py` file alongside the dataset — a Python file that registers the modality configuration for GR00T fine-tuning. This auto-generated config includes:

- Seven G1 state key groups (`left_leg` through `right_hand`)
- `RELATIVE` action representation for arms
- `ABSOLUTE` action representation for hands, waist, and optional locomotion/effort keys
- 16-step action prediction horizon (vs. 40 for sim)
- Conditional `effort_<group>` action keys (only when non-zero feed-forward torques were observed)

Source: [Real Data Export](https://docs.nvidia.com/learning/physical-ai/gr00t-e2e-workflow/latest/real-robot-workflow/real-data-export.html), [Real Fine-Tuning](https://docs.nvidia.com/learning/physical-ai/gr00t-e2e-workflow/latest/real-robot-workflow/real-fine-tuning-and-leapp.html)

### Observation: Sim vs. Real Data Model Differences

| Dimension | Simulation | Real Robot |
| --- | --- | --- |
| Source format | HDF5 | MCAP (ROS 2 bag) |
| Recording rate | 50 Hz | 30 Hz |
| Camera | Simulated `robot_head_cam_rgb` | RealSense D435 |
| Action DOFs | 43-DOF joint targets (full body) | 7 state key groups |
| Action horizon | 40 steps | 16 steps |
| Action representation | Implicit (joint-space) | Explicit (RELATIVE for arms, ABSOLUTE for hands/waist) |
| Modality config | Shared file in IsaacLab-Arena repo | Auto-generated per dataset |
| Effort keys | Not present | Conditional on recorded torques |

These differences mean the same model architecture (GR00T 1.7) is configured differently depending on data source. The modality config file is the critical artifact that bridges this gap.

## Stage 3: GR00T 1.7 Fine-Tuning

### Modality Config: The Embodiment Contract

The modality config is a Python file that calls `register_modality_config(...)` to bind an embodiment tag to a specific state/action layout. For simulation:

- Path: `isaaclab_arena_gr00t/embodiments/g1/g1_sim_wbc_data_gr00t_n_1_7_config.py`
- Content: 5 state keys, 7 action keys for the WBC layout
- Embodiment tag: `new_embodiment` → `EmbodimentTag.NEW_EMBODIMENT`

For real robot data, the converter auto-generates this file.

### Training Parameters

| Parameter | Sim Value | Real Value | Notes |
| --- | --- | --- | --- |
| Base model | `nvidia/GR00T-N1.7-3B` | Same | Hugging Face hosted |
| Batch size | 12 | 32 | GPU memory dependent |
| Training steps | 20,000 | 10,000 | |
| Action horizon | 40 | 16 | Baked into diffusion head at training time |
| Tuned modules | visual, projector, diffusion | Same | LLM frozen |

The action horizon is particularly important: it is **baked into the diffusion head at training time and cannot be changed at inference**. The modality config and server YAML must agree on this value.

### Consistency Requirements (Implicit Contracts)

The tutorial documents several parameters that **must** match between training and evaluation but are enforced only by convention:

1. `action_horizon` in modality config ↔ `action_horizon` and `action_chunk_length` in server YAML
2. `--embodiment-tag` during training ↔ `--embodiment-tag` during evaluation
3. `--modality-config-path` used for training ↔ same path for eval server
4. Recording embodiment (`g1_wbc_agile_pink`) uses PinkIK → eval embodiment (`g1_wbc_agile_joint`) uses direct joint control

Violating these produces specific error signatures:

- Wrong embodiment at eval: `ValueError: Invalid action shape, expected: 23, received: 50`
- Wrong action horizon: `Action key 'left_arm''s horizon must be 40. Got 50`

Source: [Sim Fine-Tuning](https://docs.nvidia.com/learning/physical-ai/gr00t-e2e-workflow/latest/simulation-workflow/groot-fine-tuning-sim.html), [Sim Evaluation](https://docs.nvidia.com/learning/physical-ai/gr00t-e2e-workflow/latest/simulation-workflow/sim-evaluation.html)

## Stage 4: Evaluation

### Server-Client Architecture

Evaluation uses a ZeroMQ-based server-client architecture:

- **Server** (`run_gr00t_server.py`): loads checkpoint, runs inference on GPU, serves actions via ZeroMQ on port 5555
- **Client** (`policy_runner.py`): runs Isaac Lab-Arena environment, sends observations (including camera when `--enable_cameras`), receives actions

The server and client can run in different environments (the server outside Docker in the Isaac-GR00T venv, the client inside the Arena container). The client uses `Gr00tRemoteClosedloopPolicy` as a wrapper around the ZeroMQ connection.

### Success Metrics

The task uses contact-sensor termination with `force_threshold=0.5 N` and `velocity_threshold=0.1 m/s`. Episode length is 6 seconds at 50 Hz control. The reference model achieves approximately 68% success rate over a 5×5 grid evaluation (100 rollouts).

Source: [Sim Evaluation](https://docs.nvidia.com/learning/physical-ai/gr00t-e2e-workflow/latest/simulation-workflow/sim-evaluation.html)

## Stage 5: LEAPP Export & Deployment

### LEAPP Export Artifacts

The export produces a bundle containing:

| File | Purpose |
| --- | --- |
| `<name>.yaml` | Policy metadata (pipeline specification) |
| `<name>.png` | Network graph visualization |
| `backbone.onnx` (+`.data`) | Main transformer/network body |
| `action_head.onnx` (+`.data`) | Raw action outputs from backbone features |
| `preprocess_state.onnx` | Proprioceptive input preprocessing |
| `preprocess_video.onnx` | Visual input preprocessing |
| `decode_action.onnx` | Action decoding to joint commands |
| `log.txt` | Export logs |

The `--joint_config export/data/g1_joints.json` argument is critical for G1: "Without it, export uses placeholder joint names and deployment joint mapping can fail or produce incorrect actions."

Source: [LEAPP Export](https://docs.nvidia.com/learning/physical-ai/gr00t-e2e-workflow/latest/real-robot-workflow/real-leapp-export.html)

### Deployment Architecture

Deployment on Jetson AGX Thor composes two concurrent policies via CUDA MPS GPU partitioning:

1. **AGILE WBC Policy** — lower-body balance (always running)
2. **GR00T Policy** — upper-body manipulation (enabled via `blend_ratio`)

Each has a separate safety controller node (`/safety_controller_lower_body`, `/safety_controller_upper_body`) with independent `blend_ratio` parameters (0.0 = disabled, 1.0 = fully active, smooth ramping).

The `hardware_type` launch parameter switches the backend:

| `hardware_type` | Backend | Physics | Notes |
| --- | --- | --- | --- |
| `real` | Unitree G1 hardware | Physical | Requires network setup, CUDA MPS |
| `mujoco` | MuJoCo simulation | Simulated | `blend_ratio` defaults to 1.0 |

This abstraction is achieved through **ros2_control's hardware interface system** (see [nvidia-isaac-ros-deploy-notes.md](nvidia-isaac-ros-deploy-notes.md) for architecture details):

- `mujoco_ros2_control` provides a SystemInterface plugin that presents MuJoCo as hardware to ros2_control
- The same controllers, policy nodes, and launch graph work regardless of backend
- The `hardware_type` parameter selects which SystemInterface plugin loads

Source: [Deployment](https://docs.nvidia.com/learning/physical-ai/gr00t-e2e-workflow/latest/real-robot-workflow/real-deployment.html)

## Interoperability Observations

### Format Conversion Points (Friction Sources)

Each arrow in the workflow represents a format boundary where conversion logic is needed:

| Boundary | Converter | Config Artifact | What Can Go Wrong |
| --- | --- | --- | --- |
| Sim recording → LeRobot | `convert_hdf5_to_lerobot.py` | Per-task YAML with field mappings | Wrong field names, mismatched fps |
| Real recording → LeRobot | `mcap-to-lerobot` | CLI args + auto-generated modality config | fps ↔ sync_rate mismatch, wrong robot-type |
| LeRobot → GR00T training | `launch_finetune.py` | Modality config Python file | Action horizon mismatch, wrong embodiment tag |
| Checkpoint → LEAPP bundle | `export_with_leapp.py` | `g1_joints.json` | Missing joint config → placeholder names |
| LEAPP bundle → Deployment | `ros2 launch` | LEAPP YAML + launch params | Joint mapping failure, GPU contention without MPS |
| Training → Evaluation | Server YAML + modality config | Must match training params | Action shape mismatch (23 vs. 43 vs. 50 DOF) |

### What's Explicit vs. Implicit

**Explicit (machine-readable) contracts:**

- LEAPP pipeline YAML: tensor shapes, dtypes, semantic kinds, element names, graph wiring
- LeRobot `meta/info.json`: fps, episode count, frame count
- HDF5 field paths (validated by merge script)
- Joint config JSON (`g1_joints.json`) for LEAPP export

**Implicit contracts (convention-enforced, error-prone):**

- Recording embodiment choice (`_pink` vs. `_joint`) determines action space semantics
- Action horizon must match between modality config, training, and server YAML
- Camera resolution and mounting position assumed consistent across data collection sessions
- Normalization is baked into exported ONNX models but the strategy and statistics are not declared
- The relationship between `fps` in conversion config and `sync_rate` in recorder
- PinkIK during recording produces joint targets that become the training signal — this causal chain is documented in prose but not in metadata

### The Modality Config as a De Facto Contract

The modality config Python file acts as the central contract between data collection, training, and evaluation. It defines:

- State keys (observation dimensions) and their grouping by body part
- Action keys and their representation type (RELATIVE vs. ABSOLUTE)
- Action prediction horizon
- Which body parts have effort keys

However, it's a Python file calling a registration function — not a declarative schema. This means:

- It can't be validated without executing Python
- It's not discoverable by tooling without importing the module
- Two modality configs can't be compared for compatibility without code

### The `g1_joints.json` Gap

The LEAPP export requires `--joint_config export/data/g1_joints.json` for correct joint naming. Without it, placeholder names are used and "deployment joint mapping can fail or produce incorrect actions." This file bridges between training-time joint ordering and deployment-time ROS 2 joint names — but its schema, contents, and relationship to URDF joint names are not documented in the tutorial.

## Significance for Contracts Research

This workflow validates several hypotheses from our research:

1. **Format conversions are the primary interoperability friction**: Five separate format boundaries exist in a single workflow, each with its own ad-hoc configuration.

2. **Compatibility is enforced by convention, not by contract**: The most dangerous mismatches (action horizon, embodiment type, recording-vs-eval embodiment) produce runtime errors with cryptic messages rather than being caught at pipeline configuration time.

3. **LEAPP is the strongest existing contract mechanism**: Its pipeline YAML with tensor semantics, element names, and graph topology is the closest thing to what our research envisions. But it only covers the policy→deployment boundary.

4. **The modality config pattern validates the need for embodiment contracts**: The fact that both sim and real paths require a modality config — and that the real-robot converter auto-generates one — shows that explicit embodiment metadata is necessary for training pipelines.

5. **The tutorial's own warnings are evidence of the problem**: "A weak dataset, incorrect action mapping, unstable controller, or unrepeatable workspace can break the pipeline" even when tools are correctly connected. This is the gap between format validity and functional compatibility.
