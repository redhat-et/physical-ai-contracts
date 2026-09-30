# Data Formats at Interface Boundaries — Physical AI Workflow

**Date**: 2026-09-18
**Purpose**: Document the specific data formats exchanged between components in a concrete Physical AI workflow, identifying where format negotiation or compatibility validation occurs (or is absent).

---

## Scenario

LeRobot/LIBERO dataset → pi0.5 fine-tuning → OpenPI server or vLLM-Omni serving → Franka Panda (real or Isaac Sim) → data collection (OTel/ROSbag) → dataset for retraining. Plus synthetic data generation via Isaac Sim.

---

## 1. HuggingFace Hub → Training Framework (LeRobot)

### Dataset loading

LeRobot downloads and reads files in this order:

1. **`meta/info.json`** — read first. Contains:
   - `codebase_version`: format version string
   - `robot_type`: free-form string (e.g., `"so100"`, `"aloha"`)
   - `fps`: integer
   - `features`: dict mapping feature names → `{dtype, shape, names}`. Feature keys use dot-notation:
     - `observation.state` — proprioception vector
     - `observation.images.<camera_key>` — camera observations (stored as video)
     - `action` — robot actions
   - Path templates for locating data/video shards
   - Source: [LeRobotDataset v3.0 docs](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)

2. **`meta/stats.json`** — per-feature statistics: `mean`, `std`, `min`, `max` (optionally `q01`, `q99`). Used for normalization. Exposed as `dataset.meta.stats`.

3. **`meta/tasks.jsonl`** — natural-language task descriptions mapped to integer IDs.

4. **`meta/episodes/chunk-NNN/file-NNN.parquet`** — per-episode lengths, task assignments, data/video offsets.

5. **`data/chunk-NNN/file-NNN.parquet`** — tabular data (states, actions, timestamps). Multiple episodes per file.

6. **`videos/<camera_key>/chunk-NNN/file-NNN.mp4`** — camera frames, multiple episodes per file.

### Model checkpoint loading

LeRobot reads `PI0Policy.from_pretrained(model_id)`, which downloads:

1. **`config.json`** — policy configuration:
   - `type` (e.g., `"pi0"`, `"pi05"`)
   - `n_obs_steps`, `n_action_steps`, `chunk_size`
   - `normalization_mapping`: e.g., `{"VISUAL": "IDENTITY", "STATE": "MEAN_STD", "ACTION": "MEAN_STD"}`
   - `input_features`, `output_features`
   - `max_state_dim`, `max_action_dim`
   - `resize_imgs_with_padding`, `empty_cameras`
   - Source: [Pi0.5 docs](https://huggingface.co/docs/lerobot/pi05)

2. **`model.safetensors`** — FP32 or BF16 model weights.

3. **`policy_preprocessor.json`** + **`policy_preprocessor_step_N_normalizer_processor.safetensors`** — normalization stats (mean/std or q01/q99) baked into processor pipeline.

4. **`policy_postprocessor.json`** + **`policy_postprocessor_step_0_unnormalizer_processor.safetensors`** — action unnormalization.

5. **`tokenizer/`** — PaliGemma SentencePiece tokenizer for language inputs.

6. **`train_config.json`** — training hyperparameters (lr, batch_size, etc.).

Source: [hqfang/pi05-so100_101 file tree](https://huggingface.co/hqfang/pi05-so100_101/tree/main)

### Validation at load time

- **Format version check**: `codebase_version` in `info.json` must match expected version.
- **Feature name validation**: `validate_visual_features_consistency()` checks dataset camera names against policy expectations. On mismatch, raises error suggesting `--rename_map`.
- **No action dimension check**: action shape mismatch discovered at first forward pass, not at load.
- **No normalization compatibility check**: if dataset lacks q01/q99 but policy expects QUANTILE normalization, fails at first batch (or silently degrades with wrong strategy).
- **No embodiment check**: `robot_type` is a free-form string with no validation against model expectations.

### Format negotiation

**None.** The user manually specifies `--dataset.repo_id` and `--policy.pretrained_path`. LeRobot does not verify compatibility between the two before training starts (beyond camera name checking).

---

## 2. LeRobot Training → Model Checkpoint Output

### Files produced

A checkpoint directory contains:

```
checkpoint-NNNNN/
├── config.json                                              # Policy config
├── model.safetensors                                        # Model weights
├── train_config.json                                        # Training hyperparameters
├── policy_preprocessor.json                                 # Preprocessor pipeline definition
├── policy_preprocessor_step_N_normalizer_processor.safetensors  # Normalization stats
├── policy_postprocessor.json                                # Postprocessor pipeline definition
├── policy_postprocessor_step_0_unnormalizer_processor.safetensors  # Unnormalization stats
├── tokenizer/
│   └── tokenizer.model                                      # SentencePiece model
├── code/                                                    # Source code snapshot
└── training/                                                # Training metadata
```

Source: [Checkpoint commit example](https://huggingface.co/binomialdist/lerobot-pi05-train-20260509/commit/e3c73e69339bb9e7d6be3290a3aa21981aa36383), [Issue #4647](https://github.com/huggingface/lerobot/issues/4647)

### Metadata stored in checkpoint

| What | Where | Format |
| --- | --- | --- |
| Action dimensions & chunk size | `config.json` → `n_action_steps`, `max_action_dim` | JSON int |
| Normalization strategy | `config.json` → `normalization_mapping` | JSON dict: `{"ACTION": "MEAN_STD", ...}` |
| Normalization statistics | `.safetensors` processor files | SafeTensors tensors (mean, std, q01, q99) |
| Input/output feature names | `config.json` → `input_features`, `output_features` | JSON dict |
| Camera slots & empty cameras | `config.json` → `empty_cameras` | JSON int |
| Image resize settings | `config.json` → `resize_imgs_with_padding` | JSON bool |
| Base model reference | HF model card `base_model` field | YAML frontmatter |
| Training dataset | `train_config.json` → `dataset.repo_id` | JSON string |

### What is NOT stored

- **Action space semantics**: whether actions are EE deltas, absolute joint positions, or velocities
- **Joint names or ordering**: which physical joint each action dimension controls
- **Camera roles**: which camera is wrist-mounted vs. world
- **Robot embodiment description**: joint limits, kinematic chain, DOF count
- **Compatible datasets or robots**: no cross-artifact compatibility metadata

---

## 3. Model Checkpoint → OpenPI Server

### How OpenPI loads a model

OpenPI uses a `TrainConfig` object (Python, not declarative metadata) that specifies:

- Checkpoint path (local or GCS)
- Policy config class (e.g., `pi0_fast_droid`, `pi0_fast_libero`)
- Per-robot input/output adapter classes (e.g., `LiberoInputs`/`LiberoOutputs`)
- Normalization: loads `norm_stats.json` per dataset config

Server startup:

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi0_fast_droid \
  --policy.dir=gs://openpi-assets/checkpoints/pi0_fast_droid
```

Source: [OpenPI remote inference docs](https://github.com/Physical-Intelligence/openpi/blob/main/docs/remote_inference.md)

### Serving API

**Transport**: WebSocket on port 8000.

**Client SDK**: `openpi-client` package (minimal dependencies, embeddable in robot code).

```python
from openpi_client import websocket_client_policy

client = websocket_client_policy.WebsocketClientPolicy(
    host="localhost", port=8000
)
action_chunk = client.infer(observation)
```

### Request format (observation dict)

```python
observation = {
    "observation/image": uint8_array,        # Resized to 224×224 with padding
    "observation/wrist_image": uint8_array,   # Resized to 224×224 with padding
    "observation/state": float_array,         # Proprioceptive state, sent UNNORMALIZED
    "prompt": "pick up the red block",        # Task instruction (text)
}
```

- Image preprocessing (resize + uint8 conversion) done client-side using `image_tools` utilities.
- State normalization handled server-side.
- Exact field names vary per robot config (e.g., DROID uses `observation/exterior_image_1_left`).

### Response format (actions)

```python
action_chunk = client.infer(observation)["actions"]
# shape: (action_horizon, action_dim) — e.g., (10, 7) for 10-step chunk of 7-DOF actions
# dtype: float (numpy array)
```

No metadata is returned about action semantics, units, or coordinate frame.

### Format negotiation

**None.** The server's per-robot config class (`DroidInputs`, `LiberoInputs`, etc.) defines expected observation fields. If the client sends wrong field names or shapes, the server raises a runtime error. No schema exchange or capability discovery.

Source: [OpenPI Issue #1046](https://github.com/Physical-Intelligence/openpi/issues/1046), [DeepWiki OpenPI](https://deepwiki.com/zhou-yh19/openpi)

---

## 4. Model Checkpoint → vLLM-Omni

### Supported VLA models

As of vLLM-Omni 0.28.0 (August 2026):

- **π0** (Pi-Zero) — merged in PR #4222
- **π0.5** (Pi0.5) — in progress via PR #6950
- **GR00T-N1.7** — in flight
- **InternVLA-A1**, **Alpamayo**, **AgiBot GO-1-Air**, **LingBot-VA** — in flight

Source: [vLLM-Omni GitHub](https://github.com/vllm-project/vllm-omni), [RFC #6524](https://github.com/vllm-project/vllm-omni/issues/6524)

### Serving API

**Endpoint**: `/v1/realtime/robot/openpi` — OpenPI-compatible serving layer.

Per RFC #6524:

- Inputs: multi-view camera images (1-view, 2-view, 3-view validated), proprioceptive state, language instruction
- Outputs: continuous action chunks via flow-matching denoising
- Normalization: `_build_norm_buffers` on server handles mean/std and min/max. Quantile normalization (used by pi0.5) was initially broken — returned `None`, degrading to identity. Being fixed.

### Format negotiation

**None documented.** The client must know the model's expected observation format. No capability discovery or schema exchange.

Source: [RFC #6524](https://github.com/vllm-project/vllm-omni/issues/6524), [Issue #4136](https://github.com/vllm-project/vllm-omni/issues/4136)

---

## 5. Policy Server → Robot (ROS 2)

### Observations: Robot → Policy

| Data | ROS 2 Message Type | Typical Topic | Key Fields |
| --- | --- | --- | --- |
| Joint positions/velocities | `sensor_msgs/msg/JointState` | `/joint_states` | `name[]`, `position[]`, `velocity[]`, `effort[]` |
| Camera images | `sensor_msgs/msg/Image` | `/camera/image_raw` | `header`, `height`, `width`, `encoding`, `data` |
| Compressed images | `sensor_msgs/msg/CompressedImage` | `/camera/image_raw/compressed` | `header`, `format`, `data` |
| Camera intrinsics | `sensor_msgs/msg/CameraInfo` | `/camera/camera_info` | `K[9]`, `D[]`, `R[9]`, `P[12]` |
| End-effector pose | `geometry_msgs/msg/PoseStamped` | `/ee_pose` | `header`, `pose.position`, `pose.orientation` |

### Actions: Policy → Robot

| Command type | ROS 2 Message Type | Typical Topic | Key Fields |
| --- | --- | --- | --- |
| Joint trajectory | `trajectory_msgs/msg/JointTrajectory` | `/joint_trajectory_controller/joint_trajectory` | `joint_names[]`, `points[].positions[]`, `points[].time_from_start` |
| Twist (EE velocity) | `geometry_msgs/msg/Twist` | `/cmd_vel` | `linear.{x,y,z}`, `angular.{x,y,z}` |
| Joint position command | `std_msgs/msg/Float64MultiArray` | `/forward_position_controller/commands` | `data[]` |

### Action space mapping

Configured per-robot, not standardized. Options include:

1. **Rosetta contract** (community project): A YAML file maps ROS 2 topics ↔ LeRobot features, used identically for recording, conversion, and deployment:

```yaml
robot_type: my_robot
fps: 30
observations:
  observation.state:
    channel: {topic: /joint_states, type: sensor_msgs/msg/JointState}
    align: {strategy: hold, timeline: header}
    select: [position.j1, position.j2]
  observation.images.cam:
    channel: {topic: /camera/image_raw/compressed, type: sensor_msgs/msg/CompressedImage}
    align: {strategy: hold, timeline: header}
    apply: [resize: [480, 640]]
actions:
  action:
    channel: {topic: /cmd, type: sensor_msgs/msg/JointState}
    select: [position.j1, position.j2]
```

Source: [Rosetta GitHub](https://github.com/iblnkn/rosetta)

1. **OpenPI per-robot classes**: `DroidInputs`, `AlohaInputs`, etc. — Python code, not declarative config.

2. **RDP signal routing**: YAML-defined DAG with `SignalBus` for typed signals between handler plugins.

### Format negotiation

**None at the ROS 2 level.** ROS 2 enforces message type agreement between publishers and subscribers (a publisher of `JointState` can only be read by a `JointState` subscriber). But the *semantic* mapping (which joint index = which physical joint, which camera = which model input) is configured manually per robot.

---

## 6. Robot → Data Collection

### ROSbag2

**Format**: Directory containing `metadata.yaml` + storage files (default: `.mcap`, alternative: `.db3`).

**`metadata.yaml`** structure (version 9, current):

```yaml
rosbag2_bagfile_information:
  version: 9
  storage_identifier: mcap
  relative_file_paths: [rosbag2_2026_09_18-14_30_00_0.mcap]
  duration: {nanoseconds: 30000000000}
  starting_time: {nanoseconds_since_epoch: 1726666200000000000}
  message_count: 4500
  topics_with_message_count:
    - topic_metadata:
        name: /joint_states
        type: sensor_msgs/msg/JointState
        serialization_format: cdr
        offered_qos_profiles: "..."
      message_count: 3000
    - topic_metadata:
        name: /camera/image_raw/compressed
        type: sensor_msgs/msg/CompressedImage
        serialization_format: cdr
      message_count: 900
  ros_distro: rolling
```

Source: [ros2/rosbag2 GitHub](https://github.com/ros2/rosbag2), [Rosbags docs](https://ternaris.gitlab.io/rosbags/topics/rosbag2.html)

### ROSbag → LeRobot conversion

At least 6 community tools exist, each with different approaches:

| Tool | Approach | Status |
| --- | --- | --- |
| **Rosetta** `port_bags.py` | Contract-driven YAML mapping | Active, contract-first |
| **Rebake** | Pipeline-based: ingest → TF enrich → time-sync → LeRobot transform | Active, adds derived features |
| **lerobot_ros** | Direct ROS 2 service for recording + bag conversion | Active |
| **Convert_data** | Direct SQLite reader, no ROS install needed | Active, multi-format output |
| **Forge Robotics** | Multi-format toolkit via shared Episode/Frame intermediate | Active, pip-installable |
| **leros2** | Config-driven plugin system, sample-and-hold resampling | Active, LeRobot 0.6.x |

Key challenge: bags are multi-rate (e.g., 30 Hz camera, 100 Hz joints). Every converter must choose a resampling strategy (sample-and-hold, interpolation, clock-driven).

Source: [LeRobot ROS 2 RFC #4368](https://github.com/huggingface/lerobot/issues/4368)

### OpenTelemetry for robotics

**No established conventions.** OTel's GenAI semantic conventions (spans for LLM calls, agent tool invocations) are maturing, but no robotics-specific conventions exist. Thor-testing uses OTel with `model.version` resource attributes for before/after comparison on Perses dashboards — ad hoc, not standardized.

---

## 7. Isaac Sim → Synthetic Dataset

### Data generation pipeline

1. **Isaac Lab simulation**: runs parallel environments with domain randomization (physics, materials, lighting, sensor noise).

2. **Data recording**: demonstrations recorded via teleoperation (Isaac Teleop, VR headset, or scripted policies) into **HDF5** files.

3. **HDF5 structure** (Isaac Lab):

   ```
   episode_NNN/
   ├── observations/
   │   ├── camera_obs/robot_head_cam_rgb   # Image frames
   │   └── robot_joint_pos                  # Joint position arrays
   └── processed_actions                    # Joint-space targets (e.g., 43-DOF)
   ```

4. **Replicator SDG** (perception data): separate pipeline producing images + annotations in KITTI, COCO, or custom format. Annotators: `rgb`, `bounding_box_2d_tight`, `bounding_box_3d`, `semantic_segmentation`, `instance_segmentation`, `distance_to_image_plane`.

Source: [Isaac Lab paper](https://arxiv.org/html/2511.04831v1), [Sim Data Export docs](https://docs.nvidia.com/learning/physical-ai/gr00t-e2e-workflow/latest/simulation-workflow/sim-data-export.html)

### HDF5 → LeRobot conversion

Config-driven script with a YAML mapping:

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

Output: LeRobot v3 directory (parquet + MP4 + metadata).

Source: [GR00T E2E workflow docs](https://docs.nvidia.com/learning/physical-ai/gr00t-e2e-workflow/latest/simulation-workflow/sim-data-export.html)

### LeIsaac (native Isaac Lab ↔ LeRobot integration)

As of November 2026, **LeIsaac** is the official imitation-learning simulation playground integrated into LeRobot's EnvHub. Provides `isaaclab2lerobotv3.py` for direct conversion. State machine scripted policies enable fully automated data collection without teleoperation.

Source: [LeIsaac GitHub](https://github.com/lightwheelai/leisaac), [Seeed blog](https://www.seeedstudio.com/blog/2026/07/08/seeed-rebot-arm-successfully-integrates-with-lerobot-v0-6-0-completing-the-robot-learning-loop-in-nvidia-isaac-simulation/)

### GR00T flavor

GR00T requires an additional `meta/modality.json` on top of LeRobot v2:

```json
{
  "joint_position": {"start": 0, "end": 6},
  "gripper_position": {"start": 6, "end": 7}
}
```

This splits flat concatenated arrays into semantic fields. Not part of standard LeRobot format.

Source: [Isaac-GR00T data preparation](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/data_preparation.md)

### Format negotiation

**None.** The conversion config YAML maps HDF5 field names to LeRobot feature names manually. If field names are wrong, the script fails at runtime. No schema comparison or auto-detection.

---

## 8. RHOAI Model Registry

### Data model

Based on Kubeflow Model Registry (the upstream project RHOAI uses):

```
RegisteredModel (name, owner, customProperties)
  └── ModelVersion (name, author, customProperties)
        └── ModelArtifact (name, uri, modelFormatName, modelFormatVersion,
                           storageKey, storagePath, serviceAccountName,
                           customProperties)
        └── DocArtifact (name, uri, customProperties)
```

### customProperties: typed key-value map

**Not** string→string. Values are typed:

```json
"customProperties": {
    "AWS_DEFAULT_REGION": {"string_value": "us-east-1"},
    "model_accuracy":     {"double_value": 0.95},
    "is_production":      {"bool_value": true},
    "epoch_count":        {"int_value": 100}
}
```

Types supported: `string_value`, `int_value`, `double_value`, `bool_value`.

Source: [Kubeflow Hub logical model](https://github.com/kubeflow/hub/blob/main/docs/logical_model.md)

### Model artifact storage

- `uri`: general locator (e.g., `s3://bucket/path`, `oci://registry/image:tag`)
- `modelFormatName`: string (e.g., `"onnx"`, `"safetensors"`)
- `modelFormatVersion`: string
- `storageKey`: references access credential (K8s Secret name)
- `storagePath`: path within the URI

### Can it store Physical AI metadata?

**Technically yes, practically no.** `customProperties` can hold any key-value pair, so you could store:

```json
{
    "action_dim": {"int_value": 7},
    "action_type": {"string_value": "ee_delta_7dof"},
    "normalization": {"string_value": "quantile"},
    "robot_type": {"string_value": "franka_panda"}
}
```

But there is:

- No schema enforcement (any key is accepted, no validation)
- No structured nested objects (only flat key-value)
- No querying by custom property values (cannot filter "all models with action_type=ee_delta_7dof")
- No standardized key names for Physical AI metadata

The Model Catalog Bridge (RHOAI 3.x) looks for well-known keys: `License`, `Provider`, `Registered From`, `Source model` — all generic, none Physical AI-specific.

Source: [model-catalog-bridge](https://github.com/redhat-ai-dev/model-catalog-bridge), [RHOAI Model Registry docs](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/2-latest/html/working_with_model_registries/working-with-model-registriesmodel-registry)

---

## Cross-Cutting Analysis: Format Negotiation Summary

| Interface | Sender Format | Receiver Format | Negotiation/Validation | Gap |
| --- | --- | --- | --- | --- |
| HF Hub → LeRobot (dataset) | `info.json` + parquet + MP4 | LeRobotDataset Python API | Camera name check only | No action dim, normalization, or embodiment check |
| HF Hub → LeRobot (model) | config.json + safetensors + processor files | PI0Policy.from_pretrained() | Format version check only | No dataset compatibility check |
| LeRobot → checkpoint | Python state → files | safetensors + JSON | N/A (producer) | No action semantics, joint names, or camera roles stored |
| Checkpoint → OpenPI | safetensors + per-robot Python class | WebSocket server | None — per-robot adapter hardcoded | Client must know exact observation field names |
| Checkpoint → vLLM-Omni | safetensors + model config | `/v1/realtime/robot/openpi` | None documented | Client must know observation schema |
| Policy → Robot (ROS 2) | Action array | `JointTrajectory` / `Twist` / `Float64MultiArray` | ROS 2 message type agreement only | No semantic mapping validation |
| Robot → ROSbag | ROS 2 topics (CDR) | MCAP/SQLite3 + metadata.yaml | Topic type recorded | No LeRobot compatibility metadata |
| ROSbag → LeRobot | MCAP + manual config | Parquet + MP4 + info.json | Per-tool YAML config (Rosetta, Rebake, etc.) | 6 incompatible tools, no standard |
| Isaac Sim → HDF5 | USD scene + physics | HDF5 episodes | None | No metadata about action semantics or embodiment |
| HDF5 → LeRobot | HDF5 + manual YAML config | Parquet + MP4 + info.json | Field name mapping in YAML | Wrong names fail at runtime |
| Model → RHOAI Registry | safetensors + config.json | RegisteredModel + ModelVersion + ModelArtifact | `modelFormatName` only | No Physical AI metadata schema |

### The pattern

**Every interface boundary relies on manual configuration or implicit convention. No boundary performs semantic compatibility validation.** The closest to format negotiation is:

1. **LeRobot's camera name validation** — reactive, partial, buggy
2. **ROS 2 message type matching** — structural only (type name agreement, not semantic)
3. **Rosetta's YAML contract** — single-source-of-truth for one pipeline, but project-specific

The recurring pattern at each boundary: metadata that would enable automated compatibility checking *exists* on one side but is not exposed in a form the other side can consume.

---

## Key Observations for Contract Design

1. **Rosetta's contract pattern is the closest existing precedent**: a single YAML file that maps between two worlds (ROS 2 topics ↔ LeRobot features) and is used identically for recording, conversion, and deployment. This prevents train-serve skew by design.

2. **Normalization metadata is split across artifacts**: the strategy is in the model's `config.json`, the stats are in the dataset's `stats.json`, and the processor weights are in the checkpoint's `.safetensors`. No single place declares "this model-dataset pair is normalization-compatible."

3. **Action semantics are invisible**: neither LeRobot's `info.json` nor the model's `config.json` declares whether actions are EE deltas, absolute joint positions, or velocities. This is the single largest gap — it prevents any automated compatibility check.

4. **RHOAI Model Registry can store arbitrary metadata but has no Physical AI schema**: the `customProperties` typed map could hold action space, embodiment, and normalization metadata, but without a standardized schema, no tooling can query or validate it.

5. **Camera role information exists nowhere in machine-readable form**: dataset cameras are free-form strings, model cameras are config-specific slot names, and robot cameras are ROS topic names. The three-way mapping is always manual.
