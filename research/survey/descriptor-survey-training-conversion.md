# Descriptor Survey — TrainingConfigDescriptor & DataConversionDescriptor

**Date**: 2026-09-21
**Scope**: How existing communities represent training pipeline configuration and dataset format conversion. Maps to our `TrainingConfigDescriptor` (data pipeline choices, normalization strategy, action representation) and `DataConversionDescriptor` (feature mapping, format bridging, action space unification).
**Method**: Source code analysis, cross-referencing with inference-side surveys (descriptor-survey-action.md, descriptor-survey-transform-mapping.md, descriptor-survey-checkpoint-server.md), community documentation.

---

## Training Configuration

### 1. LeRobot (Hugging Face)

**Source**: [huggingface/lerobot](https://github.com/huggingface/lerobot)

#### What they call the concept

Policy config — Python dataclasses serialized as `config.json` in checkpoint directories. Per-architecture configs: `Pi0Config`, `Pi05Config`, `DiffusionConfig`, `ACTConfig`, `TDMPCConfig`, `VQBeTConfig`. Training launched via `lerobot/scripts/train.py` with Hydra YAML overrides.

#### Explicitly declared training config fields

**Common across all policy configs:**

| Field | Type | Purpose |
| --- | --- | --- |
| `type` | string | Policy architecture identifier |
| `n_obs_steps` | int | Observation history length |
| `normalization_mapping` | dict | FeatureType → NormalizationMode (e.g., `{"VISUAL": "IDENTITY", "STATE": "MEAN_STD", "ACTION": "QUANTILE"}`) |
| `input_features` | dict | Feature name → `{shape, type}` |
| `output_features` | dict | Feature name → `{shape, type}` |

**Pi0/Pi0.5-specific:**

| Field | Type | Purpose |
| --- | --- | --- |
| `max_state_dim` | int | Max proprioception dims (zero-padded to this, default 32) |
| `max_action_dim` | int | Max action dims (zero-padded, default 32) |
| `chunk_size` | int | Maximum action chunk length |
| `n_action_steps` | int | Chunk horizon |
| `resize_imgs_with_padding` | bool | Whether images get aspect-preserving resize |
| `empty_cameras` | int | Number of unused camera slots |

**Diffusion-specific:**

| Field | Type | Purpose |
| --- | --- | --- |
| `horizon` | int | Action prediction horizon |
| `n_action_steps` | int | Steps actually executed per chunk |
| `num_inference_steps` | int | DDPM denoising steps |
| `input_shapes` | dict | Feature name → shape |
| `output_shapes` | dict | Feature name → shape |

**ACT-specific:**

| Field | Type | Purpose |
| --- | --- | --- |
| `chunk_size` | int | Action chunk length |
| `kl_weight` | float | VAE KL divergence weight |

**Training script config (Hydra YAML):**

- `dataset_repo_id` — HF dataset repo
- `training.batch_size`, `training.lr`, `training.num_epochs`
- `training.grad_accumulation_steps`, `training.optimizer`
- `wandb.enable`, `wandb.project`
- `env` — evaluation environment config (for online eval)

#### Data pipeline choices

- **Camera selection**: Determined by `input_features` keys matching `observation.images.*`. If dataset has cameras not in `input_features`, they're ignored. `validate_visual_features_consistency()` warns on mismatch and suggests `--rename_map`.
- **Normalization strategy**: Declared per-modality in `normalization_mapping`. Stats computed from dataset during training via `compute_stats.py`, stored as `stats.json` or processor safetensors.
- **Action representation**: Not declared in config — determined by dataset. No explicit field for delta vs. absolute or rotation format.
- **Data augmentation**: Image augmentation via `ImageTransforms` (random crop, color jitter, brightness/contrast). Enabled via `training.image_transforms` Hydra config.

#### What's implicit

- Which joints are used and in what order — determined by dataset, not config
- Action representation type (EE delta vs. joint position) — determined by dataset
- Sign flips, gripper conversion — not handled in generic training pipeline
- Robot-specific data transforms — via processor pipeline code, not config

**Sources**: [LeRobot train.py](https://github.com/huggingface/lerobot/blob/main/lerobot/scripts/train.py), [LeRobot Pi0Config](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/pi0/configuration_pi0.py)

---

### 2. OpenPI (Physical Intelligence)

**Source**: [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)

#### What they call the concept

`TrainConfig` — Python dataclass in `src/openpi/training/config.py`. Not a standalone file; defined programmatically with named presets.

#### Explicitly declared fields

| Field | Type | Purpose |
| --- | --- | --- |
| `model` | `BaseModelConfig` | Architecture: `action_dim`, `action_horizon`, `max_token_len` |
| `data` | `DataConfig` | Dataset + normalization config |
| `optimizer` | `OptimizerConfig` | LR, warmup, weight decay |
| `training` | various | Batch size, steps, grad accumulation |

**`DataConfig` fields:**

| Field | Type | Purpose |
| --- | --- | --- |
| `repo_id` | string | HF dataset repo |
| `use_quantile_norm` | bool | Whether to use quantile (True) or z-score (False) normalization |
| `norm_stats` | dict | Pre-computed normalization statistics |
| `repack_transforms` | list | Key remapping transforms |
| `data_transforms` | list | Robot-specific transforms (pre-normalization) |
| `model_transforms` | list | Model-specific transforms (post-normalization) |

**Named configs (presets):**

Each preset bundles model config, data config, and robot-specific transforms:

- `pi0_aloha`: action_dim=32, action_horizon=50, AlohaInputs/Outputs
- `pi0_droid`: action_dim=32, action_horizon=10, DroidInputs/Outputs
- `pi0_fast_droid`: action_dim=8, action_horizon=10, FAST model variant
- `pi05_droid`: action_dim=32, action_horizon=15

#### Data pipeline composition

Training pipeline is composed from four ordered transform stages:

1. `repack_transforms` — dataset key remapping to model-canonical names
2. `data_transforms` — robot-specific (sign flips, gripper conversion, camera remapping)
3. `Normalize` — using `norm_stats` + `use_quantile_norm`
4. `model_transforms` — model-specific (tokenization, image resize, padding)

This is the most explicit pipeline composition model — each stage is a named list of transforms. But the transforms themselves are Python code, not declarative configs.

#### What survives in the checkpoint

After training, the checkpoint directory contains:

- `config.json` — model architecture + input/output features + normalization mapping
- `norm_stats.json` — per-feature statistics
- But NOT: which robot transforms were used, sign flip masks, gripper constants, camera mappings

The training config fully specifies the pipeline, but the checkpoint loses the robot-specific transform configuration. This is why adapters must be independently reimplemented in each inference server.

**Sources**: [OpenPI config.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py), [OpenPI training docs](https://github.com/Physical-Intelligence/openpi/blob/main/docs/training.md)

---

### 3. GR00T (NVIDIA)

**Source**: [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T)

#### What they call the concept

Training configuration is distributed across multiple JSON files and Python configs. The key concept is **embodiment-specific configuration** — each robot gets its own modality config, statistics, and action representation.

#### Explicitly declared fields

**`modality.json`** (per-dataset):

```json
{
  "state": {
    "single_arm": {"start": 0, "end": 5},
    "gripper": {"start": 5, "end": 6}
  },
  "action": {
    "single_arm": {"start": 0, "end": 5},
    "gripper": {"start": 5, "end": 6}
  },
  "video": {
    "front": {"original_key": "observation.images.front"}
  }
}
```

Declares semantic array slicing (which indices are arm vs. gripper), video key remapping, and annotation key mapping.

**`EmbodimentTag` enum**: Routes to action-space projector heads. Tag naming convention encodes action representation (e.g., `oxe_droid_relative_eef_relative_joint` → relative EEF + relative joint actions).

**Per-embodiment normalization:**

- `statistics.json` — per-joint-group min/max stats
- `relative_stats.json` — per-horizon-timestep stats for relative actions (2D: timestep × dim)
- Strategy: min/max with optional percentile substitution (q01/q99)
- `clip_outliers: bool` — clamp output to [-1, 1]
- `use_percentiles: bool` — use q01/q99 instead of min/max

**Action representation config:**

- `ActionRepresentation` enum: `RELATIVE`, `ABSOLUTE`
- `ActionType` enum: `EEF`, `NON_EEF`
- `ActionFormat` enum: various rotation formats
- Configured per joint group via `ActionConfig`

**Training-specific:**

- `state_dropout_prob` — dropout for state during training (robustness)
- `image_crop_size`, `image_target_size` — image augmentation parameters
- Random crop position sampled per sample, shared across camera views/timesteps

#### What's implicit

- Joint ordering — determined by dataset creation (modality.json index ranges are the declaration)
- Camera ordering — via `video_modality_keys` with alias support
- Which embodiment tags are compatible — encoded in model weights, not metadata

**Sources**: [GR00T modality.json](https://github.com/NVIDIA/Isaac-GR00T/blob/main/examples/SO100/modality.json), [GR00T data preparation](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/data_preparation.md)

---

### 4. Isaac Lab (NVIDIA)

**Source**: [isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab)

#### What they call the concept

RL training configuration via Python config classes inheriting from `ManagerBasedRLEnvCfg`. Combines environment, task, and training agent configuration.

#### Explicitly declared fields

**Environment config (`ManagerBasedRLEnvCfg`):**

| Section | Key fields | Purpose |
| --- | --- | --- |
| `scene` | `ArticulationCfg`, `SensorCfg` | Robot + sensor definitions |
| `actions` | `ActionTermCfg` subclasses | Action space: `JointPositionActionCfg`, `JointVelocityActionCfg`, `DifferentialInverseKinematicsActionCfg` |
| `observations` | `ObservationGroupCfg` → `ObservationTermCfg` | Observation composition: joint positions, velocities, images, etc. |
| `rewards` | `RewardTermCfg` | Reward function composition |
| `terminations` | `TerminationTermCfg` | Episode termination conditions |

**Action space configuration:**

- `JointPositionActionCfg`: asset_name, joint_names (regex), scale, offset
- `JointVelocityActionCfg`: asset_name, joint_names (regex), scale
- `DifferentialInverseKinematicsActionCfg`: body_name, IK parameters

**Actuator configuration (`ActuatorCfg`):**

- `actuator_model` — implicit, DC motor, ideal PD, MLP
- `stiffness`, `damping` — PD gains
- `effort_limit`, `velocity_limit`

**Training agent config (e.g., RSL-RL `PPOCfg`):**

- `num_learning_epochs`, `num_mini_batches`
- `gamma`, `lam`, `clip_param` — PPO hyperparameters
- `normalize_obs: bool` — whether to normalize observations (RSL-RL uses `EmpiricalNormalization`)

#### What's implicit

- Joint ordering — determined by USD articulation solver ordering (differs from URDF; see issue #7750)
- Action-to-joint mapping — via regex joint_names patterns, resolved at runtime
- Observation normalization statistics — computed online by RSL-RL, not stored declaratively
- Camera configuration for visual RL — set in scene config but without ML-facing metadata

**Sources**: [Isaac Lab env design](https://isaac-sim.github.io/IsaacLab/main/source/how-to/make_fixed_prim_env.html), [Isaac Lab actuators](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.actuators.html)

---

### 5. Cosmos (NVIDIA)

**Source**: [NVIDIA/Cosmos](https://github.com/NVIDIA/Cosmos)

#### What they call the concept

Video generation training config — `DataConfig`, `ModelConfig` in training scripts. Cosmos focuses on world model training, not action prediction.

#### Relevant fields for Physical AI

| Field | Purpose |
| --- | --- |
| `dataset.video_resolution` | Target video resolution |
| `dataset.fps` | Target frame rate |
| `dataset.action_fps` | Action annotation rate (for action-conditioned models) |
| `model.num_frames` | Number of frames per sample |
| `model.latent_shape` | Tokenizer output shape |

Cosmos tokenizer converts video to discrete tokens. The training config is specific to video generation and does not include action semantics, joint descriptions, or robot-specific configuration.

**Assessment**: Cosmos is tangential to our descriptor model — its training config covers video generation, not robot policy training. The `action_fps` field is the only relevant property (maps to `control_frequency_hz` in ActionChunkDescriptor).

---

## Dataset Format Conversion

### 1. LeRobot — Format Conversion

**Source**: [huggingface/lerobot](https://github.com/huggingface/lerobot)

#### Conversion utilities

**`convert_dataset_v1_to_v2`** (now v3): Migrates dataset format versions. Handles:

- Parquet schema changes
- Video re-encoding
- `info.json` schema updates
- Stats recomputation

**`push_dataset_to_hub`**: Converts from various source formats to LeRobot format:

- Source formats: `raw`, `aloha`, `pusht`, `xarm`, `umi`, `dora`
- Per-format conversion scripts in `lerobot/common/datasets/push_dataset_to_hub/`

#### Feature mapping during conversion

Each source-format converter hardcodes:

- **Camera key mapping**: e.g., Aloha `cam_high` → `observation.images.top`
- **State key mapping**: e.g., robot joint positions → `observation.state`
- **Action key mapping**: e.g., raw actions → `action`
- **Dimension ordering**: which joints in what order
- **Video encoding parameters**: codec, fps, pixel format

Example from `aloha_hdf5_format.py`:

```python
# Camera key mapping (hardcoded)
CAMERA_NAMES = {
    "cam_high": "observation.images.top",
    "cam_low": "observation.images.bottom",
    "cam_left_wrist": "observation.images.left_wrist",
    "cam_right_wrist": "observation.images.right_wrist",
}
```

#### What's declarative vs. hardcoded

| Aspect | Status |
| --- | --- |
| Source format detection | Per-format converter scripts |
| Camera key mapping | Hardcoded per source format |
| Action dim selection | Hardcoded per source format |
| Video encoding | Configurable (fps, codec) |
| Stats computation | Automatic (computed from converted data) |
| Feature naming convention | Enforced (`observation.images.*`, `observation.state`, `action`) |

**Sources**: [LeRobot push_dataset_to_hub](https://github.com/huggingface/lerobot/tree/main/lerobot/common/datasets/push_dataset_to_hub)

---

### 2. OXE (Open X-Embodiment)

**Source**: [Google DeepMind RT-X](https://robotics-transformer-x.github.io/), [tensorflow_datasets](https://github.com/tensorflow/datasets)

#### Cross-embodiment action space unification

OXE standardized on **7-DOF end-effector delta actions**: `(dx, dy, dz, droll, dpitch, dyaw, gripper)` for the RT-X model family. Each robot's native action space is mapped to this common space.

**`OXE_DATASET_TRANSFORMS`**: Per-dataset Python transform classes that convert native action/observation formats to OXE standard:

- Action relabeling (7-DOF EE delta extraction from native spaces)
- Camera key remapping (dataset-specific camera names → standard keys)
- Observation filtering (which modalities to include)

#### RLDS format structure

All OXE data uses RLDS (Reinforcement Learning Datasets):

```python
# RLDS episode structure
{
  "steps": {
    "observation": {
      "image": tf.Tensor(shape=[H, W, C]),
      "wrist_image": tf.Tensor(shape=[H, W, C]),
      "state": tf.Tensor(shape=[D]),
    },
    "action": tf.Tensor(shape=[7]),
    "reward": tf.Tensor(shape=[]),
    "is_terminal": tf.bool,
    "language_instruction": tf.string,
  }
}
```

#### Per-dataset conversion mapping

Each of the 70+ datasets requires its own transform:

| Concept | How handled |
| --- | --- |
| Action space conversion | Per-dataset Python function: native actions → 7-DOF EE delta |
| Camera key mapping | Per-dataset: native camera names → `image`, `wrist_image` |
| State extraction | Per-dataset: which state fields to include |
| Language instruction | Per-dataset: which field contains the task description |
| Action space validation | **Not automated** — each dataset's transform is manually verified |

**Assessment**: OXE demonstrates the scale of the conversion problem (70+ datasets, each with custom transforms), but provides no declarative format for these transforms. The 7-DOF EE delta convention is a useful standardization for that model family but not universally applicable (humanoids need 30+ DOF joint-space actions).

**Sources**: [OXE paper](https://arxiv.org/abs/2310.08864), [OXE dataset list](https://docs.google.com/spreadsheets/d/1rPBD77tk60AEIGZrGSODwyyzs5FgCU9Uz3h-3_t2A9g)

---

### 3. RLDS → LeRobot Conversion

**Source**: [huggingface/lerobot](https://github.com/huggingface/lerobot)

#### Conversion pipeline

LeRobot provides a generic RLDS-to-LeRobot converter:

**What's automated:**

- Episode iteration and frame extraction
- Parquet file creation with LeRobot schema
- Video encoding from image tensors
- Stats computation
- `info.json` generation

**What requires manual specification:**

- Camera key mapping — which RLDS keys map to which `observation.images.*` keys
- State key mapping — which RLDS state fields to include
- Action key mapping — RLDS action tensor to `action` feature
- FPS — recording frame rate (not always in RLDS metadata)

#### Key observation

The RLDS-to-LeRobot converter is essentially a hardcoded `DataConversionDescriptor` per source dataset — the mappings exist in Python code but not in a portable declarative format. Each new RLDS dataset requires writing a new conversion script.

---

### 4. Rosetta

**Source**: [iblnkn/rosetta](https://github.com/iblnkn/rosetta), [ros-physical-ai/demos](https://github.com/ros-physical-ai/demos)

#### YAML contract for recording and conversion

Rosetta's key design principle: **same YAML for record, convert, and deploy**. The contract bridges ROS 2 topics and LeRobot features.

**Recording use case:**

```yaml
robot_type: my_robot
robot_interface: ros2
fps: 30
observations:
  observation.state:
    channel: {topic: /joint_states, type: sensor_msgs/msg/JointState}
    align: {strategy: hold, timeline: header}
    select: [position.j1, position.j2, position.j3]
  observation.images.cam:
    channel: {topic: /camera/image_raw/compressed, type: sensor_msgs/msg/CompressedImage}
    align: {strategy: hold, timeline: header}
    apply: [resize: [480, 640]]
actions:
  action:
    channel: {topic: /cmd, type: sensor_msgs/msg/JointState}
    select: [position.j1, position.j2, position.j3]
```

**What the YAML covers for conversion:**

| Concept | How handled |
| --- | --- |
| Topic-to-feature mapping | `channel.topic` → feature key name |
| Joint selection + ordering | `select: [position.j1, ...]` — dot-path into ROS message fields |
| Camera-to-feature mapping | Feature key bound to camera topic |
| Temporal alignment | `align.strategy` (hold/interpolate) + `align.timeline` (header/receipt) |
| Image transforms | `apply: [resize: [H, W]]` — limited to resize |
| Frame rate | `fps: 30` — recording target rate |

**What the YAML does NOT cover:**

- Normalization (no stats, no strategy)
- Sign flips, gripper conversion
- Action space conversion (delta↔absolute)
- Cross-format type conversion (RLDS↔LeRobot)
- Action semantics (dimensions are selected but not annotated)

**Assessment**: Rosetta is the closest to a DataConversionDescriptor for the ROS 2 → LeRobot boundary. Its single-source-of-truth design (same YAML for record, convert, deploy) is a strong design principle worth adopting. But it covers only transport-level mapping, not semantic transforms.

**Sources**: [Rosetta GitHub](https://github.com/iblnkn/rosetta)

---

### 5. fog_rtx

**Source**: [KeplerC/fog_rtx](https://github.com/KeplerC/fog_rtx) (now part of the fog robotics ecosystem)

#### What it provides

fog_rtx is a dataset management library for robotics, providing:

- **Unified dataset abstraction**: read/write datasets across formats (RLDS, HDF5, LeRobot)
- **Cloud-native storage**: datasets stored in CloudSQL or local Parquet
- **Format bridging**: convert between RLDS, HDF5, and fog_rtx native format

#### Conversion capabilities

| Concept | How handled |
| --- | --- |
| Format detection | Automatic based on file structure |
| Feature mapping | Manual — user specifies which keys map where |
| Episode structure | Preserves RLDS episode boundaries |
| Video encoding | Re-encodes to target format's video convention |
| Metadata | Copies available metadata, generates missing fields |

**Assessment**: fog_rtx provides programmatic format bridging but no declarative conversion descriptor. Conversion logic is API calls, not config files. The library is more focused on data management (storage, versioning, access) than semantic conversion.

**Sources**: [fog_rtx GitHub](https://github.com/KeplerC/fog_rtx), [fog_rtx paper](https://arxiv.org/abs/2401.05931)

---

## Cross-Community Comparison

### Training config — property coverage

| Property | LeRobot | OpenPI | GR00T | Isaac Lab | Cosmos |
| --- | --- | --- | --- | --- | --- |
| Architecture/model type | ✓ `type` | ✓ `BaseModelConfig` | ✓ embodiment tag | ✓ env cfg class | ✓ model config |
| Input features (shapes) | ✓ `input_features` | ✓ `DataConfig` | ✓ `modality.json` | ✓ `ObservationCfg` | ✗ |
| Output features (shapes) | ✓ `output_features` | ✓ `BaseModelConfig` | ✓ `modality.json` | ✓ `ActionsCfg` | ✗ |
| Normalization strategy | ✓ `normalization_mapping` | ✓ `use_quantile_norm` | ✓ per-group config | ✓ `normalize_obs` | ✗ |
| Normalization stats | ✓ `stats.json` + processor | ✓ `norm_stats.json` | ✓ `statistics.json` | ✗ (online) | ✗ |
| Action representation | ✗ (from dataset) | ✗ (from transforms) | ✓ `ActionRepresentation` | ✓ `ActionTermCfg` | ✗ |
| Action chunk config | ✓ `chunk_size`, `n_action_steps` | ✓ `action_horizon` | ✓ embodiment tag | ✗ (step-based) | ✗ |
| Camera selection | ✓ `input_features` keys | ✓ `repack_transforms` | ✓ `video` in modality.json | ✓ `SensorCfg` | ✗ |
| Data augmentation | ✓ `image_transforms` | ✓ augmax config | ✓ crop/resize config | ✓ domain randomization | ✗ |
| Dataset reference | ✓ `dataset_repo_id` | ✓ `DataConfig.repo_id` | ✓ dataset path | ✗ (env is dataset) | ✓ |
| Robot/embodiment ref | ✗ | ✗ (adapter class name) | ✓ `EmbodimentTag` | ✓ `ArticulationCfg` | ✗ |
| Transform pipeline order | ✓ processor pipeline | ✓ 4-stage pipeline | ✓ processor pipeline | ✗ (implicit) | ✗ |
| Base model / fine-tune from | ✓ via HF model card | ✓ `TrainConfig` preset | ✓ checkpoint reference | ✗ | ✓ |

### Data conversion — property coverage

| Property | LeRobot converters | OXE transforms | Rosetta YAML | fog_rtx |
| --- | --- | --- | --- | --- |
| Source format declaration | ✓ per-format scripts | ✓ RLDS assumed | ✓ `robot_interface: ros2` | ✓ auto-detect |
| Target format declaration | ✗ (always LeRobot) | ✗ (always RLDS) | ✗ (always LeRobot) | ✓ configurable |
| Camera key mapping | Hardcoded per source | Hardcoded per dataset | ✓ declarative (YAML) | Manual (API) |
| Action dim mapping | Hardcoded per source | Hardcoded per dataset | ✓ declarative (`select`) | Manual (API) |
| Joint ordering | Hardcoded per source | Hardcoded per dataset | ✓ declarative (`select` order) | Manual (API) |
| Sign flips | ✗ | ✗ | ✗ | ✗ |
| Gripper conversion | ✗ | ✗ | ✗ | ✗ |
| Image transforms | ✗ (post-conversion) | ✗ | ✓ `apply: [resize]` | ✗ |
| Temporal alignment | ✗ | ✗ | ✓ `align.strategy` | ✗ |
| Frame rate | ✓ configurable | ✗ (from source) | ✓ `fps` | ✗ |
| Stats computation | ✓ automatic | ✗ | ✗ | ✗ |
| Single-source-of-truth | ✗ | ✗ | ✓ (record + convert + deploy) | ✗ |

---

## Synthesis

### Recommended properties for TrainingConfigDescriptor

Based on what actually exists across communities:

```
TrainingConfigDescriptor
├── architecture                             — model family + variant
├── input_dataset
│   ├── repo_id                              — dataset reference (HF repo or path)
│   └── dataset: DatasetDescriptor           — expected dataset contract
├── input_features                           — dict of feature name → {shape, type, modality}
├── output_features                          — dict of feature name → {shape, type}
├── normalization
│   ├── strategy_mapping                     — per-modality strategy (IDENTITY, MEAN_STD, QUANTILE, ...)
│   └── stats_reference                      — path to stats file
├── action_config
│   ├── representation                       — joint_position | ee_delta | ee_absolute | relative_joint
│   ├── chunk_size                           — action chunk horizon
│   ├── max_action_dim                       — padded dimension
│   └── groups[]                             — per-group config (arm, gripper) with index ranges
├── observation_config
│   ├── camera_keys[]                        — which cameras from dataset
│   ├── image_transforms[]                   — resize, crop, augmentation
│   ├── n_obs_steps                          — observation history length
│   └── state_keys[]                         — which proprioception features
├── data_transforms[]: TransformDescriptor   — augmentation + normalization pipeline
├── base_model                               — checkpoint to fine-tune from
├── target_embodiment                        — intended deployment robot (EmbodimentTag or descriptor)
└── produces_checkpoint: CheckpointDescriptor — output checkpoint contract
```

**Best sources per property:**

- Architecture + features: LeRobot policy configs (most structured, serialized as JSON)
- Normalization strategy: LeRobot `normalization_mapping` (explicit enum per modality)
- Action representation: GR00T (per-group `ActionRepresentation` + `ActionType`)
- Transform pipeline: OpenPI (explicit 4-stage pipeline composition)
- Embodiment reference: GR00T `EmbodimentTag` (only community linking training config to robot)

### Recommended properties for DataConversionDescriptor

```
DataConversionDescriptor
├── source
│   ├── format                               — rlds | hdf5 | rosbag | lerobot_v2 | raw_files
│   └── dataset: DatasetDescriptor           — source dataset contract (if available)
├── target
│   ├── format                               — lerobot_v3 | rlds | hdf5
│   └── feature_naming                       — target naming convention
├── feature_mapping                          — source key → target key (cameras, state, action)
│   ├── camera_mapping                       — source camera → target camera + role
│   ├── state_mapping                        — source state fields → target (with ordering)
│   └── action_mapping                       — source action → target action (with ordering)
├── joint_transforms
│   ├── selection                            — which joints/dims to include
│   ├── ordering                             — index permutation
│   ├── sign_flips                           — per-dimension sign mask
│   └── gripper_conversion                   — convention mapping
├── temporal_config
│   ├── source_fps                           — original frame rate
│   ├── target_fps                           — target frame rate (with subsampling)
│   └── alignment_strategy                   — hold | interpolate
├── image_transforms[]                       — resize, crop, format conversion
├── stats_computation                        — whether to compute stats during conversion
└── provenance_chain                         — ordered list of prior conversions
```

**Best sources per property:**

- Feature mapping: Rosetta YAML (only declarative format — topic→feature binding with select)
- Camera mapping: Rosetta + GR00T modality.json
- Joint ordering: Rosetta `select` + GR00T index ranges
- Temporal alignment: Rosetta `align` config
- Image transforms: Rosetta `apply` (limited but declarative)
- Stats computation: LeRobot (automatic during conversion)
- Single-source-of-truth: Rosetta design principle (same YAML for record + convert + deploy)

### What's missing from ALL communities

1. **Declarative cross-format conversion** — every conversion is hardcoded Python per source/target pair. OXE has 70+ per-dataset transform scripts; LeRobot has per-format converter scripts. No community has a portable conversion config.

2. **Action space unification metadata** — OXE standardized on 7-DOF EE delta but the mapping from each robot's native space is hardcoded per dataset. No declarative format describes how to map native actions to a target representation.

3. **Training-to-checkpoint lineage** — what robot-specific transforms were used during training (sign flips, gripper conversion, camera mapping) is lost when the checkpoint is saved. The checkpoint stores normalization stats but not the adapter logic.

4. **Dataset compatibility declaration** — no community declares which datasets are compatible with which training configs or target robots. Compatibility is discovered at runtime (shape mismatch errors).

5. **Conversion validation** — no community validates that a conversion preserves semantic correctness (e.g., that joint ordering is preserved, that sign conventions match). Validation is manual.

6. **Provenance chain** — when a dataset has been converted through multiple formats (RLDS → HDF5 → LeRobot v2 → LeRobot v3), the conversion history and applied transforms are not tracked.

### Key findings

1. **Training configs are richer than checkpoint metadata** — the training config fully specifies the data pipeline (transforms, normalization, camera selection, action representation), but most of this information is lost when the checkpoint is saved. This is a major source of interoperability problems: the inference server must independently reconstruct what the training pipeline did.

2. **GR00T has the most explicit training-to-deployment link** — `EmbodimentTag` connects training config, checkpoint, and deployment adapter. No other community has this cross-artifact linking.

3. **Rosetta's single-source-of-truth principle is the right design pattern** — using the same descriptor for recording, conversion, and deployment prevents train-serve skew. But Rosetta only covers transport-level mapping, not semantic transforms.

4. **The 70+ dataset problem demonstrates the need for declarative conversion** — OXE's per-dataset Python transforms show that the conversion problem exists at scale. A DataConversionDescriptor that declaratively describes these mappings would eliminate redundant reimplementation.

5. **LeRobot is the de facto format target** — most conversion flows end at LeRobot format. This makes LeRobot's `info.json` schema the natural starting point for DatasetDescriptor.
