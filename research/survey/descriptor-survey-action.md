# Descriptor Survey: Action Semantics

**Scope**: How different Physical AI communities represent per-dimension action semantics (what we call "ActionDescriptor") and temporal action structure (what we call "ActionChunkDescriptor").

**ActionDescriptor** — describes a single action vector's semantics: what each dimension means (joint name, domain, unit, frame, range, representation type, normalization state).

**ActionChunkDescriptor** — describes a temporal sequence of actions: horizon (number of steps), control frequency, data type.

## 1. OpenPI (Physical Intelligence)

**Source**: [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)

### What they call it

No explicit descriptor concept. Action semantics are encoded implicitly in per-robot policy transform classes and model config dataclasses.

### ActionDescriptor equivalent

Per-dimension semantics are **hardcoded in Python transform classes**, one per robot platform:

- `DroidInputs` / `DroidOutputs` (`src/openpi/policies/droid_policy.py`): State is 7 joint positions + 1 gripper position = 8 dims. `DroidOutputs` slices actions to `[:8]`. No joint names, units, or ranges declared. Action space selectable via `DroidActionSpace` enum: `JOINT_POSITION` (default) or `JOINT_VELOCITY`. Actions concatenated as `[joint_component, gripper_position]` along last axis.
- `AlohaInputs` / `AlohaOutputs` (`src/openpi/policies/aloha_policy.py`): 14-dim actions (6 left joints + 1 left gripper + 6 right joints + 1 right gripper). Dimension layout: `"left_arm_joint_angles, left_arm_gripper, right_arm_joint_angles, right_arm_gripper"` with `"dim sizes: [6, 1, 6, 1]"`. Hardcoded `_joint_flip_mask` `[1,-1,-1,1,1,1,1,1,-1,-1,1,1,1,1]` negates joints at indices 1, 2, 8, 9 for Aloha-to-pi conversion. Gripper normalization constants hardcoded (`min_val=0.01844`, `max_val=0.05800` for linear puppet range; `min_val=0.5476`, `max_val=1.6296` for angular range). Camera mapping also hardcoded: `cam_high` -> `base_0_rgb`, `cam_left_wrist` -> `left_wrist_0_rgb`, `cam_right_wrist` -> `right_wrist_0_rgb`.
- `LiberoInputs` / `LiberoOutputs` (`src/openpi/policies/libero_policy.py`): 7-dim actions. `LiberoOutputs` slices to `[:7]`. State is 8-dimensional.

Delta/absolute conversion handled by `DeltaActions`/`AbsoluteActions` transforms with boolean masks via `make_bool_mask()`. Example for Aloha: `make_bool_mask(6, -1, 6, -1)` — joints use deltas, grippers stay absolute. For DROID: `make_bool_mask(7, -1)` — 7 joints delta, gripper absolute. For LIBERO: `make_bool_mask(6, -1)` when `extra_delta_transform=True`.

Additional transform classes: `SubsampleActions(stride)` for temporal subsampling, `PadStatesAndActions(model_action_dim)` for zero-padding to model width, `ExtractFASTActions(tokenizer, action_horizon, action_dim)` for Pi0-FAST token-to-action conversion.

### ActionChunkDescriptor equivalent

Declared in model config dataclasses (`src/openpi/models/model.py`):

```python
@dataclasses.dataclass(frozen=True)
class BaseModelConfig:
    action_dim: int        # "Action space dimension"
    action_horizon: int    # "Action sequence length"
    max_token_len: int     # "Tokenized prompt maximum length"
```

Typical values from named configs in `src/openpi/training/config.py`:

| Config | action_dim | action_horizon | Notes |
| -------- | ----------- | --------------- | ------- |
| `pi0_aloha` | 32 (default) | 50 (default) | Trossen assets |
| `pi0_droid` | 32 (default) | 10 | DROID assets |
| `pi0_fast_droid` | 8 | 10 | FAST model |
| `pi05_droid` | 32 (default) | 15 | Pi0.5 model |
| `pi0_fast_libero` | 7 | 10 | `max_token_len=180` |
| `pi05_full_droid_finetune` | 32 | 16 | Pi0.5 |
| `pi05_droid_finetune` | 32 | 16 | LeRobot format DROID |
| `pi05_libero` | 32 (default) | 10 | `discrete_state_input=False` |

Note: Pi0 models pad all actions to 32 dimensions regardless of the robot's actual DOF. Pi0-FAST uses the robot's native dimension.

**No explicit control frequency field.** Frequency is determined by the dataset's recording rate, not declared in config.

### Normalization

Handled by `NormStats` (Pydantic dataclass in `src/openpi/shared/normalize.py`):

- Fields: `mean` (NDArray), `std` (NDArray), `q01` (NDArray, optional), `q99` (NDArray, optional)
- Stored per-feature as `norm_stats.json` — a dict mapping feature names to per-dimension stats
- Quantile normalization (q01/q99) used for Pi0-FAST and Pi0.5; z-score (mean/std) for Pi0
- `DataConfig.use_quantile_norm` boolean selects strategy; defaults based on model type (`model_type != PI0` -> quantile)

Pipeline ordering (input): `repack_transforms` -> `InjectDefaultPrompt` -> `data_transforms` -> `Normalize` -> `model_transforms`. Output: reverse order with `Unnormalize`.

### Key gaps relative to ActionDescriptor

- No joint names, units, frames, or representation type declared — all hardcoded in Python
- No per-dimension range bounds (min/max) beyond normalization stats
- No control frequency
- Dimension semantics not portable across platforms (each robot needs its own transform class)

## 2. LEAPP (NVIDIA)

**Source**: [nvidia-isaac/leapp](https://github.com/nvidia-isaac/leapp), [docs](https://nvidia-isaac.github.io/leapp/)

LEAPP (Lightweight Export Annotations for Policy Pipelines) exports full policy pipelines (preprocessing + inference + postprocessing) as deployable artifacts. It is the only community with explicit, declarative action semantics metadata.

### What they call it

`TensorSemantics` — a per-tensor annotation applied to pipeline inputs and outputs.

### ActionDescriptor equivalent

`TensorSemantics` fields:

- **`name`** (string) — unique identifier within a node's I/O (e.g., `"joint_pos"`, `"command"`)
- **`kind`** (enum) — what the tensor represents. `OutputKindEnum` values for actions include:
  - `JOINT_POSITION` → `target/joint/position`
  - `JOINT_VELOCITY` → `target/joint/velocity`
  - `JOINT_TORQUES` → `target/joint/torques`
  - `JOINT_EFFORT` → `target/joint/effort`
  - `BODY_POSITION` → `target/body/position`
  - `BODY_LINEAR_VELOCITY` → `target/body/linear_velocity`
  - `BODY_ORIENTATION` → `target/body/orientation`
  - `BODY_LINEAR_ACCELERATION` → `target/body/linear_acceleration`
  - `BODY_ANGULAR_ACCELERATION` → `target/body/angular_acceleration`
  - `KP`, `KD` (controller gains)
  - Custom strings also accepted
- **`element_names`** (list) — per-dimension joint names (e.g., `["hip_l", "knee_l", "ankle_l", ...]`); marked `# deprecated` in source but still the active mechanism for per-dimension labeling
- **`extra`** (dict) — arbitrary metadata, flattened into YAML (e.g., `{"frame": "base", "units": "rad"}`)

`InputKindEnum` values for observations include `JOINT_POSITION`, `JOINT_VELOCITY`, `JOINT_EFFORT`, `BODY_POSE`, `BODY_POSITION`, `BODY_VEL`, `WRENCH`, `COMMAND_JOINT_POSITION`, etc.

Example:

```python
TensorSemantics("command", command,
                kind=OutputKindEnum.JOINT_TORQUES,
                element_names=["hip_l", "knee_l", "ankle_l", ...],
                extra={"frame": "base", "units": "Nm"})
```

### ActionChunkDescriptor equivalent

`TemporalAxis` — placed inside `element_names` to mark one tensor axis as temporal:

- **`period_ms`** (float) — time between consecutive samples (e.g., `100` = 10 Hz)
- Horizon length is implicit from tensor shape along the temporal axis
- Serialized as `__temporal_axis__` sentinel in element_names + `temporal_period_ms` field in YAML
- At most one temporal axis per tensor
- LEAPP does not validate `temporal_period_ms` against `GraphConfigs.frequency`

Graph-level metadata via `GraphConfigs`:

- **`frequency`** (`float | None`) — graph-level execution frequency in Hz
- **`extra`** (`Dict[str, Any] | None`) — arbitrary graph-level metadata, flattened into YAML

### Key gaps relative to ActionDescriptor

- No per-dimension min/max range bounds
- No normalization metadata (normalization is baked into the exported pipeline)
- `extra` dict is unstructured — frame and units are convention, not schema-enforced
- Temporal axis is optional annotation, not required

### Related: Cosmos 3 Action API (NVIDIA)

The Cosmos 3 world-generation platform ([NVIDIA/cosmos](https://github.com/NVIDIA/cosmos)) exposes action parameters at the domain level rather than per-dimension. Its action API accepts:

- **`domain_name`** (enum: `av`, `umi`, `bridge_orig_lerobot`, `droid_lerobot`) — selects the embodiment domain
- **`raw_action_dim`** (int) — action vector width (9, 10, or 8 depending on domain)
- **`action_chunk_size`** (int) — temporal horizon (number of action steps)
- **`action_fps`** (float, 1.0-60.0) — action control frequency
- **`action_space`** (enum: `joint_pos`, `midtrain`) — action space type

Per-dimension semantics are documented only in prose (e.g., "9D = translation + 6D continuous rotation"). No per-dimension joint names, units, coordinate frames, or normalization parameters are exposed as structured metadata.

## 3. GR00T (NVIDIA Isaac-GR00T)

**Source**: [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T)

### What they call it

`ModalityConfig` and `ActionConfig` — per-embodiment configuration that maps modality keys to action properties. `EmbodimentTag` identifies the robot.

### ActionDescriptor equivalent

`ActionConfig` (`gr00t/data/types.py`) fields:

- **`rep`** (`ActionRepresentation` enum): `RELATIVE`, `DELTA`, `ABSOLUTE`
- **`type`** (`ActionType` enum): `EEF` (end-effector), `NON_EEF` (joint-space)
- **`format`** (`ActionFormat` enum): `DEFAULT`, `XYZ_ROT6D`, `XYZ_ROTVEC`
- **`state_key`** (`str | None`): optional reference to which state key this action relates to

`ModalityConfig` (`gr00t/data/types.py`) fields:

- **`delta_indices`** (`list[int]`) — temporal sample indices relative to current timestep
- **`modality_keys`** (`list[str]`) — named fields (e.g., `["single_arm", "gripper"]`)
- **`sin_cos_embedding_keys`** (`list[str] | None`) — keys to apply sin/cos encoding
- **`mean_std_embedding_keys`** (`list[str] | None`) — keys to apply mean/std normalization
- **`action_configs`** (`list[ActionConfig] | None`) — one per modality key

Example (SO100 robot, `examples/SO100/so100_config.py`):

```python
"action": ModalityConfig(
    delta_indices=list(range(0, 16)),   # 16-step horizon
    modality_keys=["single_arm", "gripper"],
    action_configs=[
        ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
        ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
    ]
)
```

N1.7 expanded to 132-dim state/action with 40-step action horizon.

`EmbodimentTag` enum values encode action-space hints in naming (e.g., `oxe_droid_relative_eef_relative_joint`, `real_g1_relative_eef_relative_joints`).

Per-embodiment configs from `gr00t/configs/data/embodiment_configs.py` (`MODALITY_CONFIGS` registry):

| Embodiment | Horizon | modality_keys | ActionConfigs |
| ------------ | --------- | --------------- | --------------- |
| `oxe_droid_relative_eef_relative_joint` | 40 | `["eef_9d", "gripper_position", "joint_position"]` | RELATIVE/EEF/XYZ_ROT6D, ABSOLUTE/NON_EEF, RELATIVE/NON_EEF |
| `unitree_g1_sonic` | 40 | `["motion_token", "left_hand_joints", "right_hand_joints"]` | All ABSOLUTE/NON_EEF |
| `unitree_g1_full_body...` | 50 | `["left_arm", "right_arm", "left_hand", "right_hand", "waist", "base_height_command", "navigate_command"]` | Mixed RELATIVE + ABSOLUTE |
| `libero_sim` | 16 | `["x", "y", "z", "roll", "pitch", "yaw", "gripper"]` | None |
| `simpler_env_widowx` | 8 | `["x", "y", "z", "roll", "pitch", "yaw", "gripper"]` | None |
| `robocasa_gr1_tabletop` | 8 | `["left_arm", "right_arm", "left_hand", "right_hand", "waist"]` | Mixed RELATIVE + ABSOLUTE |

`VLAStepData` (`gr00t/data/types.py`) — core data structure per timestep:

- `actions`: `dict[str, np.ndarray]` — action_name -> `(horizon, dim)` array
- `states`: `dict[str, np.ndarray]` — state_name -> `(dim,)` or `(horizon, dim)`
- `images`: `dict[str, list[np.ndarray]]` — view_name -> temporal stack
- `embodiment`: `EmbodimentTag`

Normalization (`StateActionProcessor`, `gr00t/data/state_action/state_action_processor.py`):

- Stats stored as `{embodiment: {modality: {joint_group: {stat_type: values}}}}`
- Six stat types per joint group: `min`, `max`, `mean`, `std`, `q01`, `q99`
- Three normalization strategies: min/max (default, maps to [-1, 1]), mean/std, sin/cos encoding (doubles dimension)
- `use_percentiles` flag selects q01/q99 instead of min/max
- Stats stored in `meta/stats.json` and `meta/relative_stats.json` with SHA-256 `__fingerprints__` for cache invalidation

Action chunking classes (`gr00t/data/state_action/action_chunking.py`):

- `ActionChunk[PoseType]` base: `_poses: List[PoseType]`, `_times: NDArray[float64]`; horizon = `len(poses)`
- `JointActionChunk`: `to_array()` -> `(N, num_joints)`, supports `relative_chunking()` and `to_absolute_chunking()`
- `EndEffectorActionChunk`: format conversions via `to()` — supports `XYZ_ROT6D` `(N, 9)`, `XYZ_ROTVEC` `(N, 6)`, homogeneous `(N, 4, 4)`; SLERP interpolation for rotations

### ActionChunkDescriptor equivalent

Action horizon defined via `delta_indices` length in `ModalityConfig`. No explicit frequency field — inferred from dataset FPS.

### Key gaps relative to ActionDescriptor

- No per-dimension joint names (modality keys are group-level: `"single_arm"`, not `"shoulder"`, `"elbow"`)
- No per-dimension units, frames, or range bounds
- No per-dimension normalization stats in the config (handled separately by `BaseProcessor.set_statistics()`)
- Actual dimension count per modality key not declared in config — inferred from data

## 4. Open X-Embodiment (OXE)

**Source**: [google-deepmind/open_x_embodiment](https://github.com/google-deepmind/open_x_embodiment)

### What they call it

No dedicated descriptor. Action semantics documented in a Google Sheets spreadsheet external to the dataset format.

### ActionDescriptor equivalent

The standardized action space is 7-DOF EE delta: `(x, y, z, roll, pitch, yaw, gripper_opening)`. Each dimension can represent absolute value, delta change, or velocity — but this is documented per-dataset in the spreadsheet, not in machine-readable metadata.

Datasets use RLDS episode format (built on TensorFlow Datasets / TFDS). Action features defined as:

```python
tfds.features.Tensor(shape=(7,), dtype=np.float32)
```

TFDS `FeatureConnector` metadata per feature:

- `shape` — tensor shape
- `dtype` — data type
- `doc` — free-text description

### ActionChunkDescriptor equivalent

No explicit horizon or frequency metadata in the RLDS/TFDS format. The spreadsheet documents per-dataset control frequency.

### Key gaps relative to ActionDescriptor

- No machine-readable per-dimension semantics — joint names, units, frames, representation type all in external spreadsheet
- No normalization metadata
- No action horizon or frequency in data format
- Interoperability relies on the convention that all datasets conform to the 7-DOF EE delta layout

## 5. LeRobot (Hugging Face)

**Source**: [huggingface/lerobot](https://github.com/huggingface/lerobot)

### What they call it

Features in `info.json` (v3.0 format), stored per-dataset under `meta/info.json`.

### ActionDescriptor equivalent

The `features` section of `info.json` declares per-feature metadata. Confirmed from `lerobot/aloha_sim_insertion_human` (v3.0):

```json
{
  "codebase_version": "v3.0",
  "robot_type": "aloha",
  "fps": 50,
  "total_episodes": 50,
  "total_frames": 25000,
  "total_tasks": 1,
  "chunks_size": 1000,
  "features": {
    "action": {
      "dtype": "float32",
      "shape": [14],
      "names": [
        "left_waist", "left_shoulder", "left_elbow",
        "left_forearm_roll", "left_wrist_angle", "left_wrist_rotate",
        "left_gripper",
        "right_waist", "right_shoulder", "right_elbow",
        "right_forearm_roll", "right_wrist_angle", "right_wrist_rotate",
        "right_gripper"
      ]
    },
    "observation.state": {
      "dtype": "float32",
      "shape": [14],
      "names": ["left_waist", "...same 14 motors..."]
    },
    "observation.images.top": {
      "dtype": "video",
      "shape": [480, 640, 3],
      "names": ["height", "width", "channel"],
      "video_info": {"fps": 50.0, "codec": "av1", "pix_fmt": "yuv420p"}
    }
  }
}
```

Per-feature fields:

- **`dtype`** — data type (`"float32"`, `"int64"`, `"bool"`, `"video"`)
- **`shape`** — dimension array (e.g., `[14]`)
- **`names`** — semantic labels; either `null`, a flat array of strings, or a dict with named groups (e.g., `{"motors": [...]}`)
- **`fps`** — per-feature sampling rate (matches dataset `fps`)
- **`video_info`** — for video features: `fps`, `codec`, `pix_fmt`, `is_depth_map`, `has_audio`

Top-level `info.json` fields: `codebase_version`, `robot_type`, `fps`, `total_episodes`, `total_frames`, `total_tasks`, `chunks_size`, `files_size_in_mb`, `splits`, `data_path`, `video_path`.

Normalization stats in separate `meta/stats.json`:

```json
{
  "action": {
    "min": [per_dim_float, ...],
    "max": [per_dim_float, ...],
    "mean": [per_dim_float, ...],
    "std": [per_dim_float, ...]
  }
}
```

Four statistics per feature (min, max, mean, std), each a list of per-dimension floats.

### ActionChunkDescriptor equivalent

No explicit action horizon concept in the dataset format. `fps` field provides the control/recording frequency. Action chunking is a training-time concern, not a dataset property.

### Key gaps relative to ActionDescriptor

- No per-dimension units, frames, or coordinate systems
- No per-dimension range bounds (min/max)
- No representation type (absolute vs. delta vs. velocity)
- No normalization method declaration (stats exist in `stats.json` but the method — z-score, quantile, etc. — is not declared)
- `names` structure varies across datasets (can be null, flat array, or dict with named groups)

## 6. Gymnasium

**Source**: [Gymnasium](https://gymnasium.farama.org/) (Farama Foundation)

### What they call it

`action_space` — an instance of `gymnasium.spaces.Box` (for continuous actions) or `Discrete`/`MultiDiscrete` (for discrete actions).

### ActionDescriptor equivalent

`Box` constructor parameters:

- **`low`** (`float | NDArray`) — per-dimension lower bounds
- **`high`** (`float | NDArray`) — per-dimension upper bounds
- **`shape`** (`tuple[int, ...]`) — tensor shape
- **`dtype`** (`type`, default `np.float32`) — element type

Methods:

- `is_bounded(manner)` — checks boundedness (`"both"`, `"below"`, `"above"`)

**No semantic metadata whatsoever.** No joint names, units, coordinate frames, representation type, or normalization parameters. The space is purely a numerical specification of shape and bounds.

### ActionChunkDescriptor equivalent

None. Gymnasium's step-based API (`env.step(action)`) processes one action at a time. There is no concept of action horizon or temporal chunking. Control frequency is determined by the simulation timestep, not declared in the space.

### Key gaps relative to ActionDescriptor

- No joint names or element labels
- No units or coordinate frames
- No representation type
- No normalization metadata
- No temporal structure
- Essentially all semantic information is missing — Gymnasium provides only the numerical envelope

## 7. URDF / OpenUSD

### URDF

**Source**: [URDF Joint Spec](http://wiki.ros.org/urdf/XML/joint) (ROS Wiki)

URDF joint elements define (confirmed from [XSD schema](https://github.com/ros/urdfdom/blob/master/xsd/urdf.xsd)):

- **`name`** (attribute, `xs:string`, required) — unique joint identifier
- **`type`** (attribute, `JointType`, required) — `revolute`, `continuous`, `prismatic`, `fixed`, `floating`, `planar`
- **`<axis>`** — axis of rotation/translation; `xyz` attribute (`xs:string`, default `"1 0 0"`)
- **`<limit>`** — all `xs:double`, default `0`: `lower` (rad or m), `upper` (rad or m), `effort` (max force/torque, Nm or N), `velocity` (max velocity, rad/s or m/s)
- **`<parent>`**, **`<child>`** — connected links (`link` attribute, required)
- **`<origin>`** — transform from parent link frame (xyz + rpy)
- **`<dynamics>`** — `damping` (default 0), `friction` (default 0)
- **`<calibration>`** — `reference_position`, `rising`, `falling` (all `xs:double`)
- **`<safety_controller>`** — `soft_lower_limit`, `soft_upper_limit`, `k_position` (default 0), `k_velocity` (required)
- **`<mimic>`** — `joint` (string, required), `multiplier` (default 1), `offset` (default 0) — for coupled joints

Units are implicit by convention: radians for revolute/continuous, meters for prismatic.

### OpenUSD (UsdPhysics)

**Source**: [UsdPhysicsRevoluteJoint](https://openusd.org/release/api/class_usd_physics_revolute_joint.html)

`UsdPhysicsRevoluteJoint` attributes:

- **`physics:axis`** — `X`, `Y`, or `Z` (token)
- **`physics:lowerLimit`** — lower limit in degrees (float, default `-inf`)
- **`physics:upperLimit`** — upper limit in degrees (float, default `inf`)

Inherited from `UsdPhysicsJoint`:

- **`localPos0/1`**, **`localRot0/1`** — joint frame relative to each body
- **`jointEnabled`**, **`collisionEnabled`**, **`breakForce`**, **`breakTorque`**
- **`body0Rel`**, **`body1Rel`** — connected body relationships

### Overlap with action dimensions

URDF/USD joint definitions describe the physical properties that constrain action dimensions:

- Joint name → action dimension label
- Joint type → determines representation domain (rotational vs. translational)
- Joint limits → valid action range
- Joint axis → which degree of freedom

However, they describe the **robot**, not the **action space of a policy**. The mapping from URDF joints to action vector dimensions (ordering, which joints are actuated, what representation — position vs. velocity vs. torque) is not part of URDF/USD.

### Key gaps relative to ActionDescriptor

- No action ordering (URDF joints are a tree, not an ordered vector)
- No representation type (position vs. velocity vs. torque — the action domain)
- No normalization metadata
- No temporal structure
- Units differ from ML conventions (URDF uses radians, USD uses degrees)

## 8. ROS 2

**Source**: [trajectory_msgs](https://github.com/ros2/common_interfaces/blob/rolling/trajectory_msgs/msg/JointTrajectory.msg), [sensor_msgs](https://github.com/ros2/common_interfaces/blob/rolling/sensor_msgs/msg/JointState.msg)

### What they call it

`JointTrajectory` (for commands) and `JointState` (for observations) messages.

### ActionDescriptor equivalent

**`JointState.msg`**:

```
std_msgs/Header header
string[]  name       # joint names
float64[] position   # rad or m
float64[] velocity   # rad/s or m/s
float64[] effort     # Nm or N
```

**`JointTrajectoryPoint`** (within `JointTrajectory`):

```
float64[] positions
float64[] velocities
float64[] accelerations
float64[] effort
builtin_interfaces/Duration time_from_start
```

**`JointTrajectory.msg`**:

```
std_msgs/Header header
string[]              joint_names
JointTrajectoryPoint[] points
```

Per-dimension semantics:

- **Joint names** — explicit `string[]` field (`name` in JointState, `joint_names` in JointTrajectory)
- **Multiple representations** — position, velocity, acceleration, effort arrays coexist (all optional, same ordering as `joint_names`)
- **Units** — by convention: radians/meters for position, rad/s or m/s for velocity, Nm or N for effort
- **Coordinate frame** — in `header.frame_id`
- **Timestamp** — in `header.stamp`

### ActionChunkDescriptor equivalent

`JointTrajectory` is inherently a temporal sequence: the `points[]` array contains multiple `JointTrajectoryPoint` entries, each with a `time_from_start` duration. This is functionally an action chunk with:

- Horizon = `len(points)`
- Per-point timing via `time_from_start` (variable-rate, not fixed frequency)

### Key gaps relative to ActionDescriptor

- No per-dimension range bounds (URDF carries those, but they're not in the message)
- No normalization metadata
- Units implicit by convention, not declared in message
- No representation type annotation (the message has separate position/velocity/effort arrays, but doesn't declare which the controller should use)
- No dtype — always float64

## Summary Table

| Community | Concept name | Joint names | Units/frames | Rep. type | Range bounds | Normalization | Horizon | Frequency | Format |
| ----------- | ------------- | ------------- | -------------- | ----------- | ------------- | --------------- | --------- | ----------- | -------- |
| **OpenPI** | (none) | Hardcoded in Python | No | `DeltaActions` mask | No | `NormStats` JSON | `action_horizon` in config | No | Python dataclass + JSON |
| **LEAPP** | `TensorSemantics` | `element_names` | `extra` dict | `kind` enum | No | No (baked in) | `TemporalAxis.period_ms` | `period_ms` | Python + YAML |
| **GR00T** | `ModalityConfig` + `ActionConfig` | Group-level keys only | No | `rep`/`type`/`format` enums | No | Separate stats | `delta_indices` length | No | Python/JSON |
| **OXE** | (none) | Spreadsheet only | Spreadsheet only | Spreadsheet only | No | No | No | Spreadsheet only | RLDS/TFDS |
| **LeRobot** | `features` in `info.json` | `names` field | No | No | No | Separate `stats.json` | No | `fps` field | JSON |
| **Gymnasium** | `action_space` (Box) | No | No | No | `low`/`high` | No | No | No | Python |
| **URDF/USD** | Joint element | `name` attr | Implicit convention | Joint `type` | `limit` element | No | No | No | XML/USD |
| **ROS 2** | `JointTrajectory` | `joint_names` | `header.frame_id` | Separate arrays | No | No | `points[]` length | `time_from_start` | .msg |

## Observations

1. **No community has a complete ActionDescriptor.** LEAPP comes closest with `TensorSemantics` (kind, element_names, extra for units/frame, temporal axis), but lacks range bounds and normalization metadata.

2. **Joint names are the most commonly represented property** — present in LEAPP, LeRobot, ROS 2, and URDF, but absent from OpenPI configs, GR00T configs (group-level only), Gymnasium, and OXE's machine-readable format.

3. **Representation type (position/velocity/torque, absolute/delta/EE)** is partially addressed by LEAPP (kind enum), GR00T (ActionRepresentation + ActionType), and ROS 2 (separate arrays). OpenPI encodes it in transform masks. Others don't declare it.

4. **Normalization is universally decoupled** from action descriptors. OpenPI stores it in `NormStats` JSON. LeRobot uses `stats.json`. GR00T uses `BaseProcessor.set_statistics()`. No community includes normalization metadata in the action descriptor itself.

5. **Temporal structure is sparsely represented.** LEAPP's `TemporalAxis` is the most explicit (period_ms + axis marker). OpenPI has `action_horizon` in config. GR00T has `delta_indices`. ROS 2's `JointTrajectory` is inherently temporal. LeRobot, Gymnasium, and OXE have no horizon concept in their data formats.

6. **Control frequency is rarely declared.** Only LeRobot (`fps`), LEAPP (`period_ms`), and Cosmos 3 (`action_fps`) include it. Others rely on dataset metadata or simulation settings.

7. **The robot-description/action-space gap is universal.** URDF/USD describe joints with names, types, limits, and axes — properties needed for action semantics — but no community has a standard mapping from robot description joints to policy action vector dimensions.
