# Descriptor Survey — TransformDescriptor & MappingDescriptor

**Date**: 2026-09-21
**Purpose**: Document how Physical AI communities represent data transforms (normalization, preprocessing) and mapping/adaptation between model and robot interfaces.
**Reference model**: TransformDescriptor (type, strategy, parameters, stats_reference) and MappingDescriptor (dimension reordering, sign flips, camera remapping, gripper conversion, coordinate transforms, value scaling).

---

## 1. OpenPI (Physical Intelligence)

### What they call the concept

- **Normalization**: `Normalize` / `Unnormalize` dataclass transforms in `src/openpi/transforms.py`
- **Mapping**: Per-robot `*Inputs` / `*Outputs` adapter classes (e.g., `DroidInputs`, `AlohaInputs`) in `src/openpi/policies/`
- **Statistics**: `NormStats` dataclass in `src/openpi/shared/normalize.py`, persisted as `norm_stats.json`

### Explicitly declared properties

**Normalization:**

- `NormStats` dataclass: `mean` (NDArray), `std` (NDArray), `q01` (NDArray | None), `q99` (NDArray | None)
- Strategy selection: `use_quantile_norm: bool` on `DataConfig`. PI0 uses z-score (False); PI0_FAST and PI0.5 use quantile (True). No strategy enum -- it is a bare boolean.
- Z-score formula: `(x - mean) / (std + 1e-6)`
- Quantile formula: `(x - q01) / (q99 - q01 + 1e-6) * 2 - 1` (maps to [-1, 1])
- `norm_stats.json`: dict mapping feature names (e.g., `"actions"`, `"observation/state"`) to `{mean, std, q01, q99}` arrays. Serialized via Pydantic.
- `RunningStats`: streaming stats computation with 5000-bin histograms for quantile estimation.

**Transform pipeline composition** (in `DataConfig` / `create_trained_policy()`):

1. `repack_transforms` -- dataset key remapping
2. `data_transforms` -- robot-specific transforms (pre-normalization)
3. Normalize (using `norm_stats` + `use_quantile_norm`)
4. `model_transforms` -- model-specific transforms (post-normalization, e.g., tokenization, image resize)

Output pipeline is the reverse: model_transforms.outputs -> Unnormalize -> data_transforms.outputs -> repack_transforms.outputs.

**Mapping (per-robot classes):**

`DroidInputs` (`src/openpi/policies/droid_policy.py`):

- Camera remapping (hardcoded, model-type dependent): `exterior_image_1_left -> base_0_rgb`, `wrist_image_left -> left_wrist_0_rgb`, padding for `right_wrist_0_rgb`. PI0_FAST uses different mapping.
- State concatenation: 7-dim joint_position + 1-dim gripper_position = 8-dim state vector.
- Image format: CHW -> HWC, float -> uint8 (*255).
- No sign flips, no gripper conversion.
- `DroidOutputs`: action sliced to first 8 dims (hardcoded).

`AlohaInputs` (`src/openpi/policies/aloha_policy.py`):

- Camera remapping: `cam_high -> base_0_rgb`, `cam_left_wrist -> left_wrist_0_rgb`, `cam_right_wrist -> right_wrist_0_rgb`.
- Sign flip mask: `[1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1]` (14-element, hardcoded). Joints at indices 1, 2, 8, 9 negated.
- Gripper conversion (linear -> angular): `arcsin(clip((horn_radius^2 + pos^2 - arm_length^2) / (2 * horn_radius * pos), -1, 1))` with `arm_length=0.036`, `horn_radius=0.022`, puppet gripper range min=0.01844, max=0.05800, angular range min=0.5476, max=1.6296. Applied at gripper indices [6, 13] in 14-dim state.
- `adapt_to_pi: bool = True` controls whether these conversions are applied.
- `AlohaOutputs`: action sliced to first 14 dims, then inverse gripper conversion applied.

`LiberoInputs`: camera remapping (`observation/image -> base_0_rgb`, `observation/wrist_image -> left_wrist_0_rgb`), state passthrough (8-dim). `LiberoOutputs`: action sliced to 7 dims.

**Image preprocessing** (in `src/openpi/models/model.py`):

- uint8 images normalized to [-1, 1]: `data / 255.0 * 2.0 - 1.0`
- Resize with padding to 224x224 via `resize_with_pad`
- Color augmentation during training only (RandomCrop, Rotate, ColorJitter via augmax)

### Implicit or hardcoded properties

| Concept | Status |
| --- | --- |
| Camera name remapping | Hardcoded per-robot Python class |
| Sign flip masks | Hardcoded constants |
| Gripper conversion constants | Hardcoded constants (derived from Interbotix hardware) |
| Action dim slicing | Hardcoded integer per robot |
| State concatenation order | Hardcoded logic |
| Image format conversion (CHW->HWC) | Hardcoded |
| Camera slot naming convention | De facto standard (`base_0_rgb`, `left_wrist_0_rgb`, etc.) but not formally specified |
| Which adapter class to use | Determined by config name, not stored in model metadata |

### Key gaps relative to TransformDescriptor/MappingDescriptor

1. **No declarative mapping format** -- all robot-specific mappings are Python code, not config.
2. **No strategy enum** -- normalization strategy is a bare boolean.
3. **No mapping metadata in checkpoints** -- which input class to use is determined by config name, not declared in model metadata.
4. **No sign flip / gripper conversion parameters in any config file** -- all baked into Python constants.
5. Camera slot naming is a de facto standard but not formally specified.

**Sources**: [OpenPI transforms.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/transforms.py), [OpenPI normalize.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/shared/normalize.py), [OpenPI droid_policy.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/policies/droid_policy.py), [OpenPI aloha_policy.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/policies/aloha_policy.py), [OpenPI config.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py)

---

## 2. LeRobot (Hugging Face)

### What they call the concept

- **Normalization**: `NormalizerProcessorStep` / `UnnormalizerProcessorStep` in `src/lerobot/processor/normalize_processor.py`
- **Normalization mode mapping**: `norm_map: dict[FeatureType, NormalizationMode]`
- **Statistics**: `stats.json` in dataset metadata (`src/lerobot/datasets/compute_stats.py`)
- **Transform pipeline**: `PolicyProcessorPipeline` with ordered `ProcessorStep` instances

### Explicitly declared properties

**NormalizationMode enum** (5 modes):

| Mode | Required stats | Forward formula | Inverse formula | Output range |
| --- | --- | --- | --- | --- |
| `IDENTITY` | none | `return tensor` | `return tensor` | unchanged |
| `MEAN_STD` | mean, std | `(x - mean) / (std + eps)` | `x * std + mean` | centered, unit variance |
| `MIN_MAX` | min, max | `2 * (x - min) / (max - min) - 1` | `(x + 1) / 2 * (max - min) + min` | [-1, 1] |
| `QUANTILES` | q01, q99 | `2 * (x - q01) / (q99 - q01) - 1` | `(x + 1) * (q99 - q01) / 2 + q01` | [-1, 1] |
| `QUANTILE10` | q10, q90 | `2 * (x - q10) / (q90 - q10) - 1` | `(x + 1) * (q90 - q10) / 2 + q10` | [-1, 1] |

**Normalization mapping** (`norm_map`): Maps `FeatureType` (VISUAL, STATE, ACTION) to a `NormalizationMode`. Defaults to `IDENTITY` for unmapped types. Stored in processor config JSON with enum values as strings.

**`stats.json` format** -- per-feature statistics with these fields:

| Field | Description | Shape |
| --- | --- | --- |
| `min` | Per-dimension minimum | `(D,)` for vectors, `(C,1,1)` for images |
| `max` | Per-dimension maximum | same |
| `mean` | Per-dimension mean | same |
| `std` | Per-dimension standard deviation | same |
| `count` | Number of samples | `(1,)` |
| `q01` | 1st percentile | same as min/max |
| `q10` | 10th percentile | same |
| `q50` | 50th percentile (median) | same |
| `q90` | 90th percentile | same |
| `q99` | 99th percentile | same |

Default quantiles: `[0.01, 0.10, 0.50, 0.90, 0.99]`, with keys `q{int(q*100):02d}`.

**Processor pipeline** (e.g., for pi0 in `src/lerobot/policies/pi0/processor_pi0.py`):

Input pipeline order:

1. `RenameObservationsProcessorStep` -- feature name remapping
2. `AddBatchDimensionProcessorStep`
3. `Pi0NewLineProcessor` -- appends `\n` to task strings
4. `TokenizerProcessorStep` -- PaliGemma tokenizer
5. `DeviceProcessorStep` -- move to GPU
6. `RelativeActionsProcessorStep` -- optional delta action conversion
7. `NormalizerProcessorStep` -- normalization using dataset_stats

Output pipeline:

1. `UnnormalizerProcessorStep` -- reverse normalization
2. `AbsoluteActionsProcessorStep` -- optional inverse of relative actions
3. `DeviceProcessorStep(device="cpu")`

**Migration from embedded normalization** (`src/lerobot/processor/migrate_policy_normalization.py`):

LeRobot migrated from normalization layers embedded in model state_dicts to external `PolicyProcessorPipeline`. The migration script strips 11+ legacy key patterns (e.g., `normalize_inputs.buffer_*`, `unnormalize_outputs.buffer_*`) and extracts stats into the new processor config files (`preprocessor_config.json`, `postprocessor_config.json`).

**`hotswap_stats()`**: A function that deep-copies a processor pipeline and replaces normalization stats, enabling adaptation to new data distributions without retraining.

### Implicit or hardcoded properties

| Concept | Status |
| --- | --- |
| Camera name remapping | Via `RenameObservationsProcessorStep` -- configurable but manual |
| Sign flips | Not handled in the generic processor pipeline |
| Gripper conversion | Not handled in the generic processor pipeline |
| Action dim slicing | Not handled generically |
| Feature type classification | Heuristic: keys with "image"/"visual" -> VISUAL, "state"/"joint" -> STATE, "action" -> ACTION |

### Known issues

- **Issue #4415**: Normalization-related bug (details not retrievable due to search budget).
- **Issue #4647**: Normalization-related bug (details not retrievable due to search budget).
- The migration path from embedded-in-model to external-processor normalization indicates that the normalization architecture has been unstable/evolving.

### Key gaps relative to TransformDescriptor/MappingDescriptor

1. **No MappingDescriptor equivalent** -- LeRobot's processor pipeline handles normalization but not embodiment-specific mappings (sign flips, gripper conversion, joint reordering).
2. **No formal action semantics** -- `stats.json` has statistics per feature key but no declaration of what action dimensions mean.
3. **Feature type classification is heuristic** -- relies on key name patterns, not explicit declarations.
4. **No camera role declarations** -- camera features are identified by naming convention only.

**Sources**: [LeRobot normalize_processor.py](https://github.com/huggingface/lerobot/blob/main/src/lerobot/processor/normalize_processor.py), [LeRobot compute_stats.py](https://github.com/huggingface/lerobot/blob/main/src/lerobot/datasets/compute_stats.py), [LeRobot processor_pi0.py](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/pi0/processor_pi0.py), [LeRobot migrate_policy_normalization.py](https://github.com/huggingface/lerobot/blob/main/src/lerobot/processor/migrate_policy_normalization.py)

---

## 3. vLLM-Omni

### What they call the concept

- **Preprocessing**: `Pi0ImageProcessor`, `build_model_inputs()` in `vllm_omni/diffusion/models/pi0/processor_pi0.py`
- **Normalization**: `_normalize_state()` / `_unnormalize_actions()` methods on `Pi0ForActionPrediction` in `modeling_pi0.py`
- **Config**: `Pi0Config` in `vllm_omni/diffusion/models/pi0/config.py` with `norm_stats: dict | None`
- **GR00T mapping**: `StateActionProcessor` in `vllm_omni/diffusion/models/gr00t/dataio/state_action/state_action_processor.py`

### Explicitly declared properties

**Pi0 normalization config:**

- `norm_stats: dict | None` -- loaded from `config.json` in checkpoint directory. Schema described as matching "LeRobot's `NormalizerProcessorStep`". When `None`, acts as identity/pass-through.
- No `use_quantile_norm` field -- vLLM-Omni does not expose quantile normalization as a config option.
- `image_key_map: dict[str, str]` -- maps raw observation keys to checkpoint feature keys for camera identity translation. Empty dict = identity match.
- `image_feature_keys: list[str] | None` -- ordered camera keys from checkpoint's `input_features`, auto-derived from keys starting with `"observation.images."`.

**Pi0 image preprocessing (`Pi0ImageProcessor`):**

- uint8 -> float32: `pixel / 255.0 * 2.0 - 1.0` (SigLIP-style [-1, 1])
- `resize_with_pad`: bilinear interpolation, symmetric padding with fill value -1.0, clamp to [-1, 1]
- Empty/missing cameras: `torch.full(..., -1.0)` with mask=False
- Constants: `PI0_IMAGE_SIZE = 224`, `PI0_MAX_CAMERAS = 3`, `PI0_NUM_IMAGE_TOKENS = 256`

**Pi0 state preprocessing:**

- Converts to float32, zero-pads to `max_state_dim` (32) or truncates if larger.

**Pi0 norm_stats loading:**

- Primary: from deploy YAML via `Pi0Config.from_model_config()`
- Fallback: from checkpoint directory via `Pi0Config.from_pretrained()` if deploy YAML omits them
- Applied via `_build_norm_buffers` on the model (details in modeling code, not processor)

**GR00T `StateActionProcessor`** (separate from pi0):

- Per-embodiment configuration via `ModalityConfig` objects indexed by `embodiment_tag`
- Three normalization strategies per joint group:
  - **Sin/cos encoding** (state only): doubles dimension (D -> 2D), not reversible
  - **Mean/std normalization**: `(x - mean) / std`
  - **Min/max normalization** (default): `2 * (x - min) / (max - min) - 1` with optional percentile (q01/q99) substitution and outlier clipping to [-1, 1]
- Absolute <-> relative action conversion per joint group via `ActionRepresentation.RELATIVE` / `ABSOLUTE`
- `ActionConfig` per joint group: `ActionRepresentation`, `ActionType` (EEF/NON_EEF), `ActionFormat`, optional `state_key`
- Dimension queries: `get_state_dim()`, `get_action_dim()` per embodiment tag

### Implicit or hardcoded properties

| Concept | Status |
| --- | --- |
| Camera key translation | Configurable via `image_key_map` -- better than OpenPI's hardcoded mappings |
| Camera ordering | Declared in `image_feature_keys` from checkpoint |
| Image normalization range | Hardcoded [-1, 1] (SigLIP convention) |
| Prompt suffix | Hardcoded `\n` append (matching LeRobot convention) |
| Max cameras | Hardcoded `PI0_MAX_CAMERAS = 3` |
| State zero-padding dim | Hardcoded or from config |

### Key gaps relative to TransformDescriptor/MappingDescriptor

1. **No normalization strategy declaration for pi0** -- `norm_stats` is loaded but whether to apply z-score vs quantile is not declared in config.
2. **GR00T has the richest mapping model** (per-group normalization strategy, relative/absolute action handling, embodiment-specific configs) but it's proprietary to the GR00T model family.
3. **API is a transparent transport** -- the OpenPI-compatible WebSocket API explicitly delegates all transforms to the policy pipeline server-side. No transform metadata is exposed to the client.

**Sources**: [vLLM-Omni processor_pi0.py](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/pi0/processor_pi0.py), [vLLM-Omni Pi0Config](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/pi0/config.py), [vLLM-Omni pipeline_pi0.py](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/pi0/pipeline_pi0.py), [vLLM-Omni StateActionProcessor](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/gr00t/dataio/state_action/state_action_processor.py), [vLLM-Omni OpenPI API docs](https://github.com/vllm-project/vllm-omni/blob/main/docs/serving/openpi_api.md)

---

## 4. SGLang

### What they call the concept

- **Preprocessing**: `Pi05Preprocessor` in `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/pi05_preprocess.py`
- **Configuration**: `Pi05PipelineConfig` in `python/sglang/multimodal_gen/configs/pipeline_configs/pi05.py`

### Explicitly declared properties

**Image preprocessing (`_preprocess_image`):**

- Channel detection: if first dim in (1,3,4), assumes CHW; if last dim in (1,3,4), assumes HWC.
- Normalization detection: checks `is_byte_scaled` (integer or max > 2.0), `is_normalized` (min < 0.0).
- Byte-scaled images: `/ 255.0` then `* 2.0 - 1.0` to [-1, 1].
- Already-normalized images: clamp to [-1, 1].
- `resize_with_pad`: bilinear interpolation, byte-rounding step (`round(tensor * 255.0).clamp(0, 255) / 255.0`) to match OpenPI reference, padding with -1.0 for normalized or 0.0 for unnormalized.

**State discretization:**

- 256 bins linearly spaced from -1 to 1: `np.linspace(-1, 1, 257)[:-1]`
- `np.digitize(state, bins) - 1` for zero-indexed bin indices.
- Discretized state embedded as text tokens in the prompt.

**Prompt template**: `"Task: {cleaned}, State: {state_str};\nAction: "`

**Pipeline config (`Pi05PipelineConfig`):**

| Field | Default | Purpose |
| --- | --- | --- |
| `image_size` | `(224, 224)` | Target resolution |
| `image_keys` | `("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")` | Expected camera slots |
| `empty_cameras` | `0` | Number of padded camera slots |
| `action_horizon` | `50` | Steps per action chunk |
| `action_dim` | `32` | Action vector dimensionality |
| `state_dim` | `32` | State vector dimensionality |
| `output_action_dim` | `32` | Output dimensions |
| `max_token_len` | `200` | Max prompt tokens |
| `default_num_inference_steps` | `10` | Flow matching denoising steps |
| `image_normalization_mean` | `(0.5, 0.5, 0.5)` | Image norm (channel-wise) |
| `image_normalization_std` | `(0.5, 0.5, 0.5)` | Image norm (channel-wise) |

### Implicit or hardcoded properties

| Concept | Status |
| --- | --- |
| Norm_stats / quantile normalization | **Not present** -- no norm_stats loading, no z-score/quantile normalization in the preprocessor |
| Camera slot names | Hardcoded in config defaults |
| State discretization bins | Hardcoded 256 bins in [-1, 1] |
| Missing camera fill value | Hardcoded -1.0 |
| Image normalization formula | Hardcoded SigLIP-style `* 2 - 1` |
| Byte-rounding emulation | Hardcoded to match OpenPI reference |

### Key gaps relative to TransformDescriptor/MappingDescriptor

1. **No normalization infrastructure** -- SGLang's preprocessor handles only image preprocessing and state discretization. No norm_stats, no z-score/quantile transforms.
2. **No mapping logic** -- no sign flips, gripper conversion, or joint reordering. The preprocessor is model-specific, not robot-specific.
3. **State discretization is a novel transform type** (quantization to bins) not covered by our TransformDescriptor model.
4. **All parameters are hardcoded** in the pipeline config -- no mechanism for per-robot or per-checkpoint configuration.

**Sources**: [SGLang pi05_preprocess.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/pi05_preprocess.py), [SGLang pi05 pipeline config](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/configs/pipeline_configs/pi05.py)

---

## 5. LEAPP (NVIDIA)

### What they call the concept

- **Normalization**: "Observation normalizer" baked into ONNX graph at export time.
- **Tensor semantics**: Per-dimension metadata on model I/O tensors -- `kind`, `element_names`, `extra`.
- **Export**: `leapp export --embodiment_tag ... --joint_config ...`

### Explicitly declared properties

**Observation normalizer baked into ONNX:**

- RSL-RL's `EmpiricalNormalization` (running mean/variance) traced into the ONNX computational graph as constant nodes.
- Formula: `normalized_obs = (obs - mean) / sqrt(variance + epsilon)`
- Raw observations fed to ONNX are automatically normalized -- no external preprocessing needed.
- The normalizer parameters come from `policy.actor_obs_normalizer` in the trained checkpoint.

**Tensor semantics (the most declarative metadata in any ecosystem):**

- `kind`: Semantic category per tensor dimension -- `JOINT_POSITION`, `JOINT_VELOCITY`, `GRIPPER_STATE`, `EE_POSITION`, `EE_ORIENTATION`, etc.
- `element_names`: Per-dimension names (e.g., `hip_l`, `knee_l`, `ankle_l`).
- `extra`: Extensible dict per dimension for `frame` (reference frame) and `units` (radians, meters).

**Export-time parameters:**

- `--embodiment_tag`: Selects robot-specific preprocessing and action head.
- `--joint_config`: Joint configuration file specifying active joints, ordering, and control properties.

**ONNX bundle contents:**

- Full pipeline (preprocessing + backbone + action head) traced as ONNX.
- Model input/output tensor names and shapes in ONNX metadata.
- `.yaml` companion files (presumably containing normalizer parameters or metadata).

### Implicit or hardcoded properties

| Concept | Status |
| --- | --- |
| Joint ordering | Baked into ONNX at export via `--joint_config` |
| Normalization | Baked into ONNX graph as constant nodes |
| Embodiment tag semantics | Opaque string -- mapping from tag to joint config is internal |
| Camera/observation semantics | Not covered -- tensor semantics annotate action outputs only |
| Joint config format | Not publicly documented |

### Key gaps relative to TransformDescriptor/MappingDescriptor

1. **Baking transforms into the artifact** is the opposite of our model (which declares transforms as metadata). Once baked, the transform parameters are not inspectable or composable.
2. **Camera/observation semantics not covered** by tensor semantics.
3. **No cross-artifact compatibility checking** -- cannot verify model-robot compatibility without running the export.
4. **`extra` fields are conventions, not enforced schema** -- `frame` and `units` are documented but not validated.

**Sources**: [LEAPP docs](https://nvidia-isaac.github.io/leapp/), [LEAPP tensor semantics](https://nvidia-isaac.github.io/leapp/semantics/usage.html), [nvidia-industrial-wbc-pipeline export_onnx.py](https://github.com/say-paul/nvidia-industrial-wbc-pipeline/blob/main/src/wbc_pipeline/export_onnx.py)

---

## 6. Rosetta

### What they call the concept

- **Contract**: A YAML file that maps between ROS 2 topics and LeRobot features. Used identically for recording, bag conversion, and deployment.
- **Channel**: A binding between a ROS 2 topic (name + message type) and a training feature key.

### Explicitly declared properties

**YAML contract structure:**

```yaml
robot_type: my_robot
robot_interface: ros2
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

**Mapping concepts covered:**

| Concept | How handled |
| --- | --- |
| Topic-to-feature mapping | `channel.topic` + feature key name |
| Joint selection + ordering | `select: [position.j1, position.j2]` -- dot-path into ROS message fields |
| Camera-to-feature mapping | Feature key `observation.images.cam` bound to camera topic |
| Image transforms | `apply: [resize: [480, 640]]` -- limited to resize |
| Temporal alignment | `align.strategy` (hold) + `align.timeline` (header) |
| Message type | `channel.type` -- ROS 2 message type declaration |

### Implicit or not covered

| Concept | Status |
| --- | --- |
| Normalization | Not covered -- no normalization parameters or stats references |
| Sign flips | Not covered |
| Gripper conversion | Not covered |
| Coordinate transforms | Not covered (no TF frame references) |
| Action semantics | Not covered -- action dimensions are selected but not semantically annotated |
| Value scaling | Not covered |

### Key gaps relative to TransformDescriptor/MappingDescriptor

1. **Covers transport-level mapping only** -- which ROS 2 topics map to which training features, with field selection and temporal alignment.
2. **No normalization or value transforms** -- the contract bridges the ROS 2 / LeRobot boundary but does not describe any data transformations.
3. **No action or observation semantics** -- joint names appear as message field paths (e.g., `position.j1`) but without semantic annotations.
4. **Single-purpose**: designed specifically for the ROS 2 <-> LeRobot boundary, not a general-purpose transform/mapping descriptor.

The contract's strength is its single-source-of-truth design: the same YAML is used for recording, conversion, and deployment, preventing train-serve skew.

**Sources**: [Rosetta GitHub (iblnkn/rosetta)](https://github.com/iblnkn/rosetta), [Rosetta in ros-physical-ai/demos](https://github.com/ros-physical-ai/demos)

---

## 7. knownrobot

### What they call the concept

- **Skill manifest**: `robot-skill.yaml` (or `.json`) -- a portable metadata manifest covering policy identity, hardware, runtime, compatibility, and evaluation evidence.
- **Evidence contract (1.0)**: The schema covering 10 domains.
- **JSON Schema**: `robot-skill.schema.json` (JSON Schema Draft 2020-12)

### Explicitly declared properties

**Schema fields** (from `embodied-registry/schema/robot-skill.schema.json`):

| Section | Required fields | Purpose |
| --- | --- | --- |
| `skill` | name, version, source (type, repository, revision) | Skill identity and provenance |
| `policy` | framework, framework_version, architecture, checkpoint | Policy metadata |
| `hardware` | robot_family, gripper, sensors[] (type, name, calibration) | Hardware declaration |
| `runtime` | control_frequency_hz, observation_shape, action_shape, dependencies | Runtime contract |
| `dataset` | repository, schema | Training data reference |
| `compatibility[]` | robot_family, status (compatible/incompatible/untested), evidence, adapter | Cross-robot compatibility |
| `evaluations[]` | benchmark, trials, successes, evaluator, date | Evaluation evidence |
| `attribution[]` | name, role, source_url | Credit tracking |

**Example manifest** (SO-101 with ACT policy):

- `observation_shape`: `{"observation.images.wrist": [3, 480, 640], "observation.state": [6]}`
- `action_shape`: `{"action": [6]}`
- `sensors`: `[{type: "rgb", name: "wrist", calibration: "calibration/wrist.json"}]`
- `compatibility`: `[{robot_family: "SO-101", status: "compatible", adapter: null}]`

### Implicit or not covered

| Concept | Status |
| --- | --- |
| Normalization | Not covered -- no normalization parameters, stats, or strategy |
| Transforms | Not covered -- no data transform declarations |
| Camera mapping | Partial -- sensors have `type` and `name`, but no camera-to-model-input mapping |
| Joint mapping | Not covered -- no joint names, ordering, or limits |
| Sign flips | Not covered |
| Gripper conversion | Not covered |
| Action semantics | Not covered -- `action_shape` gives dimensions but not meaning |

### Key gaps relative to TransformDescriptor/MappingDescriptor

1. **No TransformDescriptor equivalent** -- the schema is focused on identity, compatibility evidence, and hardware declarations, not data transforms.
2. **No MappingDescriptor equivalent** -- `compatibility[].adapter` is a nullable string reference, not a structured mapping description.
3. **Shapes without semantics** -- `observation_shape` and `action_shape` declare dimensions but not what they mean.
4. **Sensor calibration is a file reference** (`calibration: "calibration/wrist.json"`) -- not inline structured data.
5. **Schema is minimal by design** -- focused on pre-deployment compatibility checking, not runtime transform configuration.

**Sources**: [arcofdescent1/knownrobot](https://github.com/arcofdescent1/knownrobot), [robot-skill.schema.json](https://github.com/arcofdescent1/knownrobot/blob/main/embodied-registry/schema/robot-skill.schema.json), [example.robot-skill.json](https://github.com/arcofdescent1/knownrobot/blob/main/embodied-registry/schema/example.robot-skill.json)

---

## 8. GR00T (NVIDIA)

### What they call the concept

- **Embodiment adaptation**: Embodiment-specific encoder/decoder networks selected by `embodiment_tag` (an `EmbodimentTag` enum).
- **Modality config**: `modality.json` -- per-dataset JSON declaring semantic array slicing and video key remapping.
- **Normalization**: Per-joint-group min/max normalization with optional percentile (q01/q99) substitution, stored in `statistics.json` and `relative_stats.json`.
- **Processor pipeline**: `GrootN17PackInputsStep`, `GrootN17ActionDecodeStep`, `GrootN17VLMEncodeStep` in LeRobot's `src/lerobot/policies/groot/processor_groot.py`.

### Explicitly declared properties

**`modality.json`** (per-dataset, e.g., `examples/SO100/modality.json`):

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
    "front": {"original_key": "observation.images.front"},
    "wrist": {"original_key": "observation.images.wrist"}
  },
  "annotation": {
    "human.task_description": {"original_key": "task_index"}
  }
}
```

Declares semantic array slicing (index ranges to named groups), video key remapping, and annotation key remapping.

**`EmbodimentTag` enum** (`gr00t/data/embodiment_tags.py`):

- Pretrain tags (in base model): `OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT`, `XDOF`, `REAL_G1`, `REAL_R1_PRO_SHARPA`, etc.
- Posttrain tags (require finetuned checkpoint): `UNITREE_G1`, `UNITREE_G1_SONIC`, `SIMPLER_ENV_GOOGLE`, `LIBERO_PANDA`, etc.
- Finetune-only tags: `NEW_EMBODIMENT`, `ROBOCASA_GR1_TABLETOP`, `ROBOCASA_PANDA_OMRON`.
- Tag naming convention encodes action representation: e.g., `xdof_relative_eef_relative_joint`.
- Tags route to appropriate action-space projector heads in the model.
- Custom mappings loadable from `embodiment_id.json` in checkpoint directory.

**Normalization (`GrootN17PackInputsStep`):**

- Default: min/max normalization: `2 * (x - min) / (max - min) - 1` -> [-1, 1].
- Percentile mode (`use_percentiles=True`): q01/q99 replace min/max.
- `clip_outliers=True`: clamp output to [-1, 1].
- Per-joint-group, per-horizon-timestep 2D stats for relative actions.
- Stats loaded from `statistics.json` (per-embodiment) and `relative_stats.json`.

**Relative action handling:**

- `_infer_n1_7_action_groups()`: partitions action dimensions into groups, marking some as relative.
- Relative groups: `action[..., start:end] -= reference_state[:, None, :]`.
- Separate `_make_relative_action_training_stats()` computes per-chunk-timestep statistics with native action horizon of 40.

**State processing:**

- State validated to fit `max_state_dim` (default 132).
- Min/max normalized, zero-padded to max dim.
- State dropout during training: with probability `state_dropout_prob`, entire batch items' states zeroed.

**Checkpoint processor assets** (`_GrootN17CheckpointProcessorAssets`):

- `stats`, `raw_stats`, `modality_config`, `embodiment_mapping`
- `use_percentiles`, `use_relative_action`, `state_dropout_prob`, `clip_outliers`
- `video_modality_keys`, `image_crop_size`, `image_target_size`
- Loaded from `processor_config.json` and `statistics.json` in the checkpoint directory.

**Image transforms:**

- Optional letterbox padding to square.
- Resize shortest edge, center crop by `crop_fraction`, resize to `image_target_size`.
- Training: random crop position sampled once per sample, shared across views/timesteps.
- VLM encoding via Qwen3-VL processor.

**LIBERO-specific decode transform:**

- Binarizes gripper action: `-np.sign(2.0 * gripper - 1.0)`.

### Implicit or hardcoded properties

| Concept | Status |
| --- | --- |
| Embodiment tag semantics | Partially encoded in naming convention, but tag-to-config mapping is internal |
| Camera ordering | Via `video_modality_keys` from checkpoint config, with alias support |
| Joint group definitions | From `modality.json` (explicit index ranges) |
| Action representation | Encoded in embodiment tag name (e.g., `relative_eef`) |
| Normalization strategy per group | Configurable (min_max, percentile) via checkpoint config |

### Key gaps relative to TransformDescriptor/MappingDescriptor

1. **`modality.json` is the closest to a MappingDescriptor** -- it declares semantic array slicing and camera key remapping. But it is GR00T-specific and lives only in the dataset, not in model metadata.
2. **No cross-model portability** -- the modality config, embodiment tags, and normalization infrastructure are tightly coupled to the GR00T architecture.
3. **Embodiment tags are learned, not declared** -- the model learns per-embodiment projectors, making the "mapping" a trained network rather than a declarative configuration.
4. **No sign flip or explicit joint reordering declarations** -- these are handled implicitly by the training data and modality config index ranges.
5. **Per-horizon-timestep normalization stats** are a pattern not covered by our TransformDescriptor model (which assumes a single set of stats per feature).

**Sources**: [NVIDIA/Isaac-GR00T modality.json](https://github.com/NVIDIA/Isaac-GR00T/blob/main/examples/SO100/modality.json), [GR00T embodiment_tags.py](https://github.com/NVIDIA/Isaac-GR00T/blob/main/gr00t/data/embodiment_tags.py), [LeRobot processor_groot.py](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/groot/processor_groot.py), [GR00T N1.7 HF blog](https://huggingface.co/blog/nvidia/gr00t-n1-7), [GR00T data preparation](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/data_preparation.md)

---

## Cross-Community Comparison

### Normalization strategies

| Community | Strategies | Stats format | Where stored | Strategy selection |
| --- | --- | --- | --- | --- |
| OpenPI | z-score, quantile (q01/q99) | `norm_stats.json` (Pydantic) | Checkpoint assets dir | `use_quantile_norm: bool` on DataConfig |
| LeRobot | identity, mean_std, min_max, quantiles (q01/q99), quantile10 (q10/q90) | `stats.json` (numpy arrays) + processor config | Dataset metadata + model processor | `norm_map: dict[FeatureType, NormalizationMode]` |
| vLLM-Omni (pi0) | Loaded from checkpoint (matches LeRobot schema) | `config.json` `norm_stats` field | Model checkpoint | Not configurable -- determined by checkpoint |
| vLLM-Omni (GR00T) | min_max, mean_std, sin_cos encoding | `statistics.json` per embodiment | Model checkpoint | Per-joint-group via `ModalityConfig` |
| SGLang | Fixed mapping to [-1, 1] + 256-bin discretization | None | Hardcoded | None (fixed pipeline) |
| LEAPP | Running mean/variance (z-score) | Baked into ONNX graph | ONNX constant nodes | Not configurable post-export |
| GR00T | min_max with optional percentile, per-timestep | `statistics.json` + `relative_stats.json` | Checkpoint + dataset | `use_percentiles: bool`, per-group config |

### Mapping concepts

| Concept | OpenPI | LeRobot | vLLM-Omni | SGLang | LEAPP | Rosetta | knownrobot | GR00T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Camera remapping | Hardcoded per-robot | Via `RenameObservationsStep` | `image_key_map` config | Hardcoded | N/A | `channel.topic` binding | sensor name only | `video_key` in modality.json |
| Joint selection/ordering | Hardcoded per-robot | Not in generic pipeline | Not in pi0; per-group in GR00T | N/A | `--joint_config` at export | `select: [position.j1, ...]` | Not covered | modality.json index ranges |
| Sign flips | Hardcoded (Aloha only) | Not covered | Not in pi0 | N/A | N/A | Not covered | Not covered | Not covered |
| Gripper conversion | Hardcoded (Aloha only) | Not covered | Not in pi0 | N/A | N/A | Not covered | Not covered | LIBERO binarize only |
| Action dim slicing | Hardcoded per-robot | Not in generic pipeline | Not in pi0 | N/A | Baked into ONNX | `select` field | action_shape only | modality.json index ranges |
| Value scaling | Via normalization | Via normalization | Via normalization | Hardcoded | Baked into ONNX | Not covered | Not covered | Via normalization |
| Relative/absolute action | `DeltaActions` transform | `RelativeActionsProcessorStep` | GR00T ActionConfig | N/A | N/A | Not covered | Not covered | Per-group `ActionRepresentation` |

### What is declarative vs. hardcoded

| Community | Declarative | Hardcoded |
| --- | --- | --- |
| **OpenPI** | Normalization strategy (bool), norm_stats (JSON file), transform pipeline order | Camera mapping, sign flips, gripper conversion, action slicing, state concatenation |
| **LeRobot** | Normalization mode mapping, stats, processor pipeline composition, feature rename config | Feature type classification heuristic |
| **vLLM-Omni** | norm_stats (in config.json), image_key_map, image_feature_keys | Image normalization range, max cameras, prompt suffix |
| **SGLang** | Pipeline config fields (image_size, action_dim, etc.) | All preprocessing logic, discretization bins |
| **LEAPP** | Tensor semantics (kind, element_names, extra) | Normalizer parameters (baked into ONNX) |
| **Rosetta** | Topic-feature mapping, joint selection, image resize | No normalization, no value transforms |
| **knownrobot** | Shapes, hardware, compatibility evidence | No transforms, no mappings |
| **GR00T** | modality.json (array slicing, video keys), embodiment tags, normalization config, per-group action representation | Camera ordering aliases, LIBERO-specific gripper logic |

### Key findings

1. **Normalization formulas have converged** -- z-score `(x - mean) / std` and min/max/quantile `2 * (x - ref) / range - 1` are universal. The difference is in strategy selection and stats format.

2. **Mapping is the underdeveloped area** -- every community handles camera remapping, sign flips, gripper conversion, and joint reordering differently, and almost always in hardcoded Python code. GR00T's `modality.json` and Rosetta's YAML contract are the only declarative approaches, and both are narrowly scoped.

3. **No community declares the full transform+mapping pipeline declaratively** -- even LeRobot (the most structured) requires Python code for robot-specific adaptation beyond normalization.

4. **Stats format is near-convergent** -- OpenPI's `norm_stats.json` and LeRobot's `stats.json` carry the same core fields (mean, std, min, max, q01, q99). The difference is packaging (Pydantic vs numpy arrays) and additional quantiles in LeRobot (q10, q50, q90).

5. **LEAPP's tensor semantics are unique** -- the only system annotating model I/O with per-dimension semantic metadata (kind, element_names, frame, units). But they annotate outputs only, not inputs, and they describe what the dimensions mean, not how to transform them.

6. **GR00T has the richest mapping model** -- `modality.json` + embodiment tags + per-group normalization + relative/absolute action handling. But it is tightly coupled to the GR00T architecture and not portable.

7. **Our TransformDescriptor model is well-aligned with community practice** -- the type/strategy/parameters/stats_reference structure captures what every community does. The main gap is that no community has made this metadata portable across models.

8. **Our MappingDescriptor model addresses the biggest gap** -- camera remapping, sign flips, gripper conversion, and joint reordering are universally needed but nowhere declaratively described in a portable format. This is where a new spec could add the most value.
