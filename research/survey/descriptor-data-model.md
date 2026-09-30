# Descriptor Data Model

**Date**: 2026-09-21
**Purpose**: Define the abstract descriptor types needed for the Physical AI stack (inference and training), their composition relationships, and the research agenda for each. Properties will be assembled from existing community schemas, not invented.

**Usage model**: Descriptors are static metadata attached to artifacts (checkpoints, server configs, robot configs, datasets). An external entity (UI, orchestrator, AI agent) reads descriptors from multiple artifacts and validates compatibility before connecting them. This is pre-deployment validation — not runtime negotiation or data-plane enforcement.

---

## Design Principles

1. **Describe, don't prescribe.** Descriptors declare what an artifact expects/provides. They don't enforce a single canonical format.

2. **Reuse community schemas.** Where a community schema (OpenUSD, URDF, LeRobot info.json, etc.) already has well-defined semantics for a concept, reference or embed it — don't reinvent. A single community schema may serve multiple descriptor types.

3. **Composability.** Complex descriptors compose from simpler ones. An ActionChunkDescriptor references an ActionDescriptor; a CheckpointDescriptor references ObservationDescriptor and ActionChunkDescriptor.

4. **Explicit state.** Data transforms (normalization, preprocessing, coordinate frame changes) alter what a descriptor describes. Each boundary's descriptor reflects the state of the data *at that point* — before or after transforms.

5. **Compatibility is external.** Descriptors don't negotiate with each other. An external validator compares descriptors from two artifacts and reports whether they're compatible, partially compatible (adapter needed), or incompatible.

---

## Descriptor Types

### Leaf Descriptors (no further decomposition)

#### ActionDescriptor

Describes the semantic meaning of a single action vector — what each dimension represents.

```
ActionDescriptor
├── representation_type       — joint_position, joint_velocity, ee_delta, ee_absolute, latent, ...
├── normalization_state       — raw | normalized | baked_in
├── dimensions[]
│   ├── name                  — joint/axis name (e.g., "shoulder_pan", "gripper")
│   ├── kind                  — joint_position, joint_velocity, ee_delta, gripper_binary, ...
│   ├── unit                  — rad, m, Nm, normalized, ...
│   ├── reference_frame       — base_link, world, ee_frame, ...
│   ├── is_relative           — absolute | delta | relative
│   ├── rotation_format       — euler_rpy, rot6d, rotvec, quaternion (when applicable)
│   ├── range                 — [min, max] value bounds
│   ├── subsystem             — left_arm, right_arm, gripper, torso, ...
│   └── description           — human-readable (optional)
└── sign_convention           — per-dimension sign mask (e.g., [1, -1, -1, 1, ...])
```

**Sources**: LEAPP `TensorSemantics` (kind, element_names, extra.frame, extra.units — only per-dimension semantics); GR00T `ActionConfig` (representation, type, format enums); Gymnasium `Box` (low/high bounds, dtype); URDF `<limit>` (joint limits); OpenPI adapter classes (sign flip masks — hardcoded, not declarative).

**Gaps**: Per-dimension range bounds absent from all ML communities (only URDF/Gymnasium have bounds). Sign flips universally hardcoded. Gripper convention (binary vs. continuous, open-high vs. open-low) everywhere implicit. Joint-to-action-index mapping absent from all formats.

**Composition**: Referenced by ActionChunkDescriptor, ProprioceptionDescriptor (similar structure), MappingDescriptor.

#### CameraDescriptor

Describes a single camera's properties and role.

```
CameraDescriptor
├── role                      — wrist | world | head | ego | overhead (NEW — nowhere formalized)
├── resolution                — width × height
├── color_encoding            — RGB, BGR, grayscale, depth, RGBD
├── stereo                    — mono | stereo_left | stereo_right
├── extrinsics                — mounting_link, relative_pose (6-DOF)
├── intrinsics                — K matrix, focal_length, aperture
├── distortion                — model (plumb_bob, equidistant, ...), coefficients
├── fps                       — capture frame rate
├── clipping_planes           — near, far
├── preprocessing_state       — raw | resized (target H×W) | normalized (mean, std)
└── validity_mask             — whether masked for multi-camera padding
```

**Sources**: ROS 2 `CameraInfo` (full K matrix, distortion model, stereo R/P, frame_id + TF — gold standard for calibration); OpenUSD `UsdGeomCamera` (focal, aperture, stereoRole, clippingRange, transform stack); LeRobot `info.json` (shape, pix_fmt, fps, codec); SGLang (image_size, image_normalization_mean/std); URDF/SDF (origin, update_rate, k1-k3/p1-p2).

**Gaps**: Camera role is universally implicit — every community encodes it in key names (`base_0_rgb`, `cam_high`, `wrist_image`) but none has a controlled vocabulary. A small enum {wrist, world, head, ego, overhead} with a mounting-link reference is low-hanging fruit.

**Composition**: Referenced by ObservationDescriptor (cameras[]).

#### TransformDescriptor

Describes a data transformation applied at a boundary (normalization, preprocessing, coordinate conversion).

```
TransformDescriptor
├── type                      — normalization | denormalization | image_preprocessing | discretization | coordinate_transform
├── strategy                  — identity | mean_std | min_max | quantile | quantile10 | sin_cos | discretize | baked_in
├── parameters
│   ├── eps                   — numerical stability (default 1e-6)
│   ├── clip_outliers         — bool
│   ├── target_resolution     — [H, W] for image resize
│   └── output_range          — [-1, 1] | centered | unchanged
├── stats                     — inline or referenced
│   ├── mean, std             — per-dimension arrays, shape (D,) or (C,1,1)
│   ├── min, max              — per-dimension arrays
│   ├── q01, q99              — quantile bounds (OpenPI + LeRobot convergent)
│   ├── q10, q50, q90         — extended quantiles (LeRobot only)
│   └── count                 — sample count (LeRobot only)
└── stats_reference           — path/URI to stats file, "inline", or "baked_in"
```

**Sources**: OpenPI `norm_stats.json` (mean/std/q01/q99 per feature, Pydantic); LeRobot `stats.json` (same + min/max/q10/q50/q90/count, numpy); LeRobot `NormalizerProcessorStep` (5 modes: identity, mean_std, min_max, quantile, quantile10); LEAPP (baked into ONNX graph); GR00T (per-horizon-step stats for relative actions).

**Convergence**: Z-score `(x - mean) / (std + eps)` and quantile `2 * (x - q01) / (q99 - q01 + eps) - 1` are universal formulas. Stats formats are near-identical between OpenPI and LeRobot — a unified schema (mean, std, min, max, q01, q99 with optional extended quantiles) formalizes what already exists.

**Gaps**: Strategy selection mechanism differs per community. Whether stats are external files or baked into ONNX is undeclared. Per-horizon-step stats (GR00T) are proprietary.

**Composition**: Referenced by CheckpointDescriptor (input/output transforms), InferenceServerDescriptor (server transforms), MappingDescriptor (adapter transforms).

### Composite Descriptors

#### ActionChunkDescriptor

Describes a temporal sequence of actions — the actual output of a VLA model.

```
ActionChunkDescriptor
├── action: ActionDescriptor  — per-dimension semantics of each step
├── action_dim                — dimensionality of action vector
├── padded_action_dim         — zero-padded dim (e.g., Pi0 pads to 32)
├── horizon                   — number of steps in chunk
├── control_frequency_hz      — intended execution rate
└── data_type                 — float32, float16, bfloat16
```

**Sources**: OpenPI config (`action_horizon`, `action_dim`); SGLang metadata (`action_dim`, `padded_action_dim`, `action_horizon`, `dtype`); LeRobot config (`chunk_size`, `fps`); LEAPP `TemporalAxis` (`period_ms`); Cosmos 3 (`action_fps`).

**Gaps**: Control frequency rarely declared alongside action chunk. Padding semantics (which dims are padding) absent.

**Composition**: Referenced by CheckpointDescriptor, InferenceServerDescriptor, RobotInterfaceDescriptor, MappingDescriptor.

#### ObservationDescriptor

Describes the full observation space — all modalities a component expects or provides.

```
ObservationDescriptor
├── cameras[]: CameraDescriptor           — ordered list with roles
├── camera_order[]                        — canonical ordering for model input
├── proprioception: ProprioceptionDescriptor
├── prompt                                — language conditioning (present/absent, format, tokenized/raw)
└── additional_modalities[]               — IMU, F/T, tactile, lidar, audio, ...
```

**Sources**: OpenPI `Observation` dataclass (closest to a composite — images dict + state + tokenized_prompt); GR00T `ModalityConfig` (temporal sampling, normalization strategy per modality); LeRobot `info.json` (features dict with shapes/types); Gymnasium `observation_space` (Dict of Box/MultiBinary); SGLang (`camera_order` metadata, `image_keys`).

**Gaps**: No community has a typed composite observation descriptor. Camera ordering matters for model input but is implicit. Language conditioning format (tokenized vs. raw, max length) is undeclared.

**Composition**: Referenced by CheckpointDescriptor, InferenceServerDescriptor, RobotInterfaceDescriptor, MappingDescriptor.

#### ProprioceptionDescriptor

Describes the proprioceptive state vector — joint positions, velocities, efforts, gripper state.

```
ProprioceptionDescriptor
├── dimensions[]              — same structure as ActionDescriptor.dimensions (name, kind, unit, frame, range)
├── includes_velocity         — whether velocity channels are present
├── includes_effort           — whether effort/torque channels are present
└── normalization_state       — raw | normalized (strategy + stats ref) | baked_in
```

**Sources**: LeRobot `info.json` (feature names dict); ROS 2 `JointState` (name[], position[], velocity[], effort[]); GR00T `modality_keys` (group-level names); LEAPP `TensorSemantics` (for output dimension semantics).

**Gaps**: Units, representation type, and reference frame for proprioceptive dimensions are absent everywhere. No community separates position/velocity/effort channels declaratively.

**Composition**: Referenced by ObservationDescriptor.

### Artifact Descriptors (attached to deployable artifacts)

#### CheckpointDescriptor

Describes what a trained model checkpoint expects and produces — the model's contract.

```
CheckpointDescriptor
├── expects_input: ObservationDescriptor     — what the model needs (post-normalization)
├── produces_output: ActionChunkDescriptor   — what the model outputs (pre-denormalization)
├── input_transform: TransformDescriptor     — normalization applied before inference
├── output_transform: TransformDescriptor    — denormalization applied after inference
├── normalization_mapping                    — per-modality strategy (IDENTITY, MEAN_STD, QUANTILE)
├── architecture                             — model family (pi0, pi0.5, octo, gr00t, ACT, diffusion, ...)
├── base_model_id                            — HF model ID or similar
├── fine_tuned_from                          — parent checkpoint reference
├── n_obs_steps                              — observation history length
└── input_features / output_features         — dict of feature name → {shape, type}
```

**Sources**: LeRobot policy config (normalization_mapping, features, chunk_size, n_obs_steps — richest file-first metadata); OpenPI config.json (action_horizon, action_dim, max_state_dim, normalization); OpenPI `norm_stats.json` (per-feature stats); HF model cards (`pipeline_tag: robotics`, base_model); LEAPP ONNX bundle (TensorSemantics — only per-dimension action semantics, but lost on export); SGLang (derives metadata from checkpoint at load time).

**Code-first vs file-first split**: LeRobot, GR00T, ONNX/Triton store metadata in files alongside weights. OpenPI and vLLM-Omni encode metadata in Python config classes. SGLang is hybrid — reads checkpoint files, then exposes via API.

**Gaps**: Action semantics (what each output dimension means) absent from all checkpoints except LEAPP. Embodiment compatibility nowhere declared as metadata — determined by which `*Inputs` adapter classes exist in server source code. Whether normalization is baked-in vs. external is undeclared.

**Composition**: Top-level artifact descriptor. Embedded in or alongside model files.

#### InferenceServerDescriptor

Describes what a running inference server accepts and provides — the server's contract as seen by clients.

```
InferenceServerDescriptor
├── accepts_input: ObservationDescriptor     — what clients should send
│   ├── image_keys[]                         — camera slot names
│   ├── image_size                           — [H, W]
│   └── state_dim                            — proprioception dimensions
├── provides_output: ActionChunkDescriptor   — what clients will receive
│   ├── action_type                          — continuous | discretized
│   ├── action_dim / padded_action_dim       — actual vs. zero-padded
│   ├── action_horizon                       — steps per chunk
│   └── dtype                                — float32, ...
├── checkpoint: CheckpointDescriptor         — loaded model
├── server_transforms[]: TransformDescriptor — preprocessing the server applies
├── supported_embodiments[]                  — which robots this server is configured for
├── capabilities[]                           — API surfaces (realtime_websocket, batch, openpi_websocket, ...)
└── runtime_config                           — parallelism, autocast, cuda_graph
```

**Sources**: SGLang `/v1/actions/metadata` (only server introspection API — returns input/output structure, capabilities, runtime config); OpenPI (no introspection — config in Python); vLLM-Omni (no introspection).

**Gaps**: Action semantics, normalization details, supported embodiments, base model lineage, and transform descriptions all absent from SGLang's metadata endpoint. No other server has any introspection API.

**Composition**: References CheckpointDescriptor. The server transforms explain the difference between the checkpoint's expects_input and the server's accepts_input.

#### EmbodimentDescriptor

Describes a physical or simulated robot's mechanical and sensor properties.

```
EmbodimentDescriptor
├── format_reference          — URI to URDF, MJCF, or OpenUSD file (authoritative source)
├── joints[]
│   ├── name                  — joint name (format-dependent ordering!)
│   ├── type                  — revolute | prismatic | continuous | fixed | ball
│   ├── limits                — lower, upper, effort, velocity
│   ├── axis                  — rotation/translation axis
│   ├── damping, friction     — dynamics properties
│   └── ml_action_index       — mapping to policy action vector dim (NEW — nowhere exists)
├── actuators[]
│   ├── control_mode          — position | velocity | effort
│   ├── gain, bias            — controller parameters
│   └── actuator_model        — implicit | DC_motor | ideal_PD | MLP
├── cameras[]: CameraDescriptor             — with roles (NEW)
├── additional_sensors[]      — IMU, F/T, tactile (MJCF has 33 sensor types)
├── reference_frames          — base, ee, tcp, world
└── named_configurations      — HOME, READY, STOW joint positions (NEW — nowhere exists)
```

**Sources**: URDF (joints, links, geometry, inertia, transmissions — universally adopted but ML-incomplete); MJCF (7 actuator types, 33 sensor types, damping/stiffness — richest actuator model); OpenUSD (articulation prims, PhysicsDriveAPI, camera geometry — richest scene representation); ros2_control (command_interface/state_interface declarations); Isaac Lab `ArticulationCfg` + `ActionsCfg` (actuator models: implicit/DC/ideal PD/MLP).

**Joint ordering problem**: URDF uses tree traversal order (implementation-dependent), MJCF uses XML definition order, OpenUSD uses articulation solver internal ordering. All three assign **different orderings for the same robot** (confirmed: Isaac Lab issue #7750). No format provides a canonical, simulator-independent mapping from joint name to policy action-vector index.

**URDF-for-ML gap**: URDF has joint names/types/limits/axes and link geometry/inertia. ML additionally needs: joint-to-action-index mapping, per-dimension semantics, camera sensors with roles, action/observation space declarations, normalization metadata, named configurations (HOME/READY), and control frequency.

**Composition**: Referenced by RobotInterfaceDescriptor. The authoritative source is an existing format (URDF, MJCF, OpenUSD) — this descriptor adds ML-specific annotations that those formats lack.

#### RobotInterfaceDescriptor

Describes what a robot (physical or simulated) accepts as commands and provides as sensor data — the robot's contract as seen by clients.

```
RobotInterfaceDescriptor
├── accepts_input: ActionChunkDescriptor     — commands the robot accepts
├── provides_output: ObservationDescriptor   — sensor data the robot provides
├── embodiment: EmbodimentDescriptor         — physical/simulated robot properties
├── control_frequency_hz                     — expected command rate
└── transport                                — ROS 2 topics | WebSocket | shared_memory (optional)
```

**Sources**: ros2_control (command_interface + state_interface declarations per joint — closest to accepted commands); Gymnasium `action_space`/`observation_space` (shape, bounds, dtype — richest space declarations); ROS 2 topic types + TF tree (discoverable at runtime); Isaac Lab env config (action/observation space declarations).

**Gaps**: No community bridges robot description (URDF joints) to ML action semantics (which joints map to which action dimensions, in what representation, with what ordering). Normalization expectations (raw/normalized/denormalized) absent everywhere. No community links runtime interface to description file.

**Composition**: References EmbodimentDescriptor.

#### MappingDescriptor

Describes how to bridge mismatches between an InferenceServer's interface and a Robot's interface — the adapter's configuration. **This is the novel contribution** — no existing community or spec covers the full adapter declaratively.

```
MappingDescriptor
├── from_action: ActionChunkDescriptor       — server output
├── to_action: ActionChunkDescriptor         — robot input
├── from_observation: ObservationDescriptor   — robot output
├── to_observation: ObservationDescriptor     — server input
└── transforms[]                             — mapping rules:
    ├── camera_key_remapping                 — server name → robot name (e.g., "base_0_rgb" → "cam_high")
    ├── joint_selection_and_ordering         — index permutation + which joints to include
    ├── action_dim_slicing                   — index ranges for subgroups (arm, gripper)
    ├── sign_flip_mask[]                     — per-dimension sign (e.g., [1, -1, -1, 1, ...])
    ├── gripper_conversion                   — convention mapping (linear↔angular, binary↔continuous, open-high↔open-low)
    ├── relative_absolute_conversion         — delta↔absolute action conversion
    ├── image_format_conversion              — CHW↔HWC, float↔uint8, BGR↔RGB
    ├── coordinate_transforms               — frame changes
    └── value_scaling                        — unit conversions (via TransformDescriptor)
```

**Sources**: OpenPI adapter classes (camera remapping, sign flips, gripper conversion — all hardcoded in Python per robot); vLLM-Omni `image_key_map` config (declarative camera remapping); GR00T `modality.json` (video keys, index ranges — partially declarative); Rosetta YAML (topic→feature binding, joint `select` ordering, `align` temporal strategy, `apply` image resize — transport-level only); LeRobot `RenameObservationsStep` (camera key remapping); LEAPP `--joint_config` (joint selection).

**What exists declaratively**: Camera key remapping (vLLM-Omni, GR00T, Rosetta). Joint selection and ordering (Rosetta `select`, GR00T index ranges). Action dim slicing (GR00T `modality.json`). Relative↔absolute conversion (GR00T `ActionRepresentation` enum).

**What is universally hardcoded**: Sign flip masks (OpenPI Aloha: `[1,-1,-1,1,1,1,1,1,-1,-1,1,1,1,1]`). Gripper conversion formulas (OpenPI Aloha: linear→angular via arcsin with hardware constants). Image format conversion (OpenPI: CHW→HWC, float→uint8). These are the novel contribution of a MappingDescriptor spec.

**Design principle from Rosetta**: Single-source-of-truth — same YAML for record, convert, and deploy. A MappingDescriptor should similarly be usable across the pipeline.

**Composition**: References ActionChunkDescriptor and ObservationDescriptor from both sides.

### Training & Dataset Descriptors

These descriptor types cover the data collection, curation, and training side of the pipeline. They reuse the same leaf descriptors (ActionDescriptor, CameraDescriptor, TransformDescriptor, etc.) as the inference stack.

#### DatasetDescriptor

Describes a dataset's structure, contents, and provenance — the dataset's contract for consumers (training pipelines, evaluation tools, dataset browsers).

```
DatasetDescriptor
├── features                                 — dict of name → {shape, dtype, modality} (dot-notation keys)
├── observation: ObservationDescriptor       — observation space across episodes
├── action: ActionChunkDescriptor            — action space across episodes
├── stats                                    — per-feature mean/std/min/max/q01/q99
├── robot_type                               — string identifier (needs controlled vocabulary)
├── embodiment: EmbodimentDescriptor         — robot description reference (URDF/MJCF/USD)
├── episode_count                            — number of episodes
├── total_frames                             — total timesteps across episodes
├── fps                                      — recording frame rate
├── tasks[]: TaskDescriptor                  — NL task labels + distribution
├── collection_method                        — teleoperation | autonomous | simulation | mixed
├── environment                              — lab, kitchen, warehouse, simulated, ...
├── transforms_applied[]: TransformDescriptor — any preprocessing baked into the dataset
├── splits                                   — train/val/test split specification
├── format_version                           — lerobot_v3 | rlds | hdf5 | webdataset | ...
├── license                                  — data license
└── provenance                               — source dataset(s), conversion lineage (Croissant PROV-O)
```

**Sources**: LeRobot v3 `info.json` (features dict with dot-notation keys, fps, episode_count, total_frames, splits, robot_type, codebase_version — de facto standard, 16K+ datasets); LeRobot `stats.json` (per-feature mean/std/min/max/q01/q99 — near-identical to OpenPI `norm_stats.json`); LeRobot `tasks.jsonl` (NL task labels per episode); HF dataset cards (license, task_categories, size_categories); Croissant v1.1 (PROV-O provenance, Split, RecordSet — governance layer, no robotics primitives); DROID (3 simultaneous action representations — avoids information loss at recording).

**Gaps**: No dataset links to URDF/MJCF/USD embodiment description. Action semantics (what each dim means) absent from all datasets. Camera roles implicit everywhere. Collection method always in docs, never structured. Dataset compatibility with models nowhere declared. RLDS/OXE solved cross-embodiment by normalizing to 7-DOF EE delta — original semantics lost.

#### EpisodeDescriptor

Describes per-episode metadata within a dataset.

```
EpisodeDescriptor
├── task                                     — NL instruction or task label
├── task_index                               — reference into dataset's task list
├── success                                  — bool or success metric
├── length                                   — number of timesteps
├── embodiment_id                            — robot used (if multi-robot dataset)
├── environment_id                           — scene/environment identifier
├── scene_metadata                           — scene properties (DROID-style)
├── formal_task_spec                         — structured task definition (LIBERO BDDL)
├── collection_timestamp                     — when recorded (ISO 8601)
└── annotations                              — quality labels, failure modes, subset tags
```

**Sources**: LeRobot (task_index → `tasks.jsonl`, episode length via parquet files); DROID (scene metadata, multi-camera); LIBERO (BDDL formal task spec, success labels — only community with structured success metrics); RLDS (episode-level features).

**Gaps**: Success labels are rare (only LIBERO has them reliably). Operator identity, collection timestamp, and quality annotations absent from all. Multi-robot episode metadata (which robot) not formalized.

#### TrainingConfigDescriptor

Describes a training pipeline's configuration — what data it consumes, how it transforms it, and what checkpoint it produces. This is the training-side counterpart to InferenceServerDescriptor.

```
TrainingConfigDescriptor
├── input_dataset                            — HF repo_id or path + DatasetDescriptor ref
├── input_features / output_features         — dict of feature name → {shape, type, modality}
├── data_transforms[]: TransformDescriptor   — ordered pipeline stages (repack → data → normalize → model)
├── normalization
│   ├── strategy_mapping                     — per-modality: IDENTITY | MEAN_STD | QUANTILE | MIN_MAX
│   └── stats_reference                      — path to stats file
├── action_config
│   ├── representation                       — joint_position | ee_delta | ee_absolute
│   ├── type                                 — continuous | discrete
│   ├── chunk_size / action_horizon          — action chunk length
│   ├── max_action_dim                       — zero-padded dimension
│   ├── relative_actions                     — bool (convert absolute → relative)
│   └── per_group_config[]                   — index ranges + representation per subsystem (GR00T)
├── observation_config
│   ├── camera_keys[]                        — which cameras from dataset
│   ├── image_transforms[]                   — resize, crop, augmentation
│   ├── n_obs_steps                          — observation history length
│   └── state_keys[]                         — which proprioception features
├── produces_checkpoint: CheckpointDescriptor — output checkpoint contract
├── architecture                             — model family + variant (Pi0, ACT, Diffusion, ...)
├── fine_tuned_from                          — base checkpoint reference
└── target_embodiment                        — intended deployment robot (GR00T EmbodimentTag)
```

**Sources**: LeRobot policy configs (normalization_mapping, input/output features, chunk_size, n_obs_steps — per-architecture: Pi0Config, DiffusionConfig, ACTConfig); OpenPI TrainConfig (explicit 4-stage data transform pipeline: repack → data → normalize → model); GR00T (ActionRepresentation + ActionType per group, EmbodimentTag — only cross-artifact link between training and deployment); Isaac Lab (env + agent + policy config, actuator models).

**Key finding**: Training configs are richer than checkpoint metadata — the full pipeline (transforms, normalization strategy, camera selection, action representation) is specified during training but **lost at checkpoint save time**. This forces inference servers to independently reconstruct what training did. GR00T's `EmbodimentTag` is the only community that links training config to deployment adapter.

**Gaps**: No community records the full data transform pipeline in the checkpoint. Training-to-checkpoint lineage (which sign flips, gripper conversions were applied) is lost. Dataset compatibility (which datasets work with which training configs) is discovered at runtime via shape mismatch errors.

#### DataConversionDescriptor

Describes how to convert between dataset formats or between a recording and a dataset — the training-side counterpart to MappingDescriptor.

```
DataConversionDescriptor
├── source_format                            — rlds | hdf5 | rosbag | raw_files | ...
├── target_format                            — lerobot_v3 | rlds | hdf5 | ...
├── feature_mapping                          — source feature name → target feature name
├── camera_mapping                           — source camera key → target camera key + role
├── action_mapping                           — source action dims → target action dims
│   ├── joint_selection                      — which joints to include
│   ├── joint_reordering                     — index permutation
│   ├── sign_flips                           — per-dimension sign mask
│   └── gripper_conversion                   — convention mapping
├── temporal_config
│   ├── source_fps / target_fps              — frame rate conversion
│   └── alignment_strategy                   — hold | interpolate (Rosetta `align`)
├── image_transforms[]                       — resize, crop (Rosetta `apply`)
├── stats_computation                        — automatic during conversion
├── transforms[]: TransformDescriptor        — normalization applied during conversion
└── provenance_chain                         — ordered list of conversions applied
```

**Sources**: Rosetta YAML (feature_mapping via topic→feature binding, joint `select` for ordering, `align` for temporal strategy, `apply` for image resize — only declarative conversion config, single-source-of-truth design); LeRobot converters (`convert_dataset_v1_to_v2` — format migration, stats computation); OXE dataset transforms (70+ per-dataset Python transforms for action relabeling, camera remapping, observation filtering — demonstrates scale of the problem); fog_rtx (format bridges between RLDS/HDF5/LeRobot).

**Key finding**: OXE's 70+ per-dataset transforms are the strongest evidence for a declarative DataConversionDescriptor — each dataset has hardcoded Python for action space unification (typically projecting to 7-DOF EE delta), camera key remapping, and observation filtering. A declarative format would eliminate this redundant reimplementation. Rosetta's single-source-of-truth principle (same YAML for record, convert, deploy) is the right design pattern.

**Gaps**: All conversion logic is hardcoded Python — nowhere declarative except Rosetta (transport-level only). Action space unification metadata absent. Provenance chain (multi-format conversion history) not tracked. Semantic correctness of conversion not verified — discovered at training time via loss divergence.

---

## Composition Map

How descriptors reference each other across the full stack:

```
TRAINING SIDE                                    INFERENCE SIDE

DatasetDescriptor                                 CheckpointDescriptor
├── observation: ObservationDescriptor            ├── expects_input: ObservationDescriptor (model-native)
├── action: ActionChunkDescriptor                 ├── produces_output: ActionChunkDescriptor (model-native)
├── embodiment: EmbodimentDescriptor              ├── input_transform: TransformDescriptor
├── episodes[]: EpisodeDescriptor                 └── output_transform: TransformDescriptor
└── transforms_applied[]: TransformDescriptor               │
          │                                       InferenceServerDescriptor
          │  ← DataConversionDescriptor →         ├── accepts_input: ObservationDescriptor (client-facing)
          │    (format + feature mapping)          ├── provides_output: ActionChunkDescriptor (client-facing)
          │                                       ├── checkpoint: CheckpointDescriptor
TrainingConfigDescriptor                          └── server_transforms[]: TransformDescriptor
├── input_dataset: DatasetDescriptor                        │
├── data_transforms[]: TransformDescriptor                  │  ← MappingDescriptor bridges this gap →
├── produces_checkpoint ─────────────────────►              │
└── target_embodiment: EmbodimentDescriptor       RobotInterfaceDescriptor
                                                  ├── accepts_input: ActionChunkDescriptor (robot-native)
                                                  ├── provides_output: ObservationDescriptor (robot-native)
                                                  └── embodiment: EmbodimentDescriptor
```

**Inference-side validation** — an external validator reads artifact descriptors and:

1. Checks if InferenceServer output is directly compatible with Robot input (no adapter needed)
2. If not, checks if a MappingDescriptor can bridge the gap (adapter configurable)
3. If not, reports specific incompatibilities (missing camera, wrong joint count, unsupported action type)

**Training-side validation** — same pattern:

1. Checks if Dataset action/observation spaces are compatible with TrainingConfig expectations
2. If not, checks if a DataConversionDescriptor can bridge the gap
3. Checks if the produced CheckpointDescriptor is compatible with the target EmbodimentDescriptor

---

## Survey Findings

Per-descriptor-type surveys completed 2026-09-21. Full results in companion files:

- [descriptor-survey-action.md](descriptor-survey-action.md) — ActionDescriptor + ActionChunkDescriptor
- [descriptor-survey-observation.md](descriptor-survey-observation.md) — ObservationDescriptor + CameraDescriptor + ProprioceptionDescriptor
- [descriptor-survey-checkpoint-server.md](descriptor-survey-checkpoint-server.md) — CheckpointDescriptor + InferenceServerDescriptor
- [descriptor-survey-embodiment-robot.md](descriptor-survey-embodiment-robot.md) — EmbodimentDescriptor + RobotInterfaceDescriptor
- [descriptor-survey-transform-mapping.md](descriptor-survey-transform-mapping.md) — TransformDescriptor + MappingDescriptor
- [descriptor-survey-dataset.md](descriptor-survey-dataset.md) — DatasetDescriptor + EpisodeDescriptor
- [descriptor-survey-training-conversion.md](descriptor-survey-training-conversion.md) — TrainingConfigDescriptor + DataConversionDescriptor

### Cross-Cutting Findings

**1. No community has a complete descriptor for any type.**

Each community covers some properties and leaves others implicit or hardcoded. LEAPP's `TensorSemantics` comes closest for action semantics (kind, element_names, extra), but lacks range bounds and normalization metadata, and is NVIDIA-specific. LeRobot's `info.json` is the richest dataset/checkpoint metadata, but has no action semantics or embodiment reference. No community has a composite observation descriptor.

**2. The MappingDescriptor addresses the biggest structural gap.**

Camera remapping, sign flips, gripper conversion, and joint reordering are needed by every deployment but exist only as hardcoded Python adapter classes — independently reimplemented in OpenPI, vLLM-Omni, and SGLang. GR00T's `modality.json` and Rosetta's YAML contract are the only declarative approaches, both narrowly scoped. This is where a new spec adds the most value.

**3. Joint ordering is the fundamental interoperability problem.**

URDF, MJCF, and OpenUSD each assign different joint orderings for the same robot (confirmed: Isaac Lab [#7750](https://github.com/isaac-sim/IsaacLab/issues/7750)). No community provides a canonical mapping from joint names to policy action-vector indices independent of simulator. Every adapter must solve this for every robot.

**4. Camera role is universally implicit.**

Every community encodes camera role in key/frame names by convention (`base_0_rgb`, `cam_high`, `wrist_image`). No community has an explicit `role` field with controlled vocabulary (wrist, world, head, ego, overhead).

**5. Normalization formulas have converged; packaging hasn't.**

Z-score `(x - mean) / (std + eps)` and quantile/min-max `2 * (x - ref) / (range + eps) - 1` are universal. Stats formats are near-convergent (OpenPI `norm_stats.json` ≈ LeRobot `stats.json`: both carry mean/std/min/max/q01/q99 per feature). The difference is strategy selection mechanisms, stats file naming, and whether stats are external files or baked into ONNX graphs.

**6. SGLang is the only server with introspection.**

`/v1/actions/metadata` returns structured JSON: `input` (image_keys, state_dim), `output` (action_type, action_dim, padded_action_dim, action_horizon, dtype), `capabilities`, and runtime config. This is the closest existing implementation to an InferenceServerDescriptor, but omits action semantics and supported embodiments.

**7. Robot-description-to-action-space mapping is universally absent.**

URDF/MJCF/OpenUSD thoroughly describe joints (names, types, limits, axes). Gymnasium thoroughly describes action spaces (shape, bounds, dtype). Nothing bridges the two: which joints map to which action dimensions, in what representation, with what ordering. This gap is where errors compound.

**8. Embodiment compatibility is nowhere declared as metadata.**

Which robots a checkpoint supports is determined by which `*Inputs` adapter classes exist in server source code, not by metadata on the checkpoint. HF model cards have `pipeline_tag: robotics` but no structured robot/embodiment fields.

**9. Training configs are richer than checkpoint metadata.**

The full data pipeline (transforms, normalization strategy, camera selection, action representation) is specified during training but lost at checkpoint save time. This forces inference servers to independently reconstruct what training did. GR00T's `EmbodimentTag` is the only cross-artifact link between training config, checkpoint, and deployment adapter.

**10. OXE's 70+ per-dataset transforms demonstrate the need for declarative conversion.**

Each OXE dataset has hardcoded Python for action space unification (typically projecting to 7-DOF EE delta), camera key remapping, and observation filtering. A declarative DataConversionDescriptor would eliminate this redundant reimplementation.

**11. LeRobot v3 is the de facto dataset metadata standard.**

With 16K+ datasets and convergent stats format (≈ OpenPI `norm_stats.json`), LeRobot `info.json` is the pragmatic starting point for DatasetDescriptor. Croissant has the governance layer (provenance, licensing) but no robotics primitives.

### Coverage Matrix (Descriptor × Community)

| Descriptor type | Best existing coverage | Second best | Biggest gap |
| --- | --- | --- | --- |
| **ActionDescriptor** | LEAPP `TensorSemantics` (kind, element_names, extra) | GR00T `ActionConfig` (rep, type, format) | Per-dimension range bounds, units absent everywhere |
| **ActionChunkDescriptor** | OpenPI config (`action_horizon`, `action_dim`) | LEAPP `TemporalAxis` (period_ms) | Control frequency rarely declared |
| **CameraDescriptor** | ROS 2 `CameraInfo` (full intrinsics, distortion, TF) | OpenUSD `UsdGeomCamera` (optics, stereo, transform) | Camera role for ML absent everywhere |
| **ProprioceptionDescriptor** | LeRobot `info.json` names + ROS 2 `JointState` | LEAPP `TensorSemantics` (for output dims) | Units, representation type absent |
| **ObservationDescriptor** | GR00T `ModalityConfig` (temporal sampling, norm strategy) | OpenPI `Observation` dataclass | No composite descriptor anywhere |
| **TransformDescriptor** | LeRobot `NormalizerProcessorStep` (5 modes, stats) | OpenPI `Normalize`/`Unnormalize` | Strategy selection differs; no portable format |
| **CheckpointDescriptor** | LeRobot policy config (normalization, features, horizon) | OpenPI config.json | Action semantics, embodiment reference absent |
| **InferenceServerDescriptor** | SGLang `/v1/actions/metadata` | (no close second) | Normalization, supported embodiments not exposed |
| **EmbodimentDescriptor** | URDF + MJCF + OpenUSD (joints, kinematics, physics) | Isaac Lab `ArticulationCfg` + `ActionsCfg` | Joint ordering for ML, camera roles absent |
| **RobotInterfaceDescriptor** | ros2_control (command/state interfaces) | Gymnasium `action_space`/`observation_space` | No bridge to ML action semantics |
| **MappingDescriptor** | GR00T `modality.json` (array slicing, video keys) | Rosetta YAML (topic-feature binding) | Sign flips, gripper conversion, coordinate transforms absent from all |
| **DatasetDescriptor** | LeRobot v3 `info.json` (features, fps, episodes, stats) | HF Cards + Croissant (governance layer) | No link to embodiment, no action semantics |
| **EpisodeDescriptor** | LeRobot (task_index, length via parquet) | DROID (scene metadata), LIBERO (BDDL, success) | Success labels rare, quality annotations absent |
| **TrainingConfigDescriptor** | LeRobot policy configs (normalization, features) | OpenPI TrainConfig (4-stage pipeline) | Training→checkpoint lineage lost at save time |
| **DataConversionDescriptor** | Rosetta YAML (topic→feature, select, align) | OXE transforms (70+ per-dataset Python) | All conversion logic hardcoded; no declarative format |

### Implications for Spec Work

The survey identifies three categories of spec work, not just new data types:

#### Category A: New descriptor types (things nobody defines today)

1. **MappingDescriptor / DataConversionDescriptor** — the adapter logic (camera remapping + joint reordering + sign flips + gripper conversion + coordinate transforms) that every deployment and every dataset conversion reimplements in Python. No existing community or spec covers this. Rosetta's single-source-of-truth pattern (same YAML for record, convert, deploy) is a good design principle. MappingDescriptor and DataConversionDescriptor should share the same mapping vocabulary.

2. **ActionDescriptor with per-dimension semantics** — extending LEAPP `TensorSemantics` (the only per-dimension semantic annotation) with range bounds, normalization state, units, and sign conventions as enforced fields.

3. **Camera role vocabulary** — a small controlled vocabulary {wrist, world, head, ego, overhead} with a mounting-link reference. Low-hanging fruit: every community needs it, none provides it.

4. **Composite ObservationDescriptor** — no community has a typed grouping of cameras + proprioception + language conditioning.

#### Category B: Extensions to existing community specs (filling identified gaps)

1. **LeRobot `info.json`** — adding action semantics, embodiment reference (link to URDF/MJCF/USD), camera roles. Pragmatic starting point: 16K+ datasets, de facto standard.

2. **HF dataset/model cards** — structured robotics metadata fields: `robot_type` as controlled vocabulary, action/observation space descriptions, embodiment compatibility.

3. **URDF/OpenUSD** — joint-to-action-index mapping annotations for ML use. The new metadata that existing robot description formats lack.

4. **SGLang `/v1/actions/metadata`** — adding normalization details, supported embodiments, action semantics to the only existing server introspection API.

5. **Checkpoint metadata** — preserving training lineage (normalization strategy, action representation, sign flips, target embodiment) that is currently lost at save time.

6. **Stats format** — formalizing the near-convergent schema (mean, std, min, max, q01, q99) that OpenPI and LeRobot already share.

#### Category C: Integration/packaging specs (how and where descriptors attach to artifacts)

 1. **OCI artifact metadata** — how to embed descriptors in OCI images containing models (labels, annotations, or sidecar files).

 2. **Model registry integration** — how to register descriptor metadata in KubeFlow ModelRegistry or similar catalogs, enabling compatibility queries.

 3. **HuggingFace surface** — structured model card and dataset card fields that expose descriptor properties for search and validation.

 4. **Server introspection standard** — standardizing what SGLang's metadata endpoint started, so any inference server can expose its descriptor.

 5. **Single-source-of-truth packaging** — how the same descriptor (e.g., a MappingDescriptor) is usable across recording, dataset conversion, and deployment without duplication.

Category C may be the most pragmatic starting point — it's closest to existing infrastructure and forces the question of *which* descriptor properties matter enough to standardize. Defining where metadata lives often clarifies what metadata is needed.
