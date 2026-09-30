# Implementation Comparison — Physical AI Workflow Components

**Date**: 2026-09-18
**Purpose**: Compare concrete implementations of each functional block, documenting exact data formats consumed and produced. Identifies convergence, subtle divergences, and ecosystem momentum.
**Input**: Domain surveys, web research on specific implementations.

---

## 1. Policy Server Implementations

Four serving approaches exist for deploying VLA models to real-time robot control.

| Property | OpenPI Server | vLLM-Omni | LeRobot PolicyServer | Isaac Lab / Isaac ROS Deploy |
| --- | --- | --- | --- | --- |
| **Transport** | WebSocket (port 8000) | HTTP REST `/v1/realtime/robot/openpi` | gRPC (HTTP/2) | In-process (gym env) or ROS 2 (`isaac_ros_deploy`) |
| **Client SDK** | `openpi-client` (Python, minimal deps) | Standard HTTP client | `RobotClient` (Python, gRPC) | Gymnasium API or ROS 2 `InferenceController` |
| **Observation input** | Python dict: `observation/image` (uint8 224x224), `observation/wrist_image` (uint8 224x224), `observation/state` (float[]), `prompt` (string). Field names vary per robot config. | Multi-view images (1-3 views, uint8), proprioceptive state (float[]), language instruction (string). Format matches OpenPI convention. | `RobotObservation` dict, pickle-serialized over gRPC. Client sends raw sensor data; server applies full preprocessor pipeline. Field mapping via `rename_map` in handshake. | GPU tensors via PhysX Direct-GPU API. Observation dict with keys defined per `ObservationCfg`. Joint data via `Articulation.data.joint_pos`. |
| **Action output** | `actions`: float[horizon x action_dim], e.g. `(10, 7)`. No metadata on semantics. | float[horizon x action_dim]. Same shape convention as OpenPI. | Pickle-serialized action chunk (float[actions_per_chunk x action_dim]). Each action timestamped. Chunk merging via configurable aggregation (weighted_average, latest_only, average, conservative). | GPU tensor `action` applied via `env.step(action)`. Shape defined by `action_space` config. |
| **Model loading** | `TrainConfig` Python object + per-robot `*Inputs`/`*Outputs` adapter classes. `norm_stats.json` per dataset. | HF `config.json` + safetensors. Model-specific code in vLLM-Omni model registry. | `policy_class.from_pretrained()` + serialized `policy_preprocessor.json` / `policy_postprocessor.json` pipelines with companion safetensors for normalization stats. Client sends `RemotePolicyConfig` (policy_type, pretrained_path, device) during handshake. | JIT-traced `policy.pt` (PyTorch), ONNX via TensorRT/OnnxRuntime, or direct Python model. |
| **Schema exchange** | None. Client must know exact field names per robot config. Runtime error on mismatch. | None documented. Client must know model's expected observation format. | Partial. Client sends `RemotePolicyConfig` with `lerobot_features` and `rename_map` during `SendPolicyInstructions` handshake. Server loads checkpoint and validates features. No capability discovery from client side. | None. Observation/action spaces defined in env config, not negotiated. |
| **Normalization** | Server-side via `norm_stats.json`. Client sends unnormalized state. | Server-side via `_build_norm_buffers`. Quantile normalization was initially broken (returned `None`). | Server-side via serialized preprocessor/postprocessor pipelines. Normalization strategy declared in `policy_preprocessor.json` (step `normalizer_processor`), statistics in companion safetensors. Denormalization in `policy_postprocessor.json` (`unnormalizer_processor`). | Baked into ONNX graph (Sub/Div/BatchNorm ops) or handled in env wrapper. |

**Sources**: [OpenPI remote inference docs](https://github.com/Physical-Intelligence/openpi/blob/main/docs/remote_inference.md), [vLLM-Omni RFC #6524](https://github.com/vllm-project/vllm-omni/issues/6524), [LeRobot async inference docs](https://huggingface.co/docs/lerobot/en/async), [HF Blog — Async Robot Inference](https://huggingface.co/blog/async-robot-inference), [Isaac Lab policy deployment](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/isaac_lab_tutorials/tutorial_policy_deployment.html), [Isaac ROS Deploy](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_deploy)

### Key observations

OpenPI, vLLM-Omni, and SGLang converge on the same wire format (observation dict with image/state/prompt keys, action chunk output) over WebSocket or HTTP. LeRobot's PolicyServer takes a fundamentally different approach: gRPC transport with pickle-serialized payloads, single-client design, and a client-driven handshake that configures the server at connection time. Isaac Lab operates in-process with GPU tensors, designed for simulation rather than real-time serving.

LeRobot's PolicyServer is the only server where the full pre/post-processing pipeline is **serialized as declarative metadata** (`policy_preprocessor.json` / `policy_postprocessor.json`) rather than hardcoded in server source code. This makes the processing chain checkpoint-portable — any server that understands the pipeline format can reproduce the exact same preprocessing. However, its pickle-based wire format is a known security vulnerability ([CVE-2026-25874](https://github.com/huggingface/lerobot/issues/3047)) and interoperability barrier ([strands-labs/robots#4257](https://github.com/strands-labs/robots/issues/4257)), with a replacement ([PR #3048](https://github.com/huggingface/lerobot/pull/3048) — safetensors + JSON payload encoding) open but unmerged for ~7 months, now on the [v0.7.0 roadmap](https://github.com/huggingface/lerobot/issues/3832).

---

## 2. Robotic Policy Implementations (Model Side)

Four VLA models compared on their input/output contracts.

| Property | pi0.5 | DreamZero-DROID | GR00T N1.7 | InternVLA-A1 |
| --- | --- | --- | --- | --- |
| **Parameters** | ~3B (PaliGemma 2 3B base) | 14B (Wan 2.1-I2V-14B backbone) | 3B (Cosmos-Reason2-2B VLM + DiT) | 2B / 3B (InternVL3 / Qwen3-VL base) |
| **Input: cameras** | Up to 3 views (224x224 uint8 with padding). Config specifies which slots filled, `empty_cameras` count. | 3 cameras (right external, wrist, left external) at 448x448. | 1+ RGB images (any resolution). Typically egocentric head camera. | Multi-view images (resolution varies). |
| **Input: proprioception** | float[] unnormalized state vector. Max 32 dims (zero-padded). | Proprioceptive state (joint positions). | Joint positions, velocities, EEF poses. Embodiment-specific encoder maps to fixed latent. | Not explicitly documented [needs confirmation]. |
| **Input: language** | Text string via PaliGemma tokenizer. | Text string via text encoder. | Natural language instruction via Cosmos-Reason2-2B VLM. | Natural language instruction via InternVL3. |
| **Output: action dims** | Per-embodiment. DROID: 7 (EE delta `[x,y,z,rot6d_subset,gripper]`). ALOHA: 24 (2x 7 joints + gripper). Max 32 dims. | 8 dims for DROID (7 joint velocities + 1 gripper). 6 dims for SO-101. | Embodiment-specific via learned decoder. REALM benchmark: 7 DOF joint positions. | Up to 32 dims. Flow-matching action generation with chunk size 50. |
| **Output: action semantics** | Cartesian delta (EE frame) for most configs. Joint position for some. Delta relative to first state in chunk. Gripper always absolute. | Joint velocity (DROID). Joint position (SO-101). | Joint position targets for single-arm. Latent action tokens decoded by whole-body controller (SONIC) for humanoids. | Continuous action chunks via flow matching. Semantics per-embodiment [needs confirmation]. |
| **Action chunk horizon** | 10-50 steps (configurable via `n_action_steps`). | 24 steps. | 16 actions per chunk (63.9ms inference on L40). System 2 at 10Hz, System 1 at 120Hz. | 50 steps (A1.5 spec). |
| **Normalization** | Quantile (q01/q99) or mean/std. Per-timestamp normalization for delta actions. Strategy in `config.json` `normalization_mapping`. Stats in `norm_stats.json` or processor safetensors. | Per-embodiment normalization via DreamTransform class [needs confirmation]. | Normalization via embodiment-specific encoder/decoder. Stats in `stats.json` + `relative_stats.json`. | Not documented [needs confirmation]. |
| **Per-robot adaptation** | Python adapter classes (`DroidInputs`, `LiberoInputs`, etc.) in `config.py`. Code, not declarative config. | `DreamTransform` class handles embodiment-specific processing based on dataset tags. Code, not declarative. | Embodiment-specific encoder/decoder networks. Trained per-robot or fine-tuned. `modality.json` for data format. | Per-robot fine-tuning. Code-based adaptation [needs confirmation]. |
| **Checkpoint format** | SafeTensors + `config.json` + processor safetensors + tokenizer. HF Hub convention. | SafeTensors + `config.json`. HF Hub. | SafeTensors + `config.json` + GR00T-specific configs. HF Hub. | SafeTensors + `config.json`. HF Hub. |

**Sources**: [pi0.5 paper](https://www.pi.website/download/pi05.pdf), [OpenPI config.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py), [DreamZero paper](https://arxiv.org/html/2602.15922v1), [DreamZero-DROID HF](https://huggingface.co/GEAR-Dreams/DreamZero-DROID), [GR00T N1.7 HF blog](https://huggingface.co/blog/nvidia/gr00t-n1-7), [GR00T N1 paper](https://arxiv.org/html/2503.14734v1), [InternVLA-A1 paper](https://arxiv.org/html/2601.02456v1), [InternVLA-A1.5 paper](https://arxiv.org/html/2607.04988v1)

### Key observations

1. **Action semantics vary fundamentally**: pi0.5 uses EE deltas (Cartesian frame), DreamZero-DROID uses joint velocities, GR00T uses joint positions. Same robot, three different action representations.
2. **All use SafeTensors + HF Hub**: checkpoint storage format is converged. The divergence is in the config and adapter layer.
3. **Per-robot adaptation is always code**: every model requires Python classes or trained networks to handle different robots. No model uses declarative metadata for this.
4. **Normalization strategies differ**: pi0.5 uses quantile with per-timestamp normalization for delta actions. GR00T uses embodiment-specific encoder/decoder that implicitly handles normalization. No interoperability between these approaches.
5. **Action chunk horizons vary**: 10-50 (pi0.5), 24 (DreamZero), 16 (GR00T), 50 (InternVLA). This affects how the robot controller consumes actions.

---

## 3. Dataset Format Implementations

| Property | LeRobot v3.0 | RLDS / OXE | GR00T Dataset Format | DROID (raw) |
| --- | --- | --- | --- | --- |
| **File structure** | `meta/` (info.json, stats.json, tasks.jsonl, episodes/) + `data/` (chunked parquet) + `videos/` (chunked MP4) | TFRecord shards + `dataset_info.json` (TFDS builder) | LeRobot v2 base + `meta/modality.json` | Per-episode dirs: `metadata_*.json` + `trajectory.h5` + `recordings/` (MP4, SVO) |
| **Observations: images** | MP4 video files, chunked per camera. Decoded on read. uint8 [H,W,3]. Resolution varies (typically 480x640 or 224x224). | TFRecord with image bytes (PNG/JPEG encoded). Decoded on read. | Same as LeRobot v2 (MP4 video). | MP4 (low-res stereo) + SVO (high-res raw ZED). Multiple stereo pairs. |
| **Observations: states** | Parquet columns. float32 arrays. Feature name `observation.state`, shape declared in `info.json`. | TFRecord features. float32 arrays. Schema in TFDS builder Python code. | Same as LeRobot base. | HDF5 `trajectory.h5`. float64 arrays. Joint positions, velocities, EEF poses, gripper state. |
| **Actions** | Parquet column `action`. float32 array. Shape in `info.json` (e.g., `[7]`). No semantics declaration. | TFRecord `action` feature. OXE normalizes to 7-DOF EE delta. Both `raw_action` and normalized `action` may coexist. | Same as LeRobot + `modality.json` maps index ranges to semantic fields (e.g., `"joint_position": {"start":0, "end":6}`). | HDF5 with multiple action representations simultaneously: joint velocity, Cartesian delta, Cartesian velocity. 8+ dims. |
| **Metadata schema** | `info.json`: features (names, shapes, dtypes), fps, robot_type (free-form string), codebase_version. No action semantics. | `dataset_info.json` + Python builder class. Feature types and shapes. No action semantics beyond OXE's 7-DOF convention. | `modality.json`: semantic array splitting + video key remapping. Adds `annotation_key`, `video_key` mapping. | `metadata_*.json`: scene ID, data collector ID, success flag, building ID. Task-level, not feature-level. |
| **Normalization stats** | `stats.json`: per-feature mean, std, min, max (optionally q01, q99). Separate file from schema. | Not included in format. Computed externally during training pipeline. | `stats.json` + `relative_stats.json` (for delta actions). | Not included. Computed during training. |
| **Episode delimitation** | `meta/episodes/` chunked parquet with per-episode lengths, task IDs, offsets. | TFRecord boundaries. Each episode is a separate example in the dataset. | Same as LeRobot v2. | Each episode in its own directory. |
| **Multi-rate handling** | Single fps declared in `info.json`. All modalities aligned to this rate. | Single rate per dataset. Alignment done during conversion. | Same as LeRobot. | Raw multi-rate: cameras at 15fps, joints at 100Hz+. Alignment during conversion to RLDS. |
| **Hub hosting** | HuggingFace Hub. 16K+ datasets. Git LFS for large files. | Google Cloud Storage. ~60 OXE datasets. | HuggingFace Hub (via LeRobot convention). | Google Cloud Bucket. RLDS + raw formats. HF Hub for annotations. |

**Sources**: [LeRobot v3 docs](https://huggingface.co/docs/lerobot/lerobot-dataset-v3), [RLDS GitHub](https://github.com/google-research/rlds), [OXE paper](https://arxiv.org/abs/2310.08864), [GR00T data preparation](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/data_preparation.md), [DROID dataset docs](https://droid-dataset.github.io/droid/the-droid-dataset), [DROID paper](https://arxiv.org/abs/2403.12945)

### Key observations

1. **LeRobot v3 is the convergence point**: GR00T builds on top of it, DROID is being converted to it, and RLDS datasets are being migrated. LeRobot v3 is winning the format war.
2. **But LeRobot v3 lacks action semantics**: the single biggest metadata gap. DROID records multiple action representations simultaneously; LeRobot stores a flat array with no declaration of what it means.
3. **GR00T's `modality.json` is the only implementation that adds semantic array splitting**, mapping index ranges to named fields. This is a pragmatic fix for a real gap, but it's NVIDIA-specific and currently LeRobot v2 only.
4. **Normalization stats location is inconsistent**: LeRobot and GR00T include them in the dataset. RLDS and DROID compute them externally during training. No agreement on what stats must be present.

---

## 4. Robot Controller / Middleware

| Property | ROS 2 + ros2_control | Isaac Lab Gym Env | MuJoCo dm_control |
| --- | --- | --- | --- |
| **Observation API** | Topic subscription. `sensor_msgs/JointState` (name[], position[], velocity[], effort[]), `sensor_msgs/Image` (header, height, width, encoding, data). Async pub/sub. | `env.step()` returns `obs_dict`. GPU tensors. Keys defined in `ObservationCfg` (e.g., `joint_pos`, `joint_vel`, `body_pose_w`). Gymnasium `Dict` space. | `env.observation_spec()` returns `OrderedDict` of `Array` specs (name, shape, dtype). `timestep.observation` dict. Named access via `physics.named.data.qpos['joint_name']`. |
| **Action API** | Topic publishing. `trajectory_msgs/JointTrajectory` (joint_names[], points[].positions[]), `geometry_msgs/Twist`, or `std_msgs/Float64MultiArray`. | `env.step(action_tensor)`. GPU tensor float32. Shape from `action_space` config. Applied via `Articulation.set_joint_position_target()` or similar. | `env.step(action_array)`. numpy float64 array. Shape from `env.action_spec()`. Applied to actuators by name. |
| **Joint naming** | Explicit names in `JointState.name[]` and `JointTrajectory.joint_names[]`. Names from URDF `<joint name="...">`. | `Articulation.data.joint_names` list. Order matches USD articulation. Can differ from URDF order (Isaac Lab [#7750](https://github.com/isaac-sim/IsaacLab/issues/7750)). | MJCF `<joint name="...">`. Named access via `physics.named.data`. Order matches MJCF definition order. |
| **Joint ordering** | By URDF definition order, but `JointTrajectory` carries explicit names so order doesn't matter. | By USD articulation order. **Known issue**: differs from MJCF ordering for same robot. Requires manual remapping arrays. | By MJCF definition order. Stable within a model file but may differ from URDF for same robot. |
| **Camera access** | Topic subscription per camera. `/camera_name/image_raw`. Resolution and encoding per camera driver config. | `env.scene["camera_name"].data.output["rgb"]`. GPU tensor uint8 [H,W,4] (RGBA). Camera defined in scene config USD. | `physics.render(camera_id=N)` or named camera. Returns numpy uint8 [H,W,3]. Camera defined in MJCF XML. |
| **Quaternion convention** | `geometry_msgs/Quaternion`: `(x,y,z,w)` | `(w,x,y,z)` (Isaac Sim/Lab convention) | `(w,x,y,z)` (MuJoCo convention) |

**Sources**: [ROS 2 message definitions](https://docs.ros.org/en/rolling/p/sensor_msgs/), [Isaac Lab migration guide](https://isaac-sim.github.io/IsaacLab/main/source/migration/migrating_from_isaacgymenvs.html), [Isaac Lab articulation API](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.html), [dm_control tutorial](https://notebook.community/deepmind/dm_control/tutorial), [MuJoCo Python docs](https://mujoco.readthedocs.io/en/stable/python.html)

### Key observations

1. **Joint naming is well-defined within each system but incompatible across systems**: ROS 2 uses URDF names, Isaac Lab uses USD articulation order, MuJoCo uses MJCF names. Same robot, three different joint orderings.
2. **ROS 2 is the only system that carries joint names with actions**: `JointTrajectory.joint_names[]` makes ordering explicit. Isaac Lab and dm_control use positional arrays with implicit ordering.
3. **Quaternion conventions differ**: ROS 2 uses `(x,y,z,w)`, Isaac Lab and MuJoCo use `(w,x,y,z)`. A silent source of bugs when crossing boundaries.
4. **Camera access patterns are incompatible**: topic-based (ROS 2), scene-object-based (Isaac Lab), physics-render-based (dm_control). No common abstraction.

---

## 5. Data Collection / Recording

| Property | ROSbag2 / MCAP | Isaac Lab HDF5 | LeRobot Direct Recording | DROID Recording |
| --- | --- | --- | --- | --- |
| **Storage format** | MCAP (default) or SQLite3 (`.db3`). Directory with `metadata.yaml` + storage files. | HDF5 (`.hdf5`). Hierarchical `episode_NNN/observations/...` + `processed_actions`. | LeRobot v3 directly: chunked Parquet + MP4 + JSON metadata. | Per-episode dirs: HDF5 (`trajectory.h5`) + MP4 + SVO (raw ZED). |
| **Metadata captured** | `metadata.yaml`: topics (name, type, serialization_format, QoS), message counts, duration, start time, ROS distro. | HDF5 attributes: dataset-level (not standardized). Episode structure implicit in group naming. | `info.json`: features, shapes, dtypes, fps, robot_type, codebase_version. `stats.json`: per-feature statistics. `tasks.jsonl`: task descriptions. | `metadata_*.json`: scene ID, collector ID, success flag, task description. Episode-level, not feature-level. |
| **Episode delimitation** | Separate bag file per recording session. No built-in episode concept. User must segment. | HDF5 groups: `episode_000/`, `episode_001/`, etc. Explicit structure. | `meta/episodes/` parquet with lengths, task IDs, offsets per episode. Automatic during `lerobot-record`. | Separate directory per episode. Explicit by filesystem structure. |
| **Multi-rate handling** | Preserves original rates. Topics can be 30Hz camera + 100Hz joints in same bag. Consumer handles alignment. | Single rate (simulation timestep). All data synchronized by construction. | Single fps declared. `lerobot-record` samples all sensors at configured rate. Hardware clock alignment. | Raw multi-rate: cameras ~15fps, joints ~100Hz. ZED SDK timestamps. Consumer aligns during conversion. |
| **Conversion to LeRobot** | 6+ incompatible tools: Rosetta (contract-driven YAML), Rebake (pipeline), lerobot_ros, Convert_data, Forge Robotics, leros2. Each with different resampling strategy. | `isaaclab2lerobotv3.py` (LeIsaac) or `convert_hdf5_to_lerobot.py`. Config YAML maps field names. | N/A (native format). | `droid_to_rlds.py` (official), then RLDS-to-LeRobot conversion. Or community direct converters. |
| **Normalization stats** | Not computed during recording. | Not computed during recording. | Computed automatically during `lerobot-record` and stored in `stats.json`. | Not computed during recording. |

**Sources**: [ros2/rosbag2](https://github.com/ros2/rosbag2), [Isaac Lab data export](https://docs.nvidia.com/learning/physical-ai/gr00t-e2e-workflow/latest/simulation-workflow/sim-data-export.html), [LeIsaac](https://github.com/lightwheelai/leisaac), [LeRobot recording docs](https://huggingface.co/docs/lerobot/il_robots), [DROID dataset](https://droid-dataset.github.io/droid/the-droid-dataset), [LeRobot ROS 2 RFC #4368](https://github.com/huggingface/lerobot/issues/4368)

### Key observations

1. **LeRobot direct recording is the only approach that produces training-ready data with metadata and stats in one step.** All other approaches require a conversion step.
2. **ROSbag has the richest raw data** (multi-rate, full-fidelity sensor streams) but the worst path to training data (6+ incompatible converters, each with different resampling).
3. **No recording tool captures action semantics or camera roles.** The metadata gap originates at recording time and propagates through the entire pipeline.

---

## 6. Model Registry

| Property | HuggingFace Hub | RHOAI Model Registry (Kubeflow) | NVIDIA NGC |
| --- | --- | --- | --- |
| **Model metadata** | YAML frontmatter in README.md: `pipeline_tag`, `library_name`, `base_model`, `datasets`, `license`, `tags`, `model-index` (eval results). Plus model's own `config.json`. | `RegisteredModel` / `ModelVersion` / `ModelArtifact`. Fields: `name`, `uri`, `modelFormatName`, `modelFormatVersion`. | Name, publisher, category, ML framework, precision, format, version, file size, labels, description. Structured JSON via API. |
| **Custom metadata** | `tags` (free-form strings). No structured custom fields beyond recognized YAML keys. Model card data is `ModelCardData` Python dataclass. | `customProperties`: typed map (`string_value`, `int_value`, `double_value`, `bool_value`, `MetadataStructValue` for JSON). Flat key-value, no nesting in basic types. | Labels (key-value strings). API returns structured metadata per catalog entry. |
| **Physical AI metadata?** | No structured robotics fields. `pipeline_tag: robotics` exists but carries no sub-fields. Operational metadata lives in model's `config.json`, not in card. | Technically yes via `customProperties`. Can store `{"action_dim": {"int_value": 7}}`. But no schema enforcement, no validation, no standardized keys. | No robotics-specific metadata. VLA models (GR00T) listed but with generic metadata only. |
| **Query / filtering** | Full-text search. Filter by `pipeline_tag`, `library_name`, `language`, `license`. Cannot filter by action space, embodiment, or normalization. | REST API with name/owner filtering. Cannot query by `customProperties` values. No structured search for Physical AI attributes. | API query by `resourceType`, `tags`, `category`. No robotics-specific query parameters. |
| **Artifact packaging** | Git LFS on HF Hub. Files in repo: safetensors, config.json, tokenizer, etc. OCI ModelCar packaging emerging. | URI-based: `s3://`, `oci://`, local paths. `storageKey` for credentials. | NGC container registry. Docker/OCI images. Models as downloadable archives. |
| **Versioning** | Git-based (commits, branches, tags). `base_model` field tracks lineage. | `RegisteredModel` -> `ModelVersion` hierarchy. Stage transitions (Staging/Production/Archived). | Version strings per model. No formal lineage tracking. |

**Sources**: [HF Model Cards docs](https://huggingface.co/docs/hub/en/model-cards), [Kubeflow Model Registry](https://www.kubeflow.org/docs/components/hub/reference/rest-api/), [model-catalog-bridge](https://github.com/redhat-ai-dev/model-catalog-bridge), [NGC Catalog docs](https://docs.nvidia.com/ngc/latest/ngc-catalog-user-guide.html)

### Key observations

1. **HuggingFace Hub is the de facto registry for VLA models**: pi0.5, DreamZero, GR00T N1.7, InternVLA all publish on HF Hub. NGC hosts NVIDIA models but they're also mirrored to HF.
2. **No registry can query by Physical AI attributes**: you cannot search for "all models compatible with Franka Panda 7-DOF" or "all models using quantile normalization" in any registry.
3. **RHOAI Model Registry has the most extensible metadata model** (`customProperties` with typed values including JSON structs) but no Physical AI schema to fill it with.

---

## Convergence Analysis

### Where implementations agree

| Aspect | Converged standard | Evidence |
| --- | --- | --- |
| **Weight storage** | SafeTensors | All four VLA models use it. ~42% of HF Hub models (2025). PyTorch Foundation stewardship. |
| **Video storage** | MP4 (H.264/H.265) | LeRobot v3, GR00T, DROID all use MP4 for camera frames. Replaced per-frame PNG. |
| **Dataset hosting** | HuggingFace Hub | 16K+ robot datasets. LeRobot, GR00T, DreamZero, DROID all publish here. |
| **Model hosting** | HuggingFace Hub | All major VLA models publish checkpoints on HF Hub. |
| **Config format** | JSON (`config.json`) | Every VLA model uses HF-convention `config.json` for policy config. |
| **Serving protocol** | OpenPI WebSocket convention (for multi-model/production serving) | vLLM-Omni and SGLang explicitly target OpenPI compatibility. LeRobot uses gRPC for single-client on-robot deployment — different niche. |
| **Training framework** | LeRobot (for fine-tuning) | pi0.5, DreamZero, GR00T all support LeRobot datasets for training. |
| **RL env API** | Gymnasium | Isaac Lab, ManiSkill, robosuite all expose Gymnasium-compatible interfaces. |

### Where implementations look the same but differ

| Aspect | Surface similarity | Actual divergence |
| --- | --- | --- |
| **`action` array** | All datasets have a float[] `action` feature | pi0.5: EE delta in wrist frame. DreamZero-DROID: joint velocities. GR00T: joint positions. Same name, different physical meaning. |
| **`config.json`** | All models ship `config.json` | Fields, structure, and semantics are model-specific. pi0.5 has `normalization_mapping`; GR00T has separate embodiment config; DreamZero has video generation params. No common schema. |
| **`stats.json`** | LeRobot and GR00T both include stats | LeRobot: `{mean, std, min, max, [q01, q99]}`. GR00T adds `relative_stats.json` for delta actions. Pi0.5 uses per-timestamp normalization not represented in stats format. |
| **Camera naming** | All have camera observations | LeRobot: `observation.images.<name>` (free-form). OpenPI: `observation/<name>` (per-robot config). GR00T: `video_key` remapping in `modality.json`. DROID: ZED camera IDs. |
| **Joint ordering** | All robots have named joints | URDF order != USD articulation order != MJCF order for the same robot (confirmed by Isaac Lab [#7750](https://github.com/isaac-sim/IsaacLab/issues/7750)). |
| **Quaternion convention** | All use quaternions for orientation | ROS 2: `(x,y,z,w)`. Isaac Lab/MuJoCo: `(w,x,y,z)`. Silent corruption if crossed without conversion. |
| **Normalization** | All normalize actions for training | Quantile (pi0.5), z-score (older policies), baked-in (ONNX export), embodiment-specific encoder (GR00T). Incompatible approaches behind a common concept. |

---

## Ecosystem Momentum

### Clear winners

| Standard | Metric | Trajectory |
| --- | --- | --- |
| **LeRobot v3** | 16K+ datasets, 2.2K+ contributors, ICLR 2026 paper | Dominant dataset format. GR00T builds on it. RLDS being migrated to it. Active v3 development. |
| **HuggingFace Hub** | 1.4M+ models, all major VLA models published here | De facto registry for open-source ML. No credible alternative for VLA models. |
| **SafeTensors** | ~42% of HF Hub models, PyTorch Foundation stewardship | Default weight format. Replaced pickle-based .bin for security. |
| **Gymnasium** | 18M+ installs | De facto RL environment API. Isaac Lab, ManiSkill, robosuite all adopt it. |
| **OpenPI serving convention** | Adopted by vLLM-Omni, multiple community forks | Emerging as the VLA serving wire format. vLLM-Omni compatibility = industry validation. |

### Active competition (no clear winner)

| Domain | Competitors | Status |
| --- | --- | --- |
| **Action representation** | EE delta (pi0.5, OXE), joint velocity (DreamZero), joint position (GR00T), learned tokenization (FAST+) | No convergence. Research papers propose unification (CalibAll, UniAct) but none adopted. |
| **ROSbag-to-LeRobot conversion** | Rosetta, Rebake, lerobot_ros, Convert_data, Forge Robotics, leros2 | 6+ tools, none dominant. LeRobot RFC #4368 discusses native ROS 2 support but no resolution. |
| **Simulation platform** | Isaac Lab, MuJoCo (dm_control), Gazebo (ROS 2), ManiSkill/SAPIEN | Isaac Lab gaining momentum (GPU acceleration, GR00T integration). MuJoCo dominant in research. No interop standard between them. |
| **VLA serving** | OpenPI, vLLM-Omni, SGLang, LeRobot PolicyServer, direct deployment (Isaac ROS) | OpenPI convention emerging for production/multi-model serving (vLLM-Omni, SGLang adopt it). LeRobot PolicyServer occupies a different niche: single-client on-robot gRPC with serialized processing pipelines. |

### Declining

| Standard | Evidence |
| --- | --- |
| **RLDS / TFRecord** | Being superseded by LeRobot. New projects standardize on LeRobot, not RLDS. OXE datasets being converted. |
| **pickle-based .bin** | Replaced by SafeTensors for security. |
| **Per-frame image storage** | Replaced by chunked MP4 (LeRobot v3). Better compression and I/O. |

---

## Summary

The Physical AI ecosystem is converging on a clear stack for **storage and hosting** (SafeTensors + MP4 + LeRobot v3 + HF Hub) and an emerging standard for **production serving** (OpenPI convention, adopted by vLLM-Omni and SGLang). LeRobot's PolicyServer occupies a complementary niche — single-client on-robot deployment via gRPC — and is the only server where the full processing pipeline is declaratively described (serialized JSON step lists) rather than hardcoded. The divergence remains in the **semantic layer**: what actions mean, how normalization works, which joints map to which indices, and what cameras are called. This semantic gap cannot be solved by format convergence alone — it requires explicit metadata declarations that no current standard provides.
