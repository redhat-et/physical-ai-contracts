# Community Solutions Survey: Are Our Observed Problems Already Being Solved?

**Date**: 2026-09-18
**Purpose**: Determine whether the interoperability problems observed in 4 Red Hat internal repos are already addressed by existing communities, before proposing new solutions.

**Important caveat**: Our observations come from a small number of Red Hat engineers working on specific repos. Some of these may be known solved problems where existing tooling was simply not used.

---

## Summary of Findings

**The short answer: partially solved, actively evolving, but no unified solution exists.**

The LeRobot ecosystem is converging on dataset-level validation tooling (lerobot-doctor, RDA, raftaar), and LeRobot v0.6–v0.7 is actively improving metadata and compatibility. However, **cross-artifact compatibility checking** (model ↔ dataset ↔ robot ↔ environment) remains a gap that no single project owns. Several small tools are emerging to fill pieces of this gap, but they are early-stage (0 stars, research prototypes) and fragmented.

| Observed Problem | Status | Key Evidence |
| --- | --- | --- |
| Camera mapping errors | **Partially addressed** by LeRobot | LeRobot's `rename_map` + feature validation catches mismatches at load time, but bugs persist (PR [#4578](https://github.com/huggingface/lerobot/pull/4578): silent dropping of stored camera maps) |
| Normalization mismatches | **Actively broken** in LeRobot | Multiple open issues: silent skip after pipeline migration ([#4415](https://github.com/huggingface/lerobot/issues/4415)), empty normalization from config bug ([#4647](https://github.com/huggingface/lerobot/issues/4647)), dual normalization disagreement. LeRobot is aware but fixes are incomplete |
| Joint ordering across simulators | **Recognized, no standard solution** | Isaac Lab GitHub [#7750](https://github.com/isaac-sim/IsaacLab/issues/7750) confirms it's a live pain point. USDPhysics + MjcPhysics schemas are in development (NVIDIA + DeepMind collaboration). ROS-Industrial [proposed](https://github.com/ros-industrial/ros_industrial_issues/issues/52) standardizing joint naming but no consensus reached |
| Checkpoint loading semantics | **Not addressed** by any community | This is LeRobot-specific (`--policy.path` vs `--policy.pretrained_path`). No external standard or tool validates checkpoint loading semantics |
| No model-dataset compatibility tool | **Emerging but early-stage** | lerobot-doctor (40 stars), knownrobot (0 stars), raftaar (0 stars), RDA (0 stars) each solve pieces. No tool checks full model-dataset-robot compatibility |

---

## Detailed Findings by Community

### 1. LeRobot / Hugging Face

**LeRobot is the most active community addressing these problems, but from within their own ecosystem, not as a cross-project standard.**

#### What LeRobot already does

- **Feature validation at load time**: LeRobot v3.0's dataset loader checks that features declared in `info.json` match what's in the parquet/video files. Missing or extra features trigger an error with a suggestion to use `--rename_map`. Source: [LeRobot Feature Mapping docs](https://deepwiki.com/huggingface/lerobot/5.3-teleoperation-and-recording)

- **`info.json` metadata**: v3.0 includes structured feature descriptions (names, shapes, dtypes), episode counts, task definitions, and aggregated statistics in `stats.json`. Source: [LeRobotDataset v3.0 docs](https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3)

- **Normalization mapping config**: Users can override normalization strategy per feature via `--policy.normalization_mapping`. Source: [pi0.5 docs](https://huggingface.co/docs/lerobot/pi05)

- **v0.6.0 (July 2026)**: Introduced "homogenizing policy and evaluation API contracts, defining explicit interfaces for 'LeRobot-compatible' policies, environments, and robots." Source: [v0.6.0 blog](https://huggingface.co/blog/lerobot-release-v060)

#### What's still broken in LeRobot

- **Normalization silently skipped for multi-dataset models** ([#4415](https://github.com/huggingface/lerobot/issues/4415)): After pipeline migration, stats saved with dataset-prefixed keys don't match unprefixed lookup keys → normalization becomes identity op → policy scores ~0.

- **Config bug causes empty normalization** ([#4647](https://github.com/huggingface/lerobot/issues/4647)): `PreTrainedConfig.from_pretrained()` leaves `pretrained_path=None` → builds fresh processor with empty stats → policy runs but produces garbage. Difference: cum reward 4.8 (broken) vs 118–199 (correct).

- **Migration script incomplete** ([#4649](https://github.com/huggingface/lerobot/issues/4649), [#4655](https://github.com/huggingface/lerobot/issues/4655)): `migrate_policy_normalization` emits empty `config.json`, doesn't copy weights, crashes on config schema drift.

- **Camera rename_map silently dropped** ([PR #4578](https://github.com/huggingface/lerobot/pull/4578)): Default empty `rename_map` override replaced the checkpoint's stored camera mapping.

#### LeRobot v0.7.0 roadmap (in progress)

Relevant planned work from [Issue #3832](https://github.com/huggingface/lerobot/issues/3832):

- **MultiLeRobotDataset enhancement**: weighted sampling, explicit key mapping per dataset, per-robot normalization stats
- **Streaming-first dataset redesign**: new format where continuous streams are the primitive
- **Multi-frequency modality recording**: research investigation into multi-rate sensor data
- **ROS2 integration exploration**: "explore native compatibility with ROS2 for broader ecosystem reach"

**Assessment**: LeRobot is actively improving its internal metadata and validation, but these are framework-internal solutions, not cross-project standards. They solve the problem for users within the LeRobot ecosystem. The v0.7 roadmap shows awareness of interoperability needs (ROS2, multi-dataset) but no movement toward a portable metadata standard.

---

### 2. Third-Party Validation Tools (LeRobot Ecosystem)

An emerging ecosystem of tools validates LeRobot datasets, but all are early-stage:

#### lerobot-doctor ([jashshah999/lerobot-doctor](https://github.com/jashshah999/lerobot-doctor))

- **Stars**: 40 | **Status**: Active, deployed on HF Spaces as "Doctor" tab on LeRobot Dataset Visualizer
- **What it does**: 12 diagnostic checks including metadata integrity, temporal consistency, action validity, statistics sanity, cross-episode schema consistency, and policy compatibility gating
- **Policy compatibility**: `gate` command checks dataset compatibility for ACT, Diffusion, SmolVLA, and Pi0 policies (wrong dims, short episodes, NaN normalization)
- **Kinematics check**: Validates actions against a robot's URDF joint limits — not just statistical bounds but physical reachability
- **Limitation**: Checks dataset quality and basic policy compatibility, but does NOT check against a specific model checkpoint's declared inputs/outputs

#### raftaar ([aslansd/raftaar](https://github.com/aslansd/raftaar))

- **Stars**: 0 | **Status**: Research prototype, explicitly labeled "honestly labelled"
- **What it does**: Predicts which policy class will fail on a dataset by analyzing demonstration distributions. 5 detectors: averaging hazard (conflicting strategies), coverage gaps, action inconsistency, representation shards, idle channels
- **Key insight**: Detects when regression-based policies will "converge to the average" of conflicting strategies — a failure mode training loss won't catch
- **Limitation**: No cross-dataset validation yet. Awaiting validation across public datasets. CPU-only, no GPU stack dependency.

#### RDA — Robot Data Audit ([liesliy/rda](https://github.com/liesliy/rda))

- **Stars**: 0 | **Status**: Active, validated on AgiBotWorld2026
- **What it does**: 21 metrics across 4 layers (integrity, trajectory, dataset profiling, recommendations). Three-tier verdicts. Recommendations calibrated to model type (temporal vs frame-wise)
- **Limitation**: Dataset quality audit, not model-dataset compatibility checking

#### knownrobot ([arcofdescent1/knownrobot](https://github.com/arcofdescent1/knownrobot))

- **Stars**: 0 | **Status**: Early, community sprint planned Sept-Oct 2026
- **What it does**: Compares declared hardware, calibration, frequency, feature semantics between a policy and target robot setup. Generates portable `robot-skill.yaml` manifests with JSON Schema validation
- **Most relevant to our problem**: This is the closest thing to a "contract" system in the wild. It defines an evidence contract (1.0) covering policy framework, robot/gripper/sensors, control frequency, observation/action shapes, dataset schema, dependencies, and evaluation evidence
- **Limitation**: Focused on SO-100/SO-101 LeRobot hardware. 0 stars, 15 commits. Very early.

**Assessment**: These tools collectively validate the hypothesis that dataset-policy compatibility checking is a real unmet need — multiple independent developers are building solutions. But they are fragmented, early-stage, and focused on dataset quality rather than cross-artifact compatibility.

---

### 3. OpenPI / Physical Intelligence

**OpenPI defines per-embodiment adapters, not a portable standard.**

- OpenPI handles action space heterogeneity through **per-robot policy classes** (e.g., `LiberoInputs`, `LiberoOutputs`) that define custom input/output mappings. Each robot embodiment gets its own adapter. Source: [OpenPI README](https://github.com/Physical-Intelligence/openpi)
- Camera mapping uses string-keyed observation dictionaries (`observation/exterior_image_1_left`, `observation/wrist_image_left`). Each robot config maps its camera topology to these keys.
- Normalization uses a dedicated `compute_norm_stats.py` step per config, with quantile-based statistics. The system warns about "dimensions that are rarely used" causing huge normalized values.
- Training data must be in **LeRobot dataset format**. Conversion scripts are provided per dataset.

**Assessment**: OpenPI does not define a standard interface that other projects could adopt. It's a well-engineered but PI-specific adapter system. The action space is implicitly defined per policy class, not declared in metadata. However, PI's market position means their conventions (action chunking, flow matching, LeRobot format) are becoming de facto norms.

---

### 4. Open X-Embodiment (OXE) / Google DeepMind

**OXE defines a cross-embodiment action space convention but not a validation tool.**

- OXE standardized on **7-DOF end-effector delta actions**: `(dx, dy, dz, droll, dpitch, dyaw, gripper)`. Each robot's native action space is mapped to this common space via forward kinematics. Source: [OXE dataset](https://robotics-transformer-x.github.io/)
- State representation: 8-dimensional `[ee_x, ee_y, ee_z, ee_quat_x, ee_quat_y, ee_quat_z, ee_quat_w, gripper_width]`
- Data format: RLDS (TensorFlow Dataset records). All data converted to RLDS for RT-X models.
- Cross-embodiment training demonstrated 50–200% success rate improvements vs single-embodiment training.

**Assessment**: OXE provides a concrete precedent for action space standardization, but it's specific to the RT-X model family and RLDS format. The 7-DOF EE delta convention is influential but not universally applicable (humanoids have 30+ DOF; whole-body controllers need joint-space actions). No validation tool checks whether a new dataset's action space is OXE-compatible.

---

### 5. ROS / Open Robotics

**Multiple interoperability efforts, but focused on simulation and control interfaces, not ML model compatibility.**

#### Joint naming

- ROS-Industrial [proposed](https://github.com/ros-industrial/ros_industrial_issues/issues/52) standardizing to `link_n`/`joint_n` naming across all robot support packages. No consensus reached.
- A [Discourse thread](https://discourse.openrobotics.org/t/naming-standards-for-joints-in-humanoid-robots/3992) on humanoid joint naming found only REP-120 (which specifies some link names but not joint names). The author noted the portability challenge of using the same code on different humanoids.
- REP-0158 defines `ros:joint:name` as "the single source of truth for ROS-facing joint identity" in OpenUSD assets, but this is for asset description, not for cross-simulator policy transfer.

#### Simulation interfaces

- The `simulation_interfaces` [package](https://github.com/ros-simulation/simulation_interfaces) defines standard ROS 2 services for controlling simulators (reset, step, spawn/delete entities, get/set state). Implemented in Gazebo Jetty, Isaac Sim 5.0, and O3DE 2505.
- `ros2_control` provides [simulator integrations](https://control.ros.org/rolling/doc/simulators/simulators.html) for Gazebo, Isaac Sim, Webots, MuJoCo, and AGX Dynamics.

#### REP-0158 status

- Currently in **Draft** status, all feedback ingested, author has [requested](https://discourse.openrobotics.org/t/draft-rep-158-openusd-conventions-for-simulation-asset-interoperability-in-open-source-robotics/55526) to move to the voting phase.
- Defines OpenUSD conventions for joint naming, camera optical frames, vendor-isolated layering, ROS integration schemas.

**Assessment**: The ROS community addresses simulation-level interoperability (standardized interfaces for controlling simulators, asset description via USD) but not ML-level interoperability (does this model's action space match this robot? can this dataset train this policy?). The gap between "simulation interoperability" and "ML training interoperability" is exactly where the unmet need sits.

---

### 6. NVIDIA / Isaac Ecosystem

#### Isaac Lab multi-backend

- Isaac Lab now supports [multiple physics backends](https://isaac-sim.github.io/IsaacLab/main/source/experimental-features/newton-physics-integration/index.html): PhysX, Newton, MuJoCo (via Warp). This addresses sim-to-sim at the physics level but not at the policy/action-space level.
- USDPhysics + MjcPhysics schemas are being co-developed by NVIDIA and Google DeepMind to standardize physics properties across USD and MuJoCo.

#### SimReady Foundation

- See `research/simready-foundation-analysis.md` for detailed analysis.
- Addresses asset validation (is this robot USD valid for Isaac Sim?) but not ML compatibility (can this model train on this dataset?).

#### NIM manifest

- NVIDIA NIM uses a model manifest (`model_manifest.yaml`) that catalogs deployment profiles (backend, precision, tensor parallelism). Source: [NIM docs](https://docs.nvidia.com/nim/large-language-models/latest/deployment/model-profiles-and-selection.html)
- Focused on inference deployment configuration, not training compatibility.

#### GR00T data format

- NVIDIA GR00T requires LeRobot v2 format plus an additional `meta/modality.json` not in standard LeRobot. Source: [Isaac-GR00T data prep](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/data_preparation.md)
- This is an example of the fragmentation problem: GR00T adds its own metadata file on top of LeRobot, creating a format variant.

**Assessment**: NVIDIA's efforts address simulation asset interoperability (SimReady, USDPhysics) and inference deployment (NIM), but not training-time model-dataset compatibility. Their GR00T integration actually *creates* interoperability friction by requiring a non-standard metadata file.

---

### 7. Formal Standards

#### ISO/WD 26264-1: Humanoid Robot Datasets

- First international standard for humanoid robot datasets, under development by ISO/TC 299/WG 16. Source: [arxiv.org/html/2606.19769v1](https://arxiv.org/html/2606.19769v1)
- Proposes a "dataset passport" with episode schema, physical-coherence record, and lifecycle record.
- Focused on making datasets interpretable and reusable, not on model-dataset compatibility checking.
- Authors are primarily from Chinese institutions (AIRS, RIAMB, SCUT). Governance: ISO committee.
- **Assessment**: Too early and too heavyweight for our needs, but worth tracking. The "physical-coherence record" concept (timing, coordinate frames, calibration, kinematics, units) overlaps with our problem space.

#### Croissant (MLCommons)

- Metadata format for ML-ready datasets, built on schema.org. 400K+ datasets on HF, Kaggle, OpenML. Source: [Croissant paper](https://arxiv.org/html/2403.19546v1)
- Croissant 1.1 released. NeurIPS now requires Croissant metadata for dataset track submissions.
- **No evidence of robotics-specific adoption**. Primary focus is biomedical/health AI.
- Describes dataset structure (fields, types, splits) but not domain-specific semantics (action spaces, joint names, camera roles).
- **Assessment**: Croissant is a general-purpose dataset metadata format. It could be extended for robotics, but no one appears to be doing this. Worth considering as a foundation layer.

---

### 8. ML-General Metadata Standards

#### ONNX metadata

- `onnx.checker.check_model()` validates structural correctness (graph structure, operator schemas). Does NOT verify numerical correctness or compatibility with input data. Source: [ONNX checker docs](https://onnx.ai/onnx/api/checker.html)
- Model metadata is a free-form `metadata_props` dict. No standard schema for documenting action spaces, normalization, or training data requirements.
- Shape mismatches at inference "produce silent garbage output with no exception thrown."
- **Assessment**: ONNX validates model structure but not model-data compatibility. No robotics-specific extensions.

#### MLflow model signatures

- Typed input/output schemas (`ColSpec` for tabular, `TensorSpec` for tensor data). Auto-inferred from data. Validated at serving time (REST endpoint rejects schema mismatches). Source: [MLflow docs](https://mlflow.org/docs/latest/ml/model/signatures/)
- Does NOT validate batch predictions — `pyfunc.predict()` silently accepts malformed input.
- Signatures travel through Model Registry but are not enforced outside MLflow serving.
- **Assessment**: MLflow signatures are the closest existing standard to "typed model contracts." They handle shapes and dtypes but lack domain semantics (what does dimension 3 of the action vector mean?). Could be extended.

#### HF model cards

- YAML frontmatter with structured metadata. Machine-readable, searchable. EU AI Act compliance driving adoption. Source: [HF model cards docs](https://huggingface.co/docs/hub/en/model-cards)
- For VLA models (e.g., SmolVLA, X-VLA): model cards include training config, action chunking params, camera setup, but as free-text prose, not machine-readable structured fields.
- **Assessment**: Model cards are documentation, not contracts. The structured metadata covers licensing, tasks, metrics — not input/output shapes, action semantics, or compatibility constraints.

#### SafeTensors metadata

- `__metadata__` section: optional free-form string-to-string map in the file header. Source: [SafeTensors docs](https://huggingface.co/docs/safetensors/metadata_parsing)
- Used informally for training metadata (learning rate, epochs, etc.). No standard schema.
- Remote parsing via HTTP range requests enables metadata inspection without downloading weights.
- **Assessment**: SafeTensors metadata is a transport mechanism, not a schema. It could carry contract metadata, but defines no structure for it.

---

### 9. Robot Description Formats (URDF / MJCF / USD)

- **URDF**: ROS standard. Cannot represent closed kinematic chains, no contact parameters, uses `package://` URIs. Canonical source for most robots.
- **MJCF**: MuJoCo native. Richer physics (contact, actuator models, tendons). Different joint hierarchy from URDF.
- **USD**: Increasingly the "interop format" with OpenUSD Core Spec 1.0 (Dec 2025), ISO certification in progress.

#### Conversion tooling

- `mujoco_ros2_control` includes [URDF→MJCF conversion](https://control.ros.org/rolling/doc/mujoco_ros2_control/mujoco_ros2_control/docs/tools.html), labeled "highly experimental."
- Multiple third-party converters exist ([urdf2mjcf](https://lobehub.com/skills/plurigrid-asi-urdf2mjcf), [discoverse-dev/urdf-to-mjcf](https://github.com/discoverse-dev/urdf-to-mjcf)), handling joint type mapping, mesh conversion, actuator generation.
- **Joint ordering is NOT handled by any converter** — converters translate the kinematic structure but don't address the index-ordering problem that causes policy transfer failures.

**Assessment**: Format conversion tools exist and are improving. The joint *ordering* problem (which joint index maps to which physical joint) is distinct from joint *naming* and is not addressed by any existing tool or standard.

---

## Gap Analysis

### What's covered (don't reinvent)

1. **Dataset structural validation**: lerobot-doctor, RDA cover this well within the LeRobot ecosystem
2. **Simulation asset interoperability**: REP-0158 + SimReady + USDPhysics are converging on USD-based standards
3. **Cross-embodiment action space convention**: OXE's 7-DOF EE delta is an established (if limited) precedent
4. **Simulator control interfaces**: ROS 2 `simulation_interfaces` provides cross-simulator control API
5. **Dataset metadata format**: LeRobot v3.0 `info.json` + Croissant provide structured dataset descriptions
6. **Model serving profiles**: NVIDIA NIM manifests handle inference deployment configuration

### What's NOT covered (potential gap to fill)

1. **Cross-artifact compatibility checking**: No tool or standard validates that a model's declared inputs match a dataset's declared outputs match a robot's declared capabilities. lerobot-doctor checks dataset-policy compatibility for 4 specific policies, but this is hardcoded, not schema-driven.

2. **Portable action space semantics**: OXE defines one convention (7-DOF EE delta), but there's no standard way to declare "this model expects 29-DOF joint-space actions in this specific ordering with these semantics." Each framework (LeRobot, OpenPI, GR00T) defines its own.

3. **Normalization contract**: No standard declares whether normalization is baked into a checkpoint, external (which format?), or expected to be computed at load time. This is the source of multiple LeRobot bugs.

4. **Camera role semantics**: No standard distinguishes world cameras from robot-mounted cameras or maps between dataset camera names and model input slots. LeRobot's `rename_map` is a runtime workaround, not a declarative metadata standard.

5. **Joint ordering mapping**: Distinct from joint naming. When transferring a policy trained in Isaac Lab to MuJoCo, which index maps to which physical joint? No standard or tool automates this.

6. **Cross-framework model metadata**: A model checkpoint currently carries no portable, machine-readable declaration of its expected inputs, action space, normalization requirements, or compatible embodiments. HF model cards describe this in prose; SafeTensors metadata could carry it but has no schema for it.

### The "crack between communities"

The gap is specifically at the intersection of:

- **ML model metadata** (what does this model expect?) — owned by no single community
- **Dataset metadata** (what does this dataset provide?) — converging within LeRobot, but LeRobot-specific
- **Robot description** (what can this robot do?) — converging on USD, but focused on physics not ML

No community owns the question: "Can this model train on this dataset for this robot?" Each community owns one vertex of this triangle.

---

## Recommendations for Next Steps

1. **Don't build dataset validation tools** — lerobot-doctor and RDA are addressing this. Contributing to them would be more impactful than creating a parallel tool.

2. **Investigate whether LeRobot would accept richer metadata standards** — LeRobot v0.7's roadmap (MultiLeRobotDataset, ROS2 exploration) suggests openness to interoperability. A proposal for standardized action-space semantics in `info.json` might land upstream.

3. **Investigate whether knownrobot's `robot-skill.yaml` schema could be a starting point** — it's the closest thing to a "contract" in the wild, with JSON Schema validation. But it's SO-100/SO-101 specific and has zero community traction.

4. **The normalization contract is the most concrete, smallest-scope problem** — it affects every training run, has multiple active LeRobot bugs, and no existing standard addresses it. A lightweight spec declaring normalization strategy (baked/external/compute-on-load, which stats format, which features) could be a focused contribution.

5. **The action space semantics problem is the highest-impact but hardest** — it requires cross-community agreement (LeRobot, OpenPI, OXE, GR00T). Start by documenting existing conventions rather than proposing a new one.

---

## Sources

### LeRobot

- [PR #4578: Fix rename_map silent dropping](https://github.com/huggingface/lerobot/pull/4578)
- [Issue #4415: Normalization silently skipped for multi-dataset models](https://github.com/huggingface/lerobot/issues/4415)
- [Issue #4647: pretrained_path=None causes empty normalization](https://github.com/huggingface/lerobot/issues/4647)
- [Issue #4649: Migration script produces incomplete output](https://github.com/huggingface/lerobot/issues/4649)
- [Issue #4655: Config schema drift crashes migration](https://github.com/huggingface/lerobot/issues/4655)
- [Issue #2787: XVLA camera setup limitations](https://github.com/huggingface/lerobot/issues/2787)
- [Issue #3134: v0.6.0 roadmap](https://github.com/huggingface/lerobot/issues/3134)
- [Issue #3832: v0.7.0 roadmap](https://github.com/huggingface/lerobot/issues/3832)
- [LeRobot v0.6.0 blog post](https://huggingface.co/blog/lerobot-release-v060)
- [LeRobotDataset v3.0 docs](https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3)
- [Pi0.5 docs](https://huggingface.co/docs/lerobot/pi05)
- [Feature Mapping and Validation](https://deepwiki.com/huggingface/lerobot/5.3-teleoperation-and-recording)

### Third-Party Tools

- [jashshah999/lerobot-doctor](https://github.com/jashshah999/lerobot-doctor) (40 stars)
- [HF Space: lerobot-doctor](https://huggingface.co/spaces/jashshah999/lerobot-doctor)
- [aslansd/raftaar](https://github.com/aslansd/raftaar) (0 stars, research prototype)
- [liesliy/rda](https://github.com/liesliy/rda) (0 stars, validated on AgiBotWorld2026)
- [arcofdescent1/knownrobot](https://github.com/arcofdescent1/knownrobot) (0 stars, community sprints planned)

### OpenPI

- [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)

### OXE

- [Open X-Embodiment dataset](https://robotics-transformer-x.github.io/)
- [OXE paper](https://arxiv.org/abs/2310.08864)

### ROS / Open Robotics

- [REP-0158 Draft Discussion](https://discourse.openrobotics.org/t/draft-rep-158-openusd-conventions-for-simulation-asset-interoperability-in-open-source-robotics/55526)
- [ROS-Industrial joint naming proposal](https://github.com/ros-industrial/ros_industrial_issues/issues/52)
- [Humanoid joint naming discussion](https://discourse.openrobotics.org/t/naming-standards-for-joints-in-humanoid-robots/3992)
- [simulation_interfaces](https://github.com/ros-simulation/simulation_interfaces)
- [ros2_control simulator integrations](https://control.ros.org/rolling/doc/simulators/simulators.html)

### NVIDIA

- [Isaac Lab GitHub #7750: Isaac→MuJoCo transfer failure](https://github.com/isaac-sim/IsaacLab/issues/7750)
- [Isaac Lab Newton integration docs](https://isaac-sim.github.io/IsaacLab/main/source/experimental-features/newton-physics-integration/index.html)
- [NIM model profiles docs](https://docs.nvidia.com/nim/large-language-models/latest/deployment/model-profiles-and-selection.html)
- [Isaac-GR00T data preparation](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/data_preparation.md)
- [SimReady Foundation](https://nvidia.github.io/simready-foundation/latest/)

### Standards & Formats

- [ISO/WD 26264-1 paper](https://arxiv.org/html/2606.19769v1)
- [Croissant paper](https://arxiv.org/html/2403.19546v1)
- [Croissant Baker](https://arxiv.org/html/2605.15079v1)
- [ONNX checker docs](https://onnx.ai/onnx/api/checker.html)
- [ONNX metadata docs](https://onnx.ai/onnx/repo-docs/MetadataProps.html)
- [MLflow model signatures](https://mlflow.org/docs/latest/ml/model/signatures/)
- [HF model cards](https://huggingface.co/docs/hub/en/model-cards)
- [SafeTensors metadata](https://huggingface.co/docs/safetensors/metadata_parsing)
- [Agent Skills spec](https://agentskills.io/specification)

### URDF/MJCF Conversion

- [mujoco_ros2_control URDF→MJCF](https://control.ros.org/rolling/doc/mujoco_ros2_control/mujoco_ros2_control/docs/tools.html)
- [discoverse-dev/urdf-to-mjcf](https://github.com/discoverse-dev/urdf-to-mjcf)

### Alliance for OpenUSD

- [OpenUSD Core Spec 1.0](https://aousd.org/news/alliance-for-openusd-drives-global-3d-data-interoperability-and-agentic-ai-workflows-with-new-core-specification-milestones-and-members/)
- [AOUSD ISO certification](https://www.linuxfoundation.org/press/aousd_prmarch2026)
