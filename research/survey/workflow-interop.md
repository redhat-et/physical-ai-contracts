# Prior Art Survey: Workflow Composition, Interoperability & Environment Interfaces

**Date**: 2026-09-18
**Scope**: Workflow composition standards, RL environment interfaces, cross-embodiment conventions, manipulation benchmark frameworks, and existing compatibility-checking tools.
**Purpose**: Input to OCTOET-2177 (Schema Research & Prior Art Survey)

---

## 1. OpenPI (Physical Intelligence)

**Source**: [github.com/Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi), [pi.website/blog/openpi](https://www.pi.website/blog/openpi)

### What it describes

OpenPI is a framework for training and deploying pi0/pi0.5 VLA models. It is _not_ a protocol specification or interoperability standard. It provides:

- **Per-robot input/output classes**: `LiberoInputs`/`LiberoOutputs`, `DroidInputs`, `AlohaInputs`, etc. — Python classes that map between a robot's observation/action format and the model's internal representation.
- **Client-server inference**: A websocket-based `openpi-client` package that decouples model inference (GPU server) from robot control (laptop/edge device). The server listens on port 8000.
- **Normalization**: Precomputed `norm_stats.json` per dataset. Supports z-score or quantile normalization. Users must inspect q01/q99/std values for correctness.
- **Config-driven embodiment**: `TrainConfig` objects in `config.py` define per-robot configurations: action dimensions, action horizons, normalization mode, data config class, checkpoint path.
- **LeRobot as the data layer**: Training data must be converted to LeRobot format. A conversion script pattern is provided.

### What it doesn't describe

- No standardized protocol or schema for model-robot interfaces. Each robot gets a bespoke Python class.
- No machine-readable metadata about action spaces, observation formats, or camera mappings. These are embedded in Python code.
- No compatibility checking between models and datasets/robots. Validation is implicit — if the Python classes are wrong, you get runtime errors.
- No formal action space taxonomy. Action spaces (joint position, EE delta, etc.) are encoded in code, not declared metadata.

### Type system & extensibility

Python dataclasses and config objects. Adding a new robot requires writing a new `*Inputs`/`*Outputs` class and a `DataConfigFactory` subclass. Extensible but not declarative.

### Community adoption

High adoption as a model framework. Widely forked (dozens of forks visible on GitHub). But adoption is for the _models_, not for any interface standard. Each fork customizes for its own robot.

### Already solving our problems?

**No.** OpenPI demonstrates the interoperability problem rather than solving it. Every new robot requires writing custom Python adapter code. The per-robot input/output classes are exactly the manual "compatibility glue" that a contract system would replace. The key observation: OpenPI's `norm_stats.json` + `config.py` + per-robot `*Inputs` classes contain the metadata a contract would capture — but in imperative Python code, not declarative metadata.

---

## 2. Gymnasium / dm_env — RL Environment Interfaces

### 2a. Gymnasium (Farama Foundation)

**Source**: [gymnasium.farama.org](https://gymnasium.farama.org/), [arxiv.org/abs/2407.17032](https://arxiv.org/abs/2407.17032)

#### What it describes

Gymnasium defines the standard API for RL environments. Its space type system is the closest existing "contract" for environment interfaces:

**Fundamental spaces**: `Box` (continuous vectors/matrices, bounded), `Discrete` (single integer from range), `MultiBinary` (binary vectors), `MultiDiscrete` (multiple independent discrete axes), `Text` (strings).

**Composite spaces**: `Dict` (named subspaces), `Tuple` (ordered subspaces), `Sequence` (variable-length repeats of one subspace), `Graph` (node+edge structure), `OneOf` (union of subspaces).

Each space provides:

- `shape` and `dtype` attributes
- `sample()` for random generation
- `contains(x)` for membership testing (the formal contract check)
- `to_jsonable()` / `from_jsonable()` for serialization

The `GoalEnv` extension (used by Gymnasium-Robotics) mandates a `Dict` observation with `observation`, `desired_goal`, `achieved_goal` keys — a structural contract for goal-conditioned tasks.

#### What it doesn't describe

- No metadata about what the dimensions _mean_ (joint angles vs. end-effector positions vs. pixel values). Spaces are structural, not semantic.
- No camera conventions, normalization metadata, or embodiment descriptions.
- No cross-environment compatibility checking. Spaces describe one environment in isolation.
- No dataset linkage. Spaces describe the live environment API, not stored data.

#### Community adoption

Dominant. 18M+ installs. Used by Stable Baselines3, CleanRL, Ray RLlib, and effectively all RL research. Gymnasium-Robotics (Farama Foundation) extends it for manipulation tasks (Fetch, Shadow Hand, etc.).

#### Already solving our problems?

**Partially.** Gymnasium's space type system solves the structural contract (shape, dtype, bounds) for RL environments. But the Physical AI interoperability problem is primarily _semantic_ (what do the dimensions mean? is this camera a wrist camera? is this joint ordering compatible?), which Gymnasium doesn't address. The space type system is a useful building block — any contract for RL-facing components should be compatible with Gymnasium spaces — but it doesn't solve cross-component compatibility.

### 2b. dm_env (DeepMind)

**Source**: [github.com/google-deepmind/dm_env](https://github.com/google-deepmind/dm_env)

Similar to Gymnasium but uses `dm_env.specs` (Array specs with dtype, shape, bounds, name) instead of spaces. Less adopted than Gymnasium. Used by Acme, dm_control. The `observation_spec()` and `action_spec()` methods return structured specs that mirror Gymnasium's spaces.

Not solving additional problems beyond Gymnasium. The community has largely converged on Gymnasium as the standard.

---

## 3. Cross-Embodiment Training Conventions

**Sources**: [Octo paper (RSS 2024)](https://octo-models.github.io/), [pi0 paper (arxiv 2410.24164)](https://arxiv.org/abs/2410.24164), [Open X-Embodiment (ICRA 2024)](https://arxiv.org/abs/2310.08864)

### How it works today

There is **no formal convention** for describing embodiment differences. Instead, each project uses ad-hoc approaches:

**pi0 zero-padding**: State and action vectors are padded to the dimensionality of the largest robot in the training set (18 dims). Smaller robots get zero-padded. This is a runtime implementation detail, not a declared metadata property.

**Open X-Embodiment (OXE) standardization**: 60+ datasets from 22 robot embodiments, standardized into RLDS format. Action is a 7-dim vector (x, y, z, roll, pitch, yaw, gripper) representing either absolute values, deltas, or velocities. The dataset records which interpretation applies, but this is per-dataset metadata, not a cross-dataset schema.

**Octo's approach**: Trains on OXE mixture of 25 datasets. Uses a small action head (~5% of weights) that can be swapped for fine-tuning to new embodiments. The model architecture handles embodiment differences; there's no metadata-driven compatibility layer.

**OpenPI per-robot configs**: Robot-specific conventions in `config.py` — joint angles in radians, gripper position in [0.0, 1.0], control frequencies of 20Hz (UR5e, Franka) or 50Hz (ALOHA). These conventions are documented in code comments, not machine-readable metadata.

### The gap

No project provides a machine-readable description of an embodiment that could be used to automatically determine compatibility. The information exists (action dimensions, joint semantics, control frequencies, camera positions) but is scattered across config files, code comments, and paper appendices.

### Is this being solved?

**Not yet, but pressure is building.** As cross-embodiment training scales (pi0.5 trains on 7+ robot configs, Octo on 25 datasets), the manual per-robot adapter pattern becomes increasingly expensive. However, no community has proposed a standard for machine-readable embodiment descriptions that would replace it.

---

## 4. Dataset Format Standards

### 4a. RLDS (Reinforcement Learning Datasets)

**Source**: [github.com/google-research/rlds](https://github.com/google-research/rlds), [Ramos et al. 2021](https://research.google/blog/rlds-an-ecosystem-to-generate-share-and-use-datasets-in-reinforcement-learning/)

**What it describes**: Episode-based storage format for sequential decision-making data, built on TensorFlow Datasets. Defines a schema for timesteps containing observations, actions, rewards, and metadata. Used by Open X-Embodiment (1M+ episodes, 22 embodiments).

**Schema**: Each dataset defines a `features` dictionary specifying field types and shapes. Standardized action schema for OXE is 7-dim (x, y, z, roll, pitch, yaw, gripper) with per-dataset interpretation (absolute, delta, or velocity).

**Limitations**: TensorFlow-native (PyTorch requires bridge libraries). Schema is per-dataset, not cross-dataset. No built-in compatibility checking between datasets or between datasets and models.

**Trajectory**: Being superseded by LeRobot format in the HuggingFace ecosystem. Robo-DM proposes EBML-based alternatives with better compression (up to 70x).

### 4b. LeRobot Dataset v3.0

**Source**: [huggingface.co/docs/lerobot/en/lerobot-dataset-v3](https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3)

**What it describes**: Self-describing dataset format with rich metadata in `meta/info.json`:

- **`robot_type`**: String identifier (aloha, koch, so100, etc.)
- **`features`**: Each feature declares `dtype`, `shape`, and `names` array (e.g., joint names for state vectors). Dot-notation keys: `observation.images.camera_name`, `observation.state`, `action`.
- **`fps`**: Recording frequency
- **`codebase_version`**: Format version tracking
- **`meta/stats.json`**: Per-feature statistics (mean, std, min, max) for normalization
- **`meta/tasks.jsonl`**: Natural-language task descriptions

**What it doesn't describe**:

- No formal link between dataset and compatible models (which models can train on this dataset?)
- No action space semantics beyond dimension names (is this an EE delta or absolute joint position?)
- No embodiment description beyond `robot_type` string
- No camera role annotations (world vs. robot-mounted)
- No inter-dataset compatibility metadata (can I mix this with another dataset?)

**Community tools gap** (from [Issue #2326](https://github.com/huggingface/lerobot/issues/2326)): Missing features include editing task descriptions after creation, adding computed features, and merging datasets with different feature keys. This suggests the community recognizes metadata limitations but is focused on tooling, not schema enrichment.

**Already solving our problems?** LeRobot's `info.json` is the closest existing format to what a "dataset manifest" would need. It captures structural metadata (shapes, types, feature names) but lacks semantic metadata (action space interpretation, camera roles, embodiment details, model compatibility). A contract system could extend `info.json` rather than replace it.

---

## 5. Manipulation Benchmark Frameworks

### 5a. ManiSkill / SAPIEN

**Source**: [github.com/haosulab/ManiSkill](https://github.com/haosulab/ManiSkill), [maniskill.readthedocs.io](https://maniskill.readthedocs.io/)

**Task specification**: Tasks defined as Python classes, registered by name (e.g., `PickCube-v1`). Each task supports configurable `obs_mode` (state, rgbd, pointcloud) and `control_mode` (joint position, EE pose, etc.). Some tasks support swappable embodiments via `robot_uids`.

**Observation/action spaces**: Uses Gymnasium `Dict` spaces. Observation structured as dictionary with `observation`, `desired_goal`, `achieved_goal` keys for goal-conditioned tasks. Action space is concatenation of per-controller action spaces.

**No metadata standard**: Task definitions are imperative (Python classes), not declarative (metadata files). No JSON/YAML schema for tasks, no machine-readable benchmark specification.

### 5b. robosuite

**Source**: [robosuite.ai](https://robosuite.ai/)

**Task specification**: Nine benchmark tasks, each as a Python class with MuJoCo backend. Configuration via constructor arguments, not metadata files.

**Observation space**: Modular and configurable — proprioception, camera RGB/depth. But configuration is embedded in code.

**No metadata standard**: "Hard-coded asset and task definitions" noted as a scalability limitation ([RoboVerse paper](https://arxiv.org/abs/2504.18904)).

### 5c. RoboVerse / MetaSim

**Source**: [github.com/RoboVerseOrg/RoboVerse](https://github.com/RoboVerseOrg/RoboVerse), [roboverse.wiki](https://roboverse.wiki/)

**What it describes**: Unified platform attempting to bridge multiple simulators (Isaac Sim, MuJoCo, SAPIEN) with a common task/robot/scene abstraction. Layered architecture: MetaSim (simulator abstraction) → RoboVerse (tasks, robots, scenes, datasets, benchmarks).

**Key concepts**: "State protocol" and "handler system" for cross-simulator abstraction. Tasks, robots, and scenes are separate concepts.

**Maturity**: Code released April 2025. Documentation is thin on the metadata format details. Too early to assess whether its abstractions will become a standard.

**Relevance**: RoboVerse is trying to solve simulator interoperability (same task, different physics backend) — a complementary problem to ours (same model, different datasets/robots). Worth monitoring.

---

## 6. KFP v2 Artifact Type System

**Source**: [kubeflow.org/docs/components/pipelines/user-guides/data-handling/artifacts/](https://www.kubeflow.org/docs/components/pipelines/user-guides/data-handling/artifacts/), [kubeflow-pipelines.readthedocs.io](https://kubeflow-pipelines.readthedocs.io/)

### What it describes

KFP v2 provides typed artifact classes (`Model`, `Dataset`, `Metrics`, `Artifact`) with automatic lineage tracking via ML Metadata (MLMD). Artifacts have `.uri`, `.path`, and `.metadata` attributes. Type annotations (`Input[Model]`, `Output[Dataset]`) trigger automatic artifact registration.

### What it doesn't describe

- **No semantic metadata schema**: Artifacts carry opaque key-value metadata. A `Model` artifact doesn't know its input shapes, action spaces, or normalization strategy. A `Dataset` artifact doesn't know its feature schema.
- **No compatibility checking**: KFP validates that a `Model` output connects to a `Model` input (type-level), but not that the model's action space matches the dataset's action dimensions (semantic-level).
- **No typed ports beyond artifact class**: Components declare `Input[Dataset]` and `Output[Model]`, but the inner structure is opaque. Two `Model` artifacts are interchangeable as far as KFP knows, even if they expect completely different input formats.

### Already solving our problems?

**No.** KFP's type system is too coarse for Physical AI compatibility checking. It provides the plumbing (artifact tracking, lineage, storage abstraction) but not the semantic layer. A contract system would give KFP artifacts richer type information — e.g., a `Model` artifact that declares its input shapes and compatible dataset features. This is a complementary relationship, not a competing one.

---

## 7. Agent Skills / Agent Plugins

**Source**: [agentskills.io/specification](https://agentskills.io/specification), [agent-plugins.org](https://agent-plugins.org/plugin-authors/skills)

### What it describes

A lightweight spec for packaging agent capabilities as `SKILL.md` files with YAML frontmatter. Progressive disclosure model: discovery (~100 tokens: name + description), activation (<5000 tokens: full instructions), execution (scripts/references loaded on demand).

Required fields: `name` (kebab-case, ≤64 chars), `description` (≤1024 chars). Optional: `license`, `compatibility`, `metadata` (arbitrary key-value), `allowed-tools` (experimental).

### Relevance as a pattern

Agent Skills demonstrates that a minimal metadata spec (2 required fields) with progressive disclosure and a directory convention can achieve wide adoption (20+ platforms adopted within ~1 year of release). Key design decisions:

- **Minimal required fields** reduce adoption friction
- **Arbitrary `metadata` map** provides extensibility without schema bloat
- **Convention over specification** for directory structure (scripts/, references/, assets/)
- **Validation tooling** (`skills-ref validate`) lowers the bar for conformance

Agent Plugins (`agent-plugins.org`) adds a composition layer on top — it defines where skills are discovered inside a plugin and how failures are isolated, without redefining the skill format itself. This is the kind of "lightweight spec that complements an existing one" pattern Frank mentioned.

### Already solving our problems?

**No** — different domain entirely (AI agent capabilities, not Physical AI artifacts). But the spec design pattern is directly relevant: minimal metadata, progressive disclosure, composability via overlay specs.

---

## 8. Existing Compatibility-Checking Tools

### What exists

- **Robot Data Audit (RDA)** ([pypi.org/project/robot-data-audit](https://pypi.org/project/robot-data-audit/)): Quality auditing tool for robot datasets. Latest version 0.7.2 (September 2026). Authored by Niu Su Tech. PyPI page was unavailable during research; actual checking capabilities unclear. This appears to be dataset quality auditing, not model-dataset compatibility checking.

- **LeRobot's built-in validation**: Dataset format validation (correct directory structure, parquet integrity, video sync). The v2.1→v3.0 conversion script validates format but not semantic compatibility. From the docs: "A successful load does not guarantee suitability for training, but it rules out a wrong directory hierarchy, missing metadata, and gaps in file numbering."

- **OpenPI's implicit checks**: Data conversion scripts validate that data can be converted to LeRobot format. Normalization stats computation catches missing or degenerate stats. But these are runtime checks embedded in training scripts, not standalone tools.

### What doesn't exist

**No tool was found that validates model-dataset or model-robot compatibility before training.** Specifically:

- No tool checks whether a dataset's action dimensions match a model's expected input
- No tool verifies camera name mappings between datasets and model configs
- No tool validates normalization strategy compatibility (quantile stats available when model expects quantile normalization)
- No tool checks joint ordering compatibility between simulator configs and dataset conventions

This gap is confirmed by the pattern across all surveyed projects: compatibility is enforced by convention (naming patterns, per-robot adapter code) and discovered by runtime error.

---

## 9. Summary: Landscape Assessment

### Where standards exist and work well

| Domain | Standard | Maturity | Notes |
| --- | --- | --- | --- |
| RL environment interface | Gymnasium spaces | High | Structural contracts (shape, dtype, bounds). Dominant. |
| Robot dataset storage | LeRobot v3.0 | Medium | Self-describing format with rich structural metadata. Growing fast. |
| Cross-embodiment dataset corpus | OXE / RLDS | Medium | Standardized 60+ datasets. But RLDS is being superseded. |
| Simulation asset validation | SimReady Foundation | Medium | Physics/material/packaging validation for USD assets. |
| Agent skill metadata | Agent Skills spec | Medium | Minimal, widely adopted pattern for capability description. |

### Where standards are absent or inadequate

| Gap | What's missing | Who might own it |
| --- | --- | --- |
| **Action space semantics** | Machine-readable description of what action dimensions mean (EE delta vs. absolute joint vs. velocity). Currently encoded in Python code or paper appendices. | Falls between LeRobot (dataset), OpenPI (model), and Gymnasium (environment). No single community owns it. |
| **Camera role metadata** | World camera vs. robot-mounted. Slot names. Mount points. Currently handled by manual CAMERA_MAP configuration. | Could extend LeRobot's `info.json` features or be a separate lightweight spec. |
| **Model-dataset compatibility** | No tool or schema validates that a model can train on a given dataset before launching a job. | Natural extension of LeRobot or a complementary spec. |
| **Embodiment description** | Machine-readable robot capabilities: joint count/types/limits, camera positions, action space conventions. URDF/MJCF describe geometry; nothing describes the ML-facing interface. | Falls between ROS/URDF (mechanical), Gymnasium (RL interface), and LeRobot (data). |
| **Cross-component normalization** | Is normalization baked into the checkpoint (ONNX) or external (stats.json)? Which strategy (quantile, z-score, identity)? No standard way to declare this. | Could extend model card metadata or be part of a compatibility spec. |

### The key finding

**No project currently provides machine-readable metadata for cross-component compatibility checking in Physical AI workflows.** The metadata needed exists in various forms (LeRobot's `info.json`, OpenPI's `config.py`, Gymnasium's spaces, URDF joint descriptions) but each covers only its own component in isolation. The gap is in the _relationships_ between components — can this model train on this dataset for this robot? — not in the individual component descriptions.

This gap falls between communities (LeRobot, ROS, OpenPI, Gymnasium) rather than being owned by any single one. This suggests either:

1. An extension to LeRobot's metadata (most natural home for dataset-model compatibility), or
2. A lightweight complementary spec (like agent-plugins.org complements agentskills.io) that adds cross-component compatibility annotations to existing formats.

Option 2 is lower risk if the spec stays minimal and references existing metadata rather than replacing it.
