# SimReady Foundation — Analysis for Open Physical AI Contracts

**Date**: 2026-09-15
**Source**: <https://nvidia.github.io/simready-foundation/latest/>
**Repo**: <https://github.com/NVIDIA/simready-foundation> (Apache 2.0, created 2026-03-03)
**Spec version reviewed**: 2026.06.0

---

## What SimReady Is

SimReady Foundation is an open specification layered on top of OpenUSD that defines what properties simulation assets must have to work correctly in different runtimes (PhysX, Isaac Sim, Newton). NVIDIA publishes it under Apache 2.0.

### Three-layer schema

1. **Requirements** — atomic, machine-checkable rules on USD prims. Each has a unique ID (e.g., `RB.COL.001`) and inspects specific USD attributes or APIs. Example: "every rigid shape must carry a `CollisionAPI`."

2. **Features** — bundles of requirements for a use case. Each feature lists its requirement IDs and declares dependencies on other features. Features resolve recursively: validating feature A automatically validates all features it depends on. Features exist in **variants** by runtime: `_BASE_NEUTRAL` (pure USD), `_BASE_PHYSX`, `_ROBOT_CORE_ISAAC`.

3. **Profiles** — bundles of features for a complete simulation scenario. A profile pins specific feature versions and represents "everything this asset needs to run in environment X." Profile versions are immutable once published.

### Defined profiles (as of 2026.06.0)

| Profile | Target |
| --- | --- |
| Prop-Robotics-Neutral | Props/objects, runtime-agnostic |
| Prop-Robotics-Physx | Props for PhysX runtime |
| Prop-Robotics-Isaac | Props for Isaac Sim |
| Robot-Body-Neutral | Articulated robots, runtime-agnostic |
| Robot-Body-Runnable | Robots with runnable articulation |
| Robot-Body-Isaac | Robots targeting Isaac Sim |

Profile naming convention: `<AssetType>-<Domain>-<Runtime>`.

### Defined features

| ID | Name | Domain |
| --- | --- | --- |
| FET_000 | Core - Base | Foundation |
| FET_001 | Minimal - Base | Foundation |
| FET_002 | Posable Bodies - Base | Physics |
| FET_003 | RBD Physics - Base | Physics |
| FET_004 | Simulate Multi-Body Physics - Base | Physics |
| FET_005 | Simulate Grasp Physics - Base | Physics |
| FET_006 | Materials - Base | Visualization |
| FET_007 | Non-Visual Materials - Base | Visualization |
| FET_011 | Semantic Labels - Base | Perception |
| FET_100 | IsaacSim Composition | Isaac Sim |
| FET_021 | Core Robot | Robot |
| FET_022 | Driven Joints | Robot |
| FET_023 | Robot Materials - Isaac Sim | Robot |
| FET_024 | Base Articulation | Robot |
| FET_030 | Packaging Core | Packaging |
| FET_031 | Package Self Contained | Packaging |
| FET_032 | Packaging Introspection | Packaging |
| FET_033 | SimReady Packaging | Packaging |

### Capability domains (grouping requirements)

Core, Visualization, Isaac Sim, Physics Bodies, Hierarchy, Non-Visual Sensors, Semantic Labels, Packaging.

### Feature adapters

A conversion framework that transforms assets between profiles (e.g., Neutral → PhysX). Adapters apply only the mutations needed for features that differ. Invoked via `workspace upgrade --output_profile <target>`.

### Vendor-isolation pattern

Attributes use colon-delimited namespace prefixes: `simready:`, `physx:`, `isaac:`. Third parties register their own prefixes (e.g., `acmeCable:`). This prevents naming collisions and keeps vendor-specific extensions isolated.

### Tooling

- `workspace validate <asset>` — validates a USD asset against a profile
- `workspace upgrade --output_profile <target>` — transforms between profiles
- Python validators via `@register_rule()` decorator
- Designed for CI/CD integration

---

## Relevance to Our Contracts

### Direct overlap: RobotManifest and EnvironmentManifest

SimReady defines exactly what our robot and environment manifests would need to capture about physical assets:

- Joint hierarchies, driven joints, articulation roots
- Collider properties, rigid body physics, grasp physics
- Material properties (visual and non-visual / friction, density)
- Semantic labels for ML perception pipelines
- Sensor placement
- Asset packaging and self-containment

**Design implication**: Our RobotManifest should reference a SimReady profile for the asset layer rather than reinventing physics/geometry validation. Our manifest adds the ML-facing contract on top (see "What SimReady doesn't cover" below).

Similarly, EnvironmentManifest references SimReady for scene asset validation. We add world camera positions/roles, task-relevant object metadata, and supported simulator declarations.

### Architectural patterns worth adopting

1. **Three-layer composition** (requirements → features → profiles) is well-designed. Our schemas could follow a similar layered validation approach: atomic checks compose into capability bundles, which compose into full compatibility profiles.

2. **Runtime variants** (`_NEUTRAL` / `_PHYSX` / `_ISAAC`) parallel our problem with normalization variants (baked-into-ONNX vs external stats.json) and action space conventions (relative EE deltas vs absolute poses). The variant + adapter pattern is directly transferable.

3. **Immutable versioned profiles** with semantic versioning match our planned Kubernetes-style API versioning (v1alpha1, v1beta1, v1).

4. **CLI pattern** (`workspace validate`, `workspace upgrade`) validates our plan for `pai validate` / `pai check-compat`.

5. **Vendor namespace prefixes** for extensibility — our controlled vocabulary for joint semantics and action spaces could use a similar prefix scheme.

### Complementary, not competing

SimReady and our contracts sit at different levels:

| Question | SimReady | Our contracts |
| --- | --- | --- |
| Is this robot USD valid for Isaac Sim? | Yes | No (not our scope) |
| Can this model train on this dataset? | No | Yes |
| Does this model's action space match this robot? | No | Yes |
| Is the camera mapping correct for this policy? | No | Yes |
| Which normalization does this checkpoint use? | No | Yes |
| Can I convert this asset from MuJoCo format to Isaac Sim? | Yes | No |
| Will this benchmark run with this model on this robot? | No | Yes |

SimReady defines the "nouns" (how to describe a robot/environment for simulation). Our contracts define the "verbs" (can these components work together for training/evaluation/deployment?).

---

## What SimReady Doesn't Cover

These are gaps where our contracts provide unique value:

1. **Models and training** — no concept of ML model manifests, checkpoint formats, normalization strategies, loading semantics (`--policy.path` vs `--policy.pretrained_path`), or action chunking parameters.

2. **Datasets** — no dataset manifests, feature schemas (image shapes, state vectors, action dimensions), stats availability (q01/q99, mean/std), episode structure, or format versioning (LeRobot v2.1 vs v3.0).

3. **Benchmarks** — no evaluation protocol definitions, success criteria, task instruction sets, or episode count requirements.

4. **Cross-artifact compatibility checking** — SimReady validates a single asset against a profile. It does not check whether two different artifacts are compatible with each other. The model-dataset, model-robot, and dataset-robot compatibility checks are our core value proposition.

5. **Action/observation semantics** — SimReady knows about joints and physics but not about action spaces (7-DOF EE delta, 10-DOF absolute pose, 29-DOF whole-body), observation vectors, or the policy-level interface between a model and a robot.

6. **Camera mapping** — SimReady validates that cameras exist on USD prims with correct optical frame conventions. It does not address the three-way relationship between policy camera slots, robot-mounted cameras, and dataset camera names (the CAMERA_MAP problem).

7. **Workflow composition** — no concept of KFP pipeline validation, typed component ports, or blueprint composition. SimReady operates on individual assets, not on workflows that connect multiple components.

8. **Training paradigm metadata** — no distinction between behavioral cloning, RL, vision SFT, or imitation learning. No concept of whether a simulator is needed in the training loop or only for evaluation.

---

## Relationship to REP-0158

SimReady does not reference REP-0158 (OpenUSD conventions for simulation asset interoperability, by Open Robotics / NVIDIA / Robotec.ai). The two are parallel efforts with overlapping scope:

- **REP-0158** focuses on ROS integration: `ros:joint:name`, ROS topic/service/action schemas on USD prims, optical frame conventions (USD -Z vs ROS +Z).
- **SimReady** focuses on simulation runtime readiness: physics correctness, materials, packaging, runtime-specific adaptations.

Both standardize OpenUSD for simulation, but from different angles (ROS ecosystem vs simulation runtime). SimReady is more mature (versioned spec, validation tooling, published profiles). For our contracts, SimReady is the more actionable reference, while REP-0158 remains relevant for ROS-facing joint naming conventions.

---

## Open Questions

1. How stable is SimReady's schema? The spec is at version 2026.06.0 and several features are at 0.1.0 — still early.
2. Will SimReady expand to cover sensor definitions (cameras, LiDAR, IMU) with enough detail for our camera mapping use case, or will that remain a gap?
3. Newton (NVIDIA's new physics engine) support is listed as "planned" — will it follow the same variant pattern as PhysX?
4. What is the governance model? The repo is under NVIDIA's GitHub org with Apache 2.0 license, but no external contributors or governance structure is documented.
5. How does SimReady relate to NVIDIA's internal asset pipeline (Omniverse, Isaac Sim asset store)? The `omni:simready:*` → `simready:*` migration suggests it evolved from an internal tool.
