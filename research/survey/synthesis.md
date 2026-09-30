# Prior Art Survey — Synthesis

**Date**: 2026-09-21 (updated; originally 2026-09-18)
**Input**: 5 domain surveys + deployment-topology analysis + community gap validation
**Purpose**: Cross-cutting analysis for OCTOET-2177

---

## The Central Finding

**The cross-artifact compatibility gap sits in the crack between communities.** Each community owns metadata for one artifact type, but no community owns the relationships between them:

| Question | Who owns it today |
| --- | --- |
| What does this model expect as input? | Nobody (framework-specific config files) |
| What does this dataset provide? | LeRobot (`info.json`) — but LeRobot-specific |
| What can this robot do? | ROS (URDF) + simulators (MJCF/USD) — but physics only, not ML-facing |
| Can this model train on this dataset? | **Nobody** |
| Can this model deploy to this robot? | **Nobody** |
| Is normalization handled correctly across model + dataset? | **Nobody** |

Multiple independent developers are building partial solutions (lerobot-doctor, knownrobot, raftaar, RDA), confirming this is a real unmet need — not a problem we imagined.

---

## Problem Validation

### Confirmed as real community problems (not just our sample)

1. **Normalization mismatches**: Multiple open LeRobot bugs — [#4415](https://github.com/huggingface/lerobot/issues/4415) (silently skipped for multi-dataset models), [#4647](https://github.com/huggingface/lerobot/issues/4647) (config bug causes empty normalization, reward drops from 118–199 to 4.8). This is an active, unresolved pain point within the LeRobot community itself.

2. **Camera mapping errors**: LeRobot has built partial tooling (`rename_map`, `validate_visual_features_consistency`) but it's reactive (catches at training time, not selection time) and buggy — [PR #4578](https://github.com/huggingface/lerobot/pull/4578) shows stored camera maps being silently dropped.

3. **Joint ordering mismatch**: Isaac Lab [#7750](https://github.com/isaac-sim/IsaacLab/issues/7750) confirms this as a live pain point. ROS-Industrial [proposed](https://github.com/ros-industrial/ros_industrial_issues/issues/52) standardizing joint naming but reached no consensus. No converter handles index ordering — they translate structure, not semantics.

4. **Action space fragmentation**: No agreed vocabulary. BridgeData2 (7-DOF EE delta), UMI (EE pose), DROID (multiple representations recorded simultaneously), GR00T (flat array with sidecar `modality.json`), OXE (normalized 7-DOF). Research proposals exist (CalibAll, UniAct, FAST+) but these are learned representations, not metadata standards.

### Confirmed as our-sample-only (engineers didn't use available tooling)

1. **Checkpoint loading semantics** (`--policy.path` vs `--policy.pretrained_path`): LeRobot-specific footgun. No external standard addresses it, but it's also not a cross-community problem — it's a LeRobot CLI design issue that should be fixed within LeRobot.

### Partially addressed by existing solutions

1. **Simulation-level interoperability**: Actively converging — REP-0158, `simulation_interfaces`, `ros2_control`, MuJoCo-in-Gazebo. Well covered at the control layer. The ML training layer remains unaddressed.

---

## What Exists and Should Not Be Reinvented

| Domain | Standard(s) | Maturity | Notes |
| --- | --- | --- | --- |
| Dataset structural validation | lerobot-doctor, RDA | Early but active | Contribute, don't compete |
| Dataset metadata (structural) | LeRobot v3.0 `info.json` | Medium | De facto standard, 16K+ datasets |
| Robot kinematics/geometry | URDF + MJCF + USD | High | Well covered, actively converging |
| Simulation asset validation | SimReady Foundation | Medium | Physics/geometry validation |
| Cross-simulator asset format | REP-0158 (OpenUSD) | Draft | Joint naming, camera conventions, vendor isolation |
| Simulator control API | ROS 2 `simulation_interfaces` | Growing | Cross-simulator spawn/reset/step |
| Hardware abstraction | ros2_control | High | Same controller stack across sim + real |
| RL environment API | Gymnasium spaces | High | De facto standard for RL |
| Agent capability metadata | agentskills.io + agent-plugins.org | Medium | Pattern exemplar, not domain-relevant |
| Dataset discoverability | HF dataset cards + Croissant | High | Discovery, licensing, provenance |
| Model discoverability | HF model cards | High | Discovery, not operational metadata |
| Inference deployment profiles | NVIDIA NIM manifests | Medium | NVIDIA-specific, LLM-focused |

---

## Genuine Gaps — Where No Standard Exists

Ordered by concreteness and tractability (most actionable first):

### Gap 1: Normalization contract

**Problem**: No metadata declares whether normalization is baked into a checkpoint, external (which stats format?), or computed on load. The strategy (quantile, z-score, identity) and the stats availability (q01/q99 vs. mean/std only) are split across model config and dataset stats with no linking.

**Evidence**: Multiple active LeRobot bugs. Affects every training run. Silent failures (policy trains but performs badly).

**Why it's in the crack**: Normalization spans model (what strategy does it expect?) and dataset (which stats are available?). Neither side owns the declaration.

**Scope**: Small. A handful of metadata fields on both sides.

### Gap 2: Camera role semantics

**Problem**: No standard vocabulary for camera roles in ML context. "wrist_camera", "agentview", "observation.images.wrist", "exterior_image_1_left" — ad hoc names with no alignment. The three-way mapping (policy slots ↔ robot cameras ↔ dataset camera names) is manual.

**Evidence**: LeRobot's `rename_map` exists specifically because this problem is real. But it's a runtime workaround, not a declarative standard. NVIDIA GR00T adds `modality.json` with video key remapping — a parallel workaround.

**Why it's in the crack**: Camera role is a property that spans robot (where is the camera?), dataset (what was it called in the recording?), and model (what input slot does it feed?). No community owns all three.

**Scope**: Medium. Requires a small vocabulary (world, wrist, head, overhead, ego) plus a mapping mechanism.

### Gap 3: Action space semantics declaration

**Problem**: No machine-readable way to declare what an action vector means. Is `action[0:3]` a Cartesian delta, absolute EE position, or joint velocity? What coordinate frame? What units? Currently encoded in Python code or paper appendices.

**Evidence**: OXE standardized on 7-DOF EE delta but this doesn't cover humanoids (29+ DOF), whole-body controllers, or joint-space actions. GR00T's `modality.json` adds index-based splitting but no semantics. Multiple research papers (CalibAll, UniAct) attempt to solve this, confirming it's an open problem.

**Why it's in the crack**: Action semantics are a model-robot interface concern. Models define expected actions; robots define executable actions. Neither model metadata (ONNX, MLflow, HF cards) nor robot descriptions (URDF, MJCF) currently describe this interface.

**Scope**: Large. Requires cross-community vocabulary agreement. Hardest gap to fill.

### Gap 4: Joint-to-action-index mapping

**Problem**: Joint names exist (URDF, REP-0158 `ros:joint:name`). But the mapping from joint names to policy action-space indices is implicit. When transferring a policy trained in Isaac Lab to MuJoCo, which action dimension controls which physical joint? Currently handled by hand-coded remapping arrays.

**Evidence**: Isaac Lab [#7750](https://github.com/isaac-sim/IsaacLab/issues/7750). The RDP handles this with explicit `mujoco_to_isaaclab` / `isaaclab_to_mujoco` arrays per robot.

**Why it's in the crack**: Joint names are an asset concern (REP-0158). Action indices are a policy concern (training framework). The mapping between them is owned by nobody.

**Scope**: Medium. Could potentially extend REP-0158 with an `ml:action_index` property, or be declared in a policy-side manifest.

### Gap 5: Cross-artifact compatibility checking

**Problem**: No tool validates model-dataset-robot compatibility before training. The metadata to make this check exists scattered across `config.json`, `info.json`, URDF, etc. — but no schema normalizes it and no tool compares it.

**Evidence**: lerobot-doctor checks dataset-policy compatibility for 4 hardcoded policies. knownrobot's `robot-skill.yaml` is the closest to a schema-driven approach but has zero traction. No general-purpose tool exists.

**Why it's in the crack**: This is inherently a cross-artifact concern. Each artifact's metadata is owned by a different community.

**Scope**: Large. Depends on gaps 1–4 being resolved first — you can't check compatibility without declared semantics.

---

## Existing Work Closest to Our Problem

| Project | What it does | Limitations | Stars |
| --- | --- | --- | --- |
| **lerobot-doctor** | 12 diagnostic checks on LeRobot datasets + policy compatibility gating for 4 policies | Hardcoded policy list, no schema-driven approach | 40 |
| **knownrobot** | `robot-skill.yaml` manifest with JSON Schema validation for policy-robot compatibility | SO-100/SO-101 specific, zero adoption | 0 |
| **raftaar** | Predicts policy failure modes from dataset distribution analysis | Research prototype, no cross-dataset validation | 0 |
| **RDA** | 21-metric dataset quality audit | Quality audit, not compatibility checking | 0 |
| **GR00T `modality.json`** | Semantic array splitting + video key remapping on top of LeRobot | NVIDIA-specific, LeRobot v2 only | — |
| **LeRobot `rename_map`** | Manual camera name remapping at training time | Reactive, buggy, cameras only | — |

---

## Deployment-Level Gaps (from deployment-topology.md, 2026-09-21)

The deployment topology research revealed additional gaps at the **serving and runtime** layer that complement the training-time gaps above. These were validated against active community efforts.

### Gap 6: Adapter code duplication across servers

**Problem**: The Embodiment Adapter is physically duplicated as hardcoded Python across 3+ server implementations (OpenPI `DroidInputs`, vLLM-Omni `DroidTransform`, SGLang checkpoint-config). Each independently reimplements the same camera remapping, joint sign flips, gripper conversion, and state vector assembly — with no shared declarative descriptor.

**Evidence**: vLLM-Omni's quantile normalization initially returned `None` for pi0.5 ([PR #6950](https://github.com/vllm-project/vllm-omni/pull/6950)). Same checkpoint, silently different outputs. Observation key whitelisting silently drops unrecognized fields.

**Community status**: **Nobody is standardizing this.** Rosetta ([ros-physical-ai/demos](https://github.com/ros-physical-ai/demos), v0.2.0) standardizes the ROS 2 ↔ LeRobot transport bridge but not adapter logic. LEAPP tensor semantics (kind annotations, element names) are the only declarative metadata but NVIDIA-only. ISO/WD 26264-1 targets dataset-level semantics, not runtime adapter logic.

**Relationship to other gaps**: This is the deployment-side manifestation of Gap 3 (action space semantics). A declarative adapter descriptor would address both — it would describe what a checkpoint expects (Gap 3) *and* eliminate per-server reimplementation (Gap 6).

### Gap 7: Inference wire format

**Problem**: Three incompatible msgpack numpy serialization dialects exist for encoding arrays over WebSocket: `openpi-client` markers (`__ndarray__`), `vLLM-native` markers (`nd` + `kind`), and the `msgpack-numpy` PyPI package markers (`nd` + `kind` with empty `kind`). Clients that pack arrays in one dialect fail on servers expecting another.

**Evidence**: vLLM-Omni had to patch its decoder ([PR #6051](https://github.com/vllm-project/vllm-omni/pull/6051)) to accept all three formats inbound. The robotics-playground vendored `msgpack_numpy.py` with auto-detection — a correct but fragile workaround.

**Community status**: **No active standardization effort.** [YEP 110](https://yeps.yaq.fyi/110/) is a formal ndarray-over-msgpack extension type spec from the yaq instrumentation community — rigorous but with zero robotics awareness. [msgpack PR #267](https://github.com/msgpack/msgpack/pull/267) proposes native typed N-D array support in the msgpack spec but is stalled (spec maintainers "very conservative"). OpenPI's encoding is becoming a de facto standard by adoption gravity (SGLang and vLLM-Omni both implement compatibility endpoints), but nobody is writing it down as a spec.

**Scope**: Small-medium. A concise wire format spec (or endorsement of YEP 110) would eliminate an entire class of integration bugs.

### Gap 8: Data recording → training dataset conversion

**Problem**: 6+ independent ROSbag-to-LeRobot converters exist, each with different resampling strategies for multi-rate sensor data (30 Hz camera, 100 Hz joints). No metadata records which resampling strategy was used, making dataset provenance opaque.

**Evidence**: [LeRobot ROS 2 RFC #4368](https://github.com/huggingface/lerobot/issues/4368) surveys the fragmentation. None of the 6 bridges installs cleanly against LeRobot 0.6.x.

**Community status**: **Early-stage effort, no code merged.** RFC #4368 proposes 5 deliverables (bidirectional converter, conventions doc, policy deployment node) but is still at the proposal stage. Intel's Physical AI Studio has an open feature request for rosbag import. RLDS (Google) is still maintained but momentum has shifted to LeRobot — they coexist with no convergence path. [Croissant 1.1](https://arxiv.org/html/2403.19546v1) (MLCommons) has a domain-extension mechanism that could theoretically support robotics-specific dataset semantics, but no robotics extension has been proposed.

**Scope**: Medium. The resampling strategy question (how to align multi-rate sensor streams into synchronized training frames) is the hardest unsolved design problem — it's not just a format conversion.

### Deployment gaps: summary table

| Gap | Severity | Community coverage | Status |
| --- | --- | --- | --- |
| **6. Adapter code duplication** | High | Rosetta (transport only), LEAPP semantics (NVIDIA-only) | **Unaddressed** |
| **7. Inference wire format** | High | YEP 110 (no robotics awareness), OpenPI de facto | **Unaddressed** |
| **8. Data recording → dataset** | High | LeRobot RFC #4368 (early, no code) | **Early-stage** |
| Image preprocessing divergence | Medium | None | Unaddressed |
| Two-protocol reality (WebSocket vs ROS 2) | Medium | GSoC 2026 trajectory upsampling (partial) | Partial |
| Missing ros2_control inference controller | Medium | MoveIt Pro ExecutePolicy (closest) | Partial |

---

## Combined Gap Ranking

Merging the training-time gaps (1–5) with deployment-time gaps (6–8), ranked by a combination of severity, tractability, and community coverage:

| Rank | Gap | Scope | Why tractable | Community coverage |
| --- | --- | --- | --- | --- |
| 1 | **Normalization contract** (Gap 1) | Small | A few metadata fields; active bugs prove need | None |
| 2 | **Adapter descriptor / action semantics** (Gaps 3+6) | Large | Highest impact; LEAPP partial prior art | LEAPP (NVIDIA-only) |
| 3 | **Camera role semantics** (Gap 2) | Medium | Small vocabulary + mapping mechanism | LeRobot `rename_map` (buggy) |
| 4 | **Inference wire format** (Gap 7) | Small | Spec exists (YEP 110); just needs adoption | YEP 110 (no awareness) |
| 5 | **Data recording → dataset** (Gap 8) | Medium | LeRobot RFC in progress; could contribute | RFC #4368 (early) |
| 6 | **Joint-to-action-index mapping** (Gap 4) | Medium | Could extend REP-0158 | None |
| 7 | **Cross-artifact compatibility** (Gap 5) | Large | Depends on gaps 1–4 being resolved | lerobot-doctor (partial) |

---

## Strategic Options

### Option A: Extend LeRobot metadata upstream

Add standardized fields to `info.json` (action space semantics, camera roles, normalization contract) and to model `config.json` (expected inputs, normalization strategy, compatible embodiments). Contribute directly to LeRobot.

**Pros**: Highest adoption potential (16K+ datasets). Builds on existing infrastructure. LeRobot v0.7 roadmap shows openness to interoperability (ROS2 exploration, multi-dataset support).

**Cons**: Framework-specific. Doesn't help ONNX-exported models, Isaac Lab configs, or non-LeRobot datasets. LeRobot maintainers may resist scope expansion.

### Option B: Lightweight complementary spec

A small, portable metadata format (like `modality.json` or `robot-skill.yaml` but standardized) that can be dropped alongside any artifact to declare its ML-facing interface. References existing metadata (URDF joint names, LeRobot feature names, Gymnasium space types) rather than replacing it.

**Pros**: Framework-agnostic. Can complement LeRobot, ONNX, GR00T simultaneously. Matches the agent-plugins.org pattern (overlay on existing specs). Scope can be narrow (start with normalization contract, expand later).

**Cons**: Adoption risk for any new spec. Must prove value before anyone adds it to their artifacts. Cold-start problem.

### Option C: Compatibility-checking tool on native metadata

Skip the new metadata format. Build a tool that reads native `config.json`, `info.json`, ONNX metadata, URDF etc. and performs compatibility checks directly. Each format gets an adapter.

**Pros**: Zero adoption barrier — works with artifacts as they exist today. Immediately useful.

**Cons**: Fragile to format changes. Framework-coupled. Adapters must be maintained per format version. Doesn't improve the metadata ecosystem — just papers over the cracks.

### Option D: Contribute to multiple communities simultaneously

Don't create a single new spec. Instead, propose specific metadata additions to each community: normalization fields in LeRobot `info.json`, camera role vocabulary for REP-0158, action-space declaration in Gymnasium. Coordinate the contributions so they reference each other.

**Pros**: No new spec adoption risk. Builds on existing communities. Each contribution is small and focused.

**Cons**: Coordination overhead. Slow (each community has its own review process). No single artifact ties the contributions together. Risk of contributions being accepted in different forms or timelines.

---

## Recommendation (Hypothesis, Not Decision)

**Start with Option A (LeRobot metadata extension) for the dataset side, combined with Option B (lightweight spec) for the model-robot interface.** Rationale:

1. LeRobot is the dominant dataset format and is actively evolving. Metadata enrichment has a clear path to adoption.
2. The model-robot interface gap (action semantics, joint mapping, camera roles) spans communities and needs a small bridging spec.
3. The normalization contract is the smallest-scope, highest-evidence problem — start there as a proof of concept for either approach.

**This recommendation should be validated by**:

- Checking LeRobot maintainer receptiveness (GitHub issue or discussion)
- Confirming that knownrobot's `robot-skill.yaml` isn't already gaining traction
- Talking to practitioners (not just reading code) about which pain points matter most

---

## Sources

All citations are inline in the domain survey documents:

- `research/survey/model-metadata.md`
- `research/survey/dataset-metadata.md`
- `research/survey/robot-environment.md`
- `research/survey/workflow-interop.md`
- `research/survey/community-solutions.md`
- `research/survey/deployment-topology.md`

### Additional sources for deployment-level gaps (2026-09-21)

- [Rosetta v0.2.0](https://github.com/ros-physical-ai/demos) — ROS 2 ↔ LeRobot bridge, schema v2
- [YEP 110](https://yeps.yaq.fyi/110/) — N-dimensional array msgpack extension type spec
- [msgpack PR #267](https://github.com/msgpack/msgpack/pull/267) — Proposed native ndarray support
- [LeRobot RFC #4368](https://github.com/huggingface/lerobot/issues/4368) — ROS 2 integration RFC
- [Croissant 1.1](https://arxiv.org/html/2403.19546v1) — MLCommons dataset metadata with domain extensions
- [ISO/WD 26264-1](https://arxiv.org/abs/2606.19769) — Humanoid robot dataset standard (early)
- [ROS2SmolVLA](https://arxiv.org/abs/2608.23320) — SmolVLA + UR robot ROS 2 integration
