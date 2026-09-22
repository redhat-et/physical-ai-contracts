# Prior Art Survey — Index & Summary

**Date**: 2026-09-22
**Jira**: [OCTOET-2177](https://redhat.atlassian.net/browse/OCTOET-2177) — Schema Research & Prior Art Survey
**Status**: 7 of 8 Definition of Done items complete. Remaining: peer review.

## Executive Summary

We surveyed 20+ existing schemas and metadata standards across the Physical AI stack to understand what metadata exists, where the gaps are, and what form new specification work should take. The survey covers model metadata, dataset metadata, robot/environment descriptions, inference servers, data conversion pipelines, workflow composition, and Red Hat platform integration (RHOAI Hubs, Model Registry).

### Key findings

1. **No community has complete metadata for any artifact type.** Each covers some properties and leaves others implicit or hardcoded. The gaps compound at component boundaries.

2. **Two novel descriptor types address the biggest structural gaps:**
   - **MappingDescriptor** (inference side) — camera remapping, joint reordering, sign flips, gripper conversion. Every deployment needs this; all communities hardcode it in Python.
   - **DataConversionDescriptor** (training side) — same mapping vocabulary applied to dataset format conversion. OXE's 70+ per-dataset Python transforms demonstrate the problem at scale.

3. **Joint ordering is the fundamental interoperability problem.** URDF, MJCF, and OpenUSD each assign different orderings for the same robot (confirmed: Isaac Lab #7750). No format maps joint names to policy action-vector indices.

4. **Training config is richer than checkpoint metadata.** The full data pipeline (transforms, normalization, camera selection, action representation) is specified during training but lost at checkpoint save time. Inference servers must independently reconstruct what training did.

5. **Normalization formulas and stats formats have converged.** OpenPI and LeRobot carry near-identical stats (mean/std/min/max/q01/q99). Z-score and quantile formulas are universal.

6. **Camera role vocabulary is universally needed, nowhere provided.** Every community encodes role in key names by convention. A small enum {wrist, world, head, ego, overhead} is low-hanging fruit.

7. **LeRobot v3 `info.json` is the de facto dataset metadata standard** (16K+ datasets), but it's a Python dataclass convention, not a broader HF or community standard.

8. **SGLang's `/v1/actions/metadata` is the only server introspection API.** No other inference server exposes structured metadata about its capabilities.

9. **RHOAI platform components can store but not validate Physical AI metadata.** Model Registry supports JSON custom properties; EvalHub/TrainingHub pass opaque file paths. A contract schema would enable structured validation.

### Three categories of spec work

**Category A — New descriptor types** (no community defines these):

- MappingDescriptor / DataConversionDescriptor (declarative adapter logic)
- ActionDescriptor with per-dimension semantics (extending LEAPP TensorSemantics)
- Camera role vocabulary
- Composite ObservationDescriptor

**Category B — Extensions to existing community specs**:

- LeRobot `info.json` — add action semantics, embodiment reference, camera roles
- HF model/dataset cards — structured robotics metadata fields
- URDF/OpenUSD — joint-to-action-index mapping for ML
- SGLang metadata endpoint — add normalization, embodiment compatibility
- Checkpoint metadata — preserve training lineage

**Category C — Integration/packaging specs**:

- OCI artifact metadata for models
- KubeFlow ModelRegistry structured properties
- Server introspection API standard
- Single-source-of-truth packaging (same descriptor for record, convert, deploy)

### Descriptor data model

The survey informed a bottom-up descriptor data model with 14 types organized across inference and training:

| Tier | Inference side | Training side |
| --- | --- | --- |
| Leaf | ActionDescriptor, CameraDescriptor, TransformDescriptor, ProprioceptionDescriptor | (shared) |
| Composite | ActionChunkDescriptor, ObservationDescriptor | (shared) |
| Artifact | CheckpointDescriptor, InferenceServerDescriptor, EmbodimentDescriptor, RobotInterfaceDescriptor, MappingDescriptor | DatasetDescriptor, EpisodeDescriptor, TrainingConfigDescriptor, DataConversionDescriptor |

Full details: [descriptor-data-model.md](descriptor-data-model.md)

---

## File Index

### Main output

| File | Lines | Content |
|---|---|---|
| [descriptor-data-model.md](descriptor-data-model.md) | 590 | **Primary deliverable.** 14 descriptor types with survey-informed properties, composition map, coverage matrix (descriptor × community), cross-cutting findings, and spec work recommendations. |

### Per-descriptor-type surveys

These survey each descriptor type across 8-10 communities, documenting what metadata exists, what's implicit, and what's missing.

| File | Lines | Descriptor types covered |
| --- | --- | --- |
| [descriptor-survey-action.md](descriptor-survey-action.md) | 537 | ActionDescriptor, ActionChunkDescriptor |
| [descriptor-survey-observation.md](descriptor-survey-observation.md) | 758 | ObservationDescriptor, CameraDescriptor, ProprioceptionDescriptor |
| [descriptor-survey-checkpoint-server.md](descriptor-survey-checkpoint-server.md) | 562 | CheckpointDescriptor, InferenceServerDescriptor |
| [descriptor-survey-embodiment-robot.md](descriptor-survey-embodiment-robot.md) | 504 | EmbodimentDescriptor, RobotInterfaceDescriptor |
| [descriptor-survey-transform-mapping.md](descriptor-survey-transform-mapping.md) | 642 | TransformDescriptor, MappingDescriptor |
| [descriptor-survey-dataset.md](descriptor-survey-dataset.md) | 484 | DatasetDescriptor, EpisodeDescriptor |
| [descriptor-survey-training-conversion.md](descriptor-survey-training-conversion.md) | 646 | TrainingConfigDescriptor, DataConversionDescriptor |

### Per-community / per-topic surveys

These survey specific communities or topics in depth, covering standards not fully captured by the descriptor-type surveys.

| File | Lines | Content |
| --- | --- | --- |
| [model-metadata.md](model-metadata.md) | 292 | ONNX, MLflow, HF model cards, NIM manifest, SafeTensors, LeRobot config — model packaging standards |
| [robot-environment.md](robot-environment.md) | 513 | URDF, MJCF, OpenUSD, REP-0158, SimReady, ros2_control, Gymnasium, MuJoCo Menagerie — robot/environment description standards. Includes joint-ordering and camera-mapping problem analysis. |
| [workflow-interop.md](workflow-interop.md) | 296 | OpenPI protocol, KFP v2 artifact types, agentskills.io, Gymnasium, manipulation benchmarks |
| [rhoai-hubs.md](rhoai-hubs.md) | 364 | RHOAI SDG Hub, EvalHub, Training Hub, Model Registry, vLLM-Omni, cross-hub data flows |

### Analysis & synthesis

| File | Lines | Content |
| --- | --- | --- |
| [synthesis.md](synthesis.md) | 270 | Cross-cutting gap analysis with problem validation, gap ranking (8 gaps), community momentum assessment |
| [deployment-topology.md](deployment-topology.md) | 326 | How logical functions (policy server, adapter, controller) map to physical deployment units |
| [data-formats-at-interfaces.md](data-formats-at-interfaces.md) | 511 | End-to-end data format walkthrough for one scenario (LIBERO dataset → pi0.5 fine-tuning → OpenPI serving → Franka Panda) |
| [implementation-comparison.md](implementation-comparison.md) | 211 | Side-by-side comparison tables of Policy Server, Embodiment Adapter, Robot Controller implementations across communities |
| [community-solutions.md](community-solutions.md) | 365 | Validation that observed interop problems aren't already solved — tracks LeRobot bugs, lerobot-doctor, knownrobot, Rosetta, RDA |

---

## Communities surveyed

LeRobot, OpenPI, SGLang, vLLM-Omni, GR00T, LEAPP, ROS 2 / ros2_control, URDF, MJCF, OpenUSD / SimReady, REP-0158, Isaac Lab, OXE (Open X-Embodiment), Croissant (MLCommons), Rosetta, DROID, LIBERO, BridgeData2, MLflow, ONNX, HF Hub, NIM, SafeTensors, KFP v2, Gymnasium, MuJoCo Menagerie, RHOAI (SDG Hub, EvalHub, Training Hub, Model Registry).

## Remaining work

- **Peer review** by Ryan Malone (OCTOET-1924, skills alignment) and Evgenii Dushkin (OCTOET-1933, training pipelines) — DoD item 8
