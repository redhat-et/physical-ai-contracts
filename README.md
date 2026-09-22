# Open Physical AI Contracts

Exploring whether a contract/interoperability layer for Physical AI would deliver value, and if so, what form it should take.

## Status

**Research phase.** We are surveying existing schemas and metadata standards across the Physical AI stack, identifying gaps, and determining where new specification work adds value vs. where existing community efforts should be extended. See [Jira epic OCTOET-2176](https://redhat.atlassian.net/browse/OCTOET-2176) for tracked work.

## Problem

Physical AI workflows (data curation, fine-tuning, evaluation, deployment) involve multiple independently developed components: models, datasets, robot descriptions, inference servers, and adapters. Today, compatibility between these components is verified at runtime — often after minutes of setup — through cryptic error messages or, worse, silent misconfiguration.

Examples from real-world repos:

- A wrong camera-name mapping (`CAMERA_MAP`) causes a crash at the first training batch, after minutes of data loading
- Using quantile normalization on a dataset without `q01`/`q99` stats fails at first batch; using mean/std when quantiles were intended "succeeds" but silently degrades the policy
- Joint ordering differences between URDF, MJCF, and OpenUSD require hand-coded remapping arrays per robot per simulator ([Isaac Lab #7750](https://github.com/isaac-sim/IsaacLab/issues/7750))
- Every inference server (OpenPI, vLLM-Omni, SGLang) independently reimplements adapter classes for each robot — the same sign flips, gripper conversions, and camera remappings, hardcoded in Python

The hypothesis is that machine-readable metadata on artifacts can make compatibility checking and format adaptation automatic — so tooling handles what engineers currently do by hand. Whether "contracts" are the right mechanism, and what form they should take, is what the research phase needs to determine.

## Research findings

The `research/` directory contains our prior art survey and analysis. Key findings so far:

**No community has complete metadata for any artifact type.** Each community (LeRobot, OpenPI, GR00T, ROS 2, URDF, etc.) covers some properties and leaves others implicit or hardcoded. The gaps compound at component boundaries.

**Three categories of spec work emerge:**

- **New descriptor types** — for concepts no community defines today (e.g., declarative adapter mappings covering camera remapping + joint reordering + sign flips + gripper conversion; per-dimension action semantics; camera role vocabulary)
- **Extensions to existing specs** — filling identified gaps in community schemas (e.g., adding action semantics and embodiment references to LeRobot `info.json`; structured robotics fields in HF model/dataset cards; joint-to-action-index mapping annotations for URDF/OpenUSD)
- **Integration/packaging specs** — how and where metadata attaches to artifacts (e.g., OCI artifact labels, model registry metadata, server introspection APIs)

See `research/survey/descriptor-data-model.md` for the full descriptor type inventory, community coverage matrix, and gap analysis.

## Repository structure

```text
research/               — Background research and analysis
  survey/               — Prior art survey files (per-community, per-descriptor-type)
  diagrams/             — Architecture and ecosystem diagrams
specs/                  — Schema definitions (empty — waiting for research to complete)
  drafts/               — Work-in-progress schemas
sdk/                    — Python SDK (empty — waiting for design phase)
docs/                   — Documentation
```

## License

Apache 2.0
