# Open Physical AI Contracts — Agent Guide

## Purpose

This repo explores whether a contract/interoperability layer for Physical AI would deliver value, and if so, what form it should take. It contains research, prior art analysis, and (eventually) specifications and code.

We are in the **research phase**. No design decisions have been made. The goal is to understand the problem space before proposing solutions.

## Target Outcome

The friction and error potential in Physical AI workflows (data curation, fine-tuning, evaluation, deployment) should be drastically reduced. The target user experience:

- Select datasets suitable for fine-tuning a robotic policy for a particular use case, environment, and embodiment — in a few clicks, not by reading format specs
- Run synthetic data generation workflows (simulation + domain randomization) to produce training data, without manually aligning data formats
- Fine-tune, evaluate, and deploy a checkpoint to a specific inference server, without writing Python adapters or format-conversion glue code

The hypothesis is that machine-readable metadata on artifacts (models, datasets, environments, etc.) can make compatibility checking and format adaptation automatic — so the tooling handles what engineers currently do by hand. Whether "contracts" are the right mechanism, and what form they should take, is what the research phase needs to determine.

## Working Principles

### 1. Intellectual Rigor

**No premature conclusions without discussion and evidence.**

- Distinguish clearly between validated facts, hypotheses, and speculation
- When presenting findings, surface the evidence gap: "This is based on X sources, but we lack data on Y"
- Every claim should cite a source: documentation link, paper, blog post, Jira issue, GitHub repo, commit

### 2. Critical Partnership

**Critique and push back where warranted.**

- Before solving anything, verify the problem isn't already being solved by communities like ROS, LeRobot, OpenPI, vLLM, or NVIDIA's open specs. Approaches backed by high-momentum communities typically win.
- Enforce YAGNI. Push back when scope creeps beyond validated needs.
- Point out where augmenting existing specs makes more sense than creating new ones, and where new spec can reuse building blocks from existing specs (maturity, user familiarity, avoiding translation layers).
- **Red flags to challenge:**
  - Solution-first thinking ("We should build X" before understanding the user problem)
  - Unvalidated pain points stated as fact
  - Jargon that obscures unclear thinking ("data flywheel", "control plane" without concrete definitions)
  - "Research" documents that contain YAML examples or CLI mockups

### 3. Community-First Disposition

**Assume someone else is already solving this until proven otherwise.**

- Before proposing new spec, exhaustively check whether an existing community is addressing the same problem.
- Prefer extending existing specs (PRs, REPs, PEPs, HF proposals) over creating parallel ones. New specs carry adoption risk.
- A new spec is justified when: (a) the problem falls between communities and none owns it, or (b) existing specs are structurally unable to address it (not just "haven't gotten to it yet").
- Lightweight specs that complement existing ones (like [agent-plugins.org](https://agent-plugins.org/specification) adds on top of agentskills.io) have the best chance of success. Even a small area of tangible improvement is valuable.
- We're willing to find community collaborators for a spec that falls into the crack between communities.

### 4. Problem-Solution Separation

**Keep research and design in separate phases and documents.**

- Research documents describe *what is* and *what hurts*. They don't propose solutions.
- Design documents propose solutions *for validated problems*. They cite the research.
- The current research docs (`research/`) contain a mix of observations and premature design ideas. Treat them as raw input to be sorted, not as decisions.

### 5. Document Quality & Style

**Concise writing — make every paragraph, sentence, and word matter.**

- Run `/stop-slop` after completing larger text sections and before committing.

## Current State

Early research phase. Frank Zdarsky (Red Hat OCTO Emerging Technology) is doing solo exploratory work on Story 1: Schema Research & Prior Art Survey (OCTOET-2177).

### What exists in the repo

| Directory / File | Status | Contains |
| --- | --- | --- |
| `research/survey/` | Active | Prior art survey files (per-community, per-descriptor-type), descriptor data model, synthesis, gap analysis. See `research/survey/README.md` for index. |
| `research/diagrams/` | Active | Architecture diagrams (logical blocks, training paradigms, ecosystem view) with build tooling. |
| `research/simready-foundation-analysis.md` | Raw input | Analysis of NVIDIA SimReady Foundation spec. Useful prior art, but the "Relevance to Our Contracts" section assumes a solution shape we haven't validated. |
| `research/semantic-metadata-gap.md` | Raw input | Problem statement: why shape metadata alone is insufficient for Physical AI interoperability. |

### What's validated vs. hypothesized

**Important caveat**: The "observed" column reflects the experience of a small number of Red Hat engineers working on specific repos. These may be *known solved problems* where existing tooling or best practices were simply not used. The research phase must check whether the broader community (LeRobot contributors, ROS maintainers, NVIDIA developer forums, OpenPI users) considers these problems open or already addressed.

| Observed (in our repos) | Hypothesized (needs validation) |
| --- | --- |
| Camera mapping errors between datasets and models waste hours of debugging | That this isn't already solved or being solved by LeRobot, OpenPI, or similar projects |
| Normalization mismatches cause silent training failures | That a *contract system* is the right fix (vs. better tooling within existing frameworks) |
| Joint ordering differences between simulators require manual remapping | That these pain points are broadly experienced, not artifacts of unfamiliarity with existing solutions |
| Metadata needed for compatibility checks *exists* in component config files | That distinct manifest types for each artifact kind are needed |
| Multiple repos independently reinvent partial compatibility validation | That a standalone spec (vs. contributions to existing projects) is warranted |

## Project Context

This project is part of the Red Hat Physical AI Platform initiative (OCTOET-1966). Project management, strategy documents, and cross-repo tracking live in a separate repo (`physical-ai-project-mgmt`).

### Key Jira Issues

- **OCTOET-1966** — Red Hat Physical AI Platform (initiative, in progress)
- **OCTOET-2176** — Contract SDK & Composition Layer (epic)
- **OCTOET-2177** — Story 1: Schema Research & Prior Art Survey (current work)
- **OCTOET-2178** — Story 2: Minimal Schema Prototype (blocked by 2177)
- **OCTOET-2179** — Story 3: Compatibility Checker
- **OCTOET-2180** — Story 4: Format Adapters
- **OCTOET-2181** — Story 5: CLI & Developer UX
- **OCTOET-2182** — Story 6: KFP Integration PoC
- **OCTOET-2183** — Story 7: Documentation & Community Prep

Use `redhat.atlassian.net` as the `cloudId` for all Atlassian MCP tool calls. Before creating or modifying any Jira issue, always confirm the action with the user.

## Repository Structure

```text
research/           — Background research and analysis (raw input, not decisions)
specs/              — Schema definitions (empty — waiting for research to complete)
  drafts/           — Work-in-progress schemas
sdk/                — Python SDK (empty — waiting for design phase)
docs/               — Documentation
```

## Research Phase: Prior Art to Investigate

These are the schemas and standards to survey for OCTOET-2177. For each: document what it describes, what it doesn't, its type system, extensibility, community adoption, and trajectory.

### Model metadata

- ONNX graph metadata and operator schemas
- MLflow model signatures and model card metadata
- Hugging Face model cards (structured metadata)
- NVIDIA NIM manifest format
- SafeTensors metadata

### Dataset metadata

- LeRobot v3.0 `info.json` / `stats.json` feature descriptions
- Hugging Face dataset cards (structured metadata)
- WebDataset / Croissant (ML dataset description)

### Robot & environment descriptions

- URDF / MJCF robot description formats
- REP-0158: OpenUSD conventions for simulation asset interoperability
- NVIDIA SimReady Foundation (see `research/simready-foundation-analysis.md`)
- USD scene description schema
- ROS 2 interface definitions (`.msg`, `.srv`, `.action`)

### Workflow & composition

- OpenPI protocol input/output definitions
- KFP v2 artifact type system
- agentskills.io / agent-plugins.org

### Platform integration

- RHOAI EvalHub BYOF adapter API
- RHOAI TrainingHub fine-tuning API
- RHOAI SDGHub output metadata

### Action space & observation conventions

Document from existing repos (not invent):

- BridgeData2 (7-DOF EE delta), UMI (10-DOF absolute EE pose), DROID (8-DOF EE + gripper)
- LeRobot normalization strategies (quantile, mean/std, identity, baked-into-checkpoint)
- Camera role conventions (world vs. robot-mounted, slot naming)

## Source Repos for Analysis

These repos contain real configurations that ground the research in concrete examples:

| Repo | What to extract |
| --- | --- |
| `redhat-et/lerobot-vla-finetuning` | pi0.5 config.json, LIBERO info.json, CAMERA_MAP, normalization configs |
| `say-paul/nvidia-industrial-wbc-pipeline` | Isaac Lab env configs, ONNX export metadata, joint orderings |
| `say-paul/robotics-rl` (unified_launcher branch) | RDP YAML execution graphs, robot state definitions, engine abstraction |
| `jeremyary/thor-testing` | Cosmos3-Edge training config, BridgeData2 dataset format, modelcar layout |

## Conventions

- Keep research documents factual — observations and analysis, not proposals
- Mark hypotheses explicitly when they appear
- Cite sources for all claims
- Use YAML for human-readable examples only when illustrating *existing* formats (not proposed ones)
