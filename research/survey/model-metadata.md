# Model Metadata Standards — Prior Art Survey

**Date**: 2026-09-18
**Scope**: Standards and formats that describe ML model metadata — inputs, outputs, types, semantics, provenance. Assessed for relevance to Physical AI interoperability.

---

## 1. ONNX (Open Neural Network Exchange)

### What it describes

ONNX defines a portable computation graph format using Protocol Buffers. Model metadata includes:

- **Input/output tensor specs**: name, element type (28 dtypes from FLOAT to INT2), shape (with symbolic dimension variables for dynamic axes), and optional type denotations
- **Type denotations**: semantic labels on tensors — `TENSOR`, `IMAGE`, `AUDIO`, `TEXT` ([source](https://github.com/onnx/onnx/blob/main/docs/TypeDenotation.md))
- **Dimension denotations**: `DATA_BATCH`, `DATA_CHANNEL`, `DATA_FEATURE` for image layout (NCHW)
- **Image metadata properties**: `Image.BitmapPixelFormat`, `Image.ColorSpaceGamma`, `Image.NominalPixelRange`
- **`metadata_props`**: arbitrary key-value string pairs on ModelProto, GraphProto, FunctionProto, and NodeProto (since IR v10) ([source](https://github.com/onnx/onnx/blob/main/docs/IR.md))
- **Opset version**: declares which operator set version the model uses

### What it doesn't

- No robotics-specific semantics (action spaces, joint states, proprioception, camera roles)
- No normalization metadata (whether normalization is baked into the graph or external)
- No embodiment or robot configuration metadata
- No checkpoint loading semantics
- No dataset provenance or training configuration
- Type denotations are limited to 4 broad categories — no controlled vocabulary for Physical AI concepts

### Type system & extensibility

Strongly typed with 28 element types. Type denotations are a fixed enum — not user-extensible without modifying the spec. `metadata_props` (string→string key-value pairs) is the extensibility escape hatch, but it's unstructured and has no schema.

A 2022 proposal ([onnx/onnx#3958](https://github.com/onnx/onnx/issues/3958)) suggested using RDF/Semantic Web for richer, extensible metadata. Status: open, no implementation.

### Community adoption

Dominant format for model portability. Supported by PyTorch, TensorFlow, and all major inference runtimes. ONNX Runtime is the primary consumer. Governed by the Linux Foundation (ONNX Foundation).

### Trajectory

Recent work focuses on new numeric types (float4, float6, int2) for quantization, not on richer semantic metadata. No visible effort toward robotics-specific extensions.

### Already solving our problems?

**No.** ONNX captures tensor shapes and types but not the semantic meaning of those tensors in a Physical AI context. The WBC pipeline (nvidia-industrial-wbc-pipeline) bakes normalization into the ONNX graph as Sub/Div/BatchNormalization ops — the graph itself becomes a normalization spec, but there's no metadata declaring this pattern. A consumer has no way to know whether an ONNX model expects pre-normalized or raw inputs without inspecting the graph structure.

---

## 2. MLflow Model Signatures

### What it describes

MLflow model signatures define typed input/output schemas stored in the `MLmodel` file:

- **Column-based schemas**: named columns with MLflow data types (for tabular data)
- **Tensor-based schemas**: named tensors with numpy dtypes and shapes (for deep learning)
- **Params schema**: inference-time parameters with types and defaults
- **Required vs. optional** fields with enforcement at load time ([source](https://mlflow.org/docs/latest/ml/model/signatures/))

The Model Registry adds:

- Version lineage (experiment → run → registered model)
- Stage transitions (Staging → Production → Archived)
- Custom tags (string key-value pairs) for governance metadata
- Model card attachment (JSON in artifact, but not queryable via API) ([source](https://mlflow.org/docs/latest/ml/model-registry/))

### What it doesn't

- No semantic annotations on tensor dimensions (what each dimension means — joint angles vs. pixel coordinates vs. action deltas)
- No action space or observation space vocabulary
- No normalization metadata
- No embodiment or robot configuration
- No cross-artifact compatibility checking (model↔dataset, model↔robot)
- Custom metadata in tags is unstructured string→string; no schema enforcement

### Type system & extensibility

Two schema kinds (column-based, tensor-based). Since MLflow 3.x, Pydantic models can define custom input schemas with type hints, validation, and nested structures. This is the main extensibility mechanism — domain-specific metadata could be encoded as Pydantic models, but there's no standard schema for robotics.

### Community adoption

Widely used in enterprise ML (especially Databricks ecosystem). Governed by Databricks (open-source under Apache 2.0). MLflow 3.x (2025–2026) added LLM/agent features; no robotics-specific features.

RHOAI integrates MLflow for experiment tracking and model registry. The WBC pipeline uses MLflow for tracking RL training metrics. LeRobot does not use MLflow.

### Trajectory

MLflow's trajectory is toward LLM/agent tooling (MLflow 3.x adds chat model signatures, agent evaluation). No indication of robotics or Physical AI-specific features.

### Already solving our problems?

**No.** MLflow signatures enforce tensor shape/type at model load time — useful but insufficient for Physical AI. They don't describe what the tensors mean (is this a 7-DOF action delta or an 8-DOF absolute pose?). The Model Registry's custom tags could store Physical AI metadata, but there's no schema or tooling to validate consistency. RHOAI's Model Registry inherits these limitations — it can store key-value metadata but can't enforce or validate robotics-specific schemas ([source: MLflow registry checklist, 2026](https://mlflow.org/articles/ai-model-registry-management-checklist)).

---

## 3. Hugging Face Model Cards

### What it describes

Model cards are Markdown README files with YAML frontmatter. Structured metadata fields include:

- **`pipeline_tag`**: task type (e.g., `robotics`, `text-generation`) — used for Hub filtering and widget selection
- **`library_name`**: framework (e.g., `lerobot`, `transformers`)
- **`base_model`**: parent model identifier(s), with inferred relationship type (`finetune`, `adapter`, `quantized`, `merge`)
- **`datasets`**: training dataset identifiers (Hub references)
- **`license`**: SPDX identifier or custom
- **`tags`**: free-form keywords for discoverability
- **`model-index`**: structured evaluation results (task, dataset, metrics, source)
- **`language`**, **`co2_eq_emissions`**, **`new_version`**, and others ([source](https://huggingface.co/docs/hub/en/model-cards))

### What it doesn't

- No robotics-specific structured fields (action space, observation space, embodiment, camera configuration, normalization strategy)
- No input/output tensor specifications (shapes, types, semantics)
- No checkpoint loading semantics
- No compatibility metadata for cross-artifact validation
- `pipeline_tag: robotics` exists but carries no structured sub-fields — it's a flat label

### Type system & extensibility

YAML frontmatter is schemaless in practice — any key can be added, but only recognized fields are parsed by the Hub. The `tags` field provides free-form extensibility. Model card data is stored as dataclass-based Python objects (`ModelCardData`) with dict-like behavior. There's no mechanism to register domain-specific structured metadata schemas at the Hub level.

### Community adoption

Ubiquitous on the Hugging Face Hub (1.4M+ models as of early 2025). De facto standard for model documentation in the open-source ML community. EU AI Act (2026 enforcement) requires model documentation for high-risk systems, increasing adoption pressure.

### Trajectory

Focus is on compliance (EU AI Act), evaluation results display, and model lineage (`base_model` relationships). As of 2026, there is **no formal proposal** for robotics-specific structured model card fields. LeRobot models use the `pipeline_tag: robotics` label and `library_name: lerobot`, but the card metadata doesn't capture action space, embodiment, or normalization in structured form — these live in the model's `config.json` instead.

### Already solving our problems?

**Partially, but at the wrong level.** HF model cards provide discoverability metadata (tags, task type, lineage) but not the operational metadata needed for compatibility checking. The robotics-specific operational metadata (action dimensions, camera slots, normalization strategy) lives in LeRobot's `config.json` / `pretrained_config.yaml`, not in the model card. There's a gap between "I can find this model on the Hub" and "I can verify this model works with my dataset and robot."

---

## 4. NVIDIA NIM Manifests

### What it describes

NIM containers ship a model manifest file (`model_manifest.yaml`) containing deployment profiles:

- **Profile tags**: backend engine (`vllm`, `sglang`, `trtllm`), precision (`bf16`, `fp8`, `nvfp4`), tensor/pipeline parallelism (`tp`, `pp`), LoRA support
- **Memory annotations**: estimated VRAM per GPU, categorized as compatible/low-memory/incompatible
- **Hardware matching**: GPU device IDs for automatic profile selection
- **Profile selection chain**: priority-ordered selectors for automatic best-fit selection at startup ([source](https://docs.nvidia.com/nim/large-language-models/latest/deployment/model-profiles-and-selection.html))

### What it doesn't

- Scoped entirely to LLM/VLM inference. No support for robotics models, VLAs, diffusion policies, or ONNX-based controllers
- No model capability metadata beyond precision and parallelism
- No input/output specifications
- No normalization, action space, or embodiment metadata
- Internal manifest schema is not publicly documented as a spec

### Type system & extensibility

YAML-based, but the schema is not published as an open spec. Profile tags follow a fixed vocabulary (`vllm`/`sglang`/`trtllm` backends, known precision formats). Not designed for extension to non-LLM domains.

### Community adoption

Proprietary to NVIDIA's NIM ecosystem. Used across NVIDIA's model catalog (build.nvidia.com). Not adopted outside NVIDIA's infrastructure.

### Trajectory

Expanding to more LLM/VLM backends and hardware targets. Separate NIM product families exist for drug discovery, digital biology, etc., but none for robotics policies. GR00T and Isaac models are not distributed as NIMs — they use separate deployment mechanisms.

### Already solving our problems?

**No.** NIM manifests solve a deployment problem (which hardware profile to use for an LLM) but don't address model-dataset compatibility, action space validation, or cross-artifact interoperability. The profile selection pattern (matching model requirements to available hardware) is conceptually interesting — a similar pattern could match model requirements to available datasets — but the format itself is not applicable to Physical AI.

---

## 5. SafeTensors Metadata

### What it describes

SafeTensors is a binary tensor serialization format with a JSON header:

- **Per-tensor metadata**: name, dtype (e.g., `F16`, `BF16`), shape, byte offsets
- **`__metadata__`**: optional string→string key-value map for arbitrary metadata ([source](https://github.com/safetensors/safetensors))

A convention proposed by Stability AI uses `__metadata__` for model architecture name, base model reference, and thumbnail images — primarily targeted at image generation models.

### What it doesn't

- No semantic metadata — tensors are identified by name only (e.g., `model.layers.0.self_attn.q_proj.weight`)
- No input/output specifications
- No computation graph (unlike ONNX)
- No normalization, action space, or embodiment metadata
- `__metadata__` is unstructured (string→string) with no schema

### Type system & extensibility

Minimal by design. 12 dtype values. Extensibility is through `__metadata__` but with no schema enforcement. The format is intentionally limited to tensor storage — it delegates all higher-level metadata to other layers.

### Community adoption

~42% of HF Hub models use SafeTensors (March 2025). Donated to PyTorch Foundation (Linux Foundation) in April 2026. Default format for most new model uploads on HF Hub. Rapidly replaced pickle-based `.bin` files for security reasons.

### Trajectory

Focus on performance (device-aware loading, zero-copy mmap) and security (no arbitrary code execution). No trajectory toward richer metadata — that's explicitly out of scope for the format.

### Already solving our problems?

**No, and it's not trying to.** SafeTensors is a storage format for tensor weights. It deliberately excludes higher-level metadata. Physical AI metadata would need to live in a separate layer (model card, config.json, a manifest file) that accompanies the SafeTensors weights.

---

## 6. Adjacent Standards Worth Noting

### Open X-Embodiment / RLDS

Not a model metadata standard, but the de facto standard for cross-embodiment *dataset* metadata in robotics research:

- **Episode-of-steps format**: observations (multiple cameras, proprioception, language), actions (7-DOF end-effector convention), rewards, metadata
- **Self-describing via `dataset_info.json`**: feature structure, dtypes, shapes
- **60 datasets, 22 robots, 527 skills** from 34 labs
- **Adopted by**: Octo, OpenVLA, pi-zero, RT-X ([source](https://github.com/google-deepmind/open_x_embodiment))
- **Limitation**: uses a coarsely aligned 7-DOF EE action space as the common denominator — lossy for robots that don't match this convention

### ARIO (All Robots In One)

A second-generation effort to standardize cross-embodiment data with stricter quality controls:

- **Unified control format** across morphologies with timestamp synchronization
- **~3M episodes**, 258 robot series, 321K tasks, 5 modalities (2D, 3D, sound, text, tactile)
- **ARIO Alliance**: 10 institutions (Pengcheng Lab, SUSTech, TUM, Agilex, HKU, CUHK, JD Tech, etc.)
- **Status**: published August 2024 ([arXiv 2408.10899](https://arxiv.org/abs/2408.10899)), ongoing development
- **Addresses**: multi-modal metadata, unified format across platforms
- **Doesn't address**: model-dataset compatibility checking, normalization validation, model metadata

### Croissant (MLCommons)

A metadata format for ML-ready datasets, based on schema.org/JSON-LD:

- **Four layers**: dataset metadata, resources (files), structure (record sets), semantics
- **Integrated into**: Hugging Face (400K+ datasets), Kaggle, OpenML
- **RAI extension**: responsible AI metadata vocabulary
- **Status**: v1.0 released, spec licensed CC BY-ND 4.0 ([source](https://github.com/mlcommons/croissant))
- **Limitation**: general-purpose ML dataset metadata — no robotics-specific vocabulary for action spaces, embodiment, or sensor configuration. 25% of HF Croissant files are invalid, suggesting enforcement challenges.

### LeRobot config.json / info.json (Model-Side)

LeRobot models store operational configuration in `config.json` (not model card YAML), including:

- Input feature names and expected shapes
- Camera slot names
- Action space dimensions and semantics (implicitly, via feature naming conventions)
- Normalization strategy (`normalization_mapping`)
- Action chunk size (`n_action_steps`)

This is the closest existing format to what a "model contract" would need to contain for Physical AI. However:

- It's framework-specific (LeRobot only)
- No formal schema — structure varies by policy architecture
- No validation tooling that checks compatibility with a given dataset's `info.json`
- The `dataset_to_policy_features` utility bridges dataset→policy feature names, but it's code, not metadata

---

## Summary: Gap Analysis for Model Metadata

| Capability needed for Physical AI | ONNX | MLflow | HF Cards | NIM | SafeTensors | LeRobot config |
| --- | --- | --- | --- | --- | --- | --- |
| Tensor shapes & types | Yes | Yes | No | No | Yes (storage) | Yes |
| Semantic meaning of tensors (action type, joint semantics) | No | No | No | No | No | Partial (naming convention) |
| Normalization strategy | No | No | No | No | No | Yes (framework-specific) |
| Camera slot names & mapping | No | No | No | No | No | Yes (framework-specific) |
| Checkpoint loading semantics | No | No | No | No | No | No |
| Embodiment / robot type | No | No | No | No | No | No |
| Cross-artifact compatibility checking | No | No | No | No | No | No |
| Discoverability (search, filter) | No | Via tags | Yes | No | No | Via HF Hub |
| Deployment profile (hardware matching) | No | No | No | Yes | No | No |

### Key finding

**No existing model metadata standard covers the Physical AI interoperability gap.** The closest is LeRobot's `config.json`, which contains the right *kind* of information (camera slots, normalization, action dimensions) but in a framework-specific, unvalidated format with no cross-artifact compatibility checking.

The LeRobot ICLR 2026 paper ([arXiv 2602.22818](https://arxiv.org/html/2602.22818v1)) explicitly identifies cross-embodiment standardization, schema validation, and sensor calibration as open gaps, and frames them as "concrete, tractable avenues for community contributions" — effectively inviting external solutions rather than committing to build them internally.

### Critical question for our research

LeRobot's `config.json` + `info.json` already contain most of the metadata fields a contract would need. The question is whether the right approach is:

1. **Extend LeRobot's metadata** — propose a formal schema for `config.json` with validation tooling, and contribute it upstream to LeRobot/HF. This has the highest adoption potential (16K+ datasets, growing fast) but is framework-specific.
2. **Create a cross-framework metadata layer** — a lightweight spec that normalizes metadata from LeRobot, ONNX, OpenPI, RLDS, and others into a common format for compatibility checking. Higher abstraction, broader scope, but higher adoption risk.
3. **Build compatibility tooling on existing metadata** — skip the new spec entirely and build validation tools that read native `config.json` / `info.json` / ONNX metadata directly. Lowest adoption barrier but framework-coupled and fragile to format changes.

This is the central "augment vs. create" question the research phase needs to resolve.
