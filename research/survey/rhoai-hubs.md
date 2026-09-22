# RHOAI Hubs, Model Registry & vLLM-Omni — Prior Art Survey

**Date**: 2026-09-18
**Scope**: Red Hat OpenShift AI platform components relevant to Physical AI workflows
**Purpose**: Understand what data flows between platform Hubs and external components, identify where interoperability metadata is needed

---

## 1. SDG Hub (Synthetic Data Generation Hub)

**Source**: [GitHub](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub), [Red Hat Developer article](https://developers.redhat.com/articles/2025/10/27/sdg-hub-building-synthetic-data-pipelines-modular-blocks), [Red Hat blog](https://www.redhat.com/en/blog/red-hat-ai-modular-building-blocks-scalable-repeatable-model-customization)

### Purpose

Python framework for building synthetic data generation pipelines. Modular, composable blocks that chain into YAML-defined flows. Available as tech preview in Red Hat AI 3 with supported builds of the SDG Hub library.

### Architecture

Three-layer design:

1. **Blocks** — self-contained processing units. Types: LLM, Parsing, Transform, Filtering, Agent, Custom. Each block declares `input_cols` and `output_cols` as lists of column name strings. Data flows as tabular/row-oriented samples (dictionaries). Blocks implement a `generate(self, samples, **kwargs)` method.

2. **Flows** — YAML-defined pipelines chaining blocks. Columns thread through: one block's output columns become available as input columns for downstream blocks. Flows support async parallelism, debugging, dry-run validation.

3. **Registries** — `FlowRegistry` for discovering and loading built-in flows. `BlockRegistry` with `@BlockRegistry.register()` decorator for custom blocks.

### API

```python
FlowRegistry.discover_flows()
flow = Flow.from_yaml(FlowRegistry.get_flow_path("MCP Server Distillation"))
flow.set_model_config(model="openai/gpt-4o")
result = flow.generate(dataset)
```

No REST API documented for job submission — currently used as a Python library within notebooks or KFP pipeline steps.

### Data formats

- **Input**: tabular datasets (column-keyed dictionaries)
- **Output**: tabular datasets with generated columns added
- **No metadata schema** for generated datasets — column names serve as the implicit schema
- **No dataset manifest** attached to output — no declaration of what the generated data contains, its provenance, or quality characteristics

### Physical AI relevance

SDG Hub is currently **LLM-focused** (text-in, text-out blocks). No blocks exist for:

- Robot episode generation (simulation → trajectory data)
- Sensor data synthesis (camera, LiDAR, IMU)
- Domain randomization workflows
- Scene asset manipulation

For Physical AI synthetic data generation (e.g., Isaac Sim domain randomization → training episodes), the SDG Hub block/flow pattern could be extended, but no Physical AI blocks exist. The `BlockRegistry` extension mechanism would allow adding simulation-based data generation blocks.

**Gap for contracts**: If SDG Hub produced Physical AI datasets, it would need to declare the output schema (action space, observation format, embodiment, camera configuration, normalization stats) so downstream consumers (Training Hub, model training) could validate compatibility. Currently no mechanism for this.

---

## 2. EvalHub (Evaluation Hub)

**Source**: [EvalHub overview](https://developers.redhat.com/articles/2026/05/19/evalhub-because-looks-good-me-isnt-benchmark), [BYOF article](https://developers.redhat.com/articles/2026/06/09/bring-your-own-evaluation-framework-evalhub), [GitHub (upstream)](https://github.com/eval-hub/eval-hub), [GitHub (Red Hat)](https://github.com/red-hat-data-services/eval-hub)

### Purpose

Framework-agnostic evaluation orchestration platform. Written in Go (server) + Python (SDK/CLI). GA in Red Hat AI 3.5 (September 2026). Deploys as part of the TrustyAI stack on OpenShift.

### Architecture

- **Server**: Go REST API, central orchestration control plane
- **SDK** (Python): REST client, CLI, BYOF adapter, MCP server, OCI artifact persistence
- **Execution**: Each evaluation runs as a separate K8s Job with a sidecar for progress reporting. Local mode available for development.
- **Results storage**: MLflow (experiment tracking), OCI registry (artifact persistence), collection scoring (weighted aggregation)

### Evaluation submission API

```
POST /api/v1/evaluations
```

```json
{
  "name": "llama-3-v2-healthcare-eval",
  "model": { "url": "http://vllm-service:8080/v1" },
  "collection": { "id": "healthcare_safety_v1" },
  "experiment": { "name": "llama-3-v2-healthcare-eval" }
}
```

A single request fans out to multiple backend frameworks in parallel, aggregates results with configurable weights, writes the complete experiment record.

### BYOF (Bring Your Own Framework) adapter

Single abstract method:

```python
def run_benchmark_job(self, config: JobSpec, callbacks: JobCallbacks) -> JobResults
```

**JobSpec** fields: `id`, `provider_id`, `benchmark_id`, `model` (url + name), `parameters` (opaque dict), `num_examples`, `experiment_name`, `tags`, `exports`, `timeout_seconds`.

**JobResults** fields: `results` (list of `EvaluationResult` with `metric_name`, `metric_value`, `confidence_interval`, `num_samples`, `metadata`), `overall_score`, `num_examples_evaluated`, `duration_seconds`, `evaluation_metadata`.

Status phases: INITIALIZING → LOADING_DATA → RUNNING_EVALUATION → POST_PROCESSING → PERSISTING_ARTIFACTS → COMPLETED.

### Default providers

| Provider | Purpose |
| --- | --- |
| lm-evaluation-harness | 167 benchmarks for capability, reasoning, knowledge |
| Garak | Red-teaming, safety probes |
| GuideLLM | Throughput, latency profiling |
| LightEval | Fast capability benchmarks |
| MTEB | Text embedding benchmarks |

### Physical AI relevance

EvalHub is currently **LLM-focused**. No default providers evaluate robot policies. However, the BYOF pattern is directly applicable:

A Physical AI BYOF adapter could:

1. Accept a model URL (OpenPI server or vLLM-Omni serving a VLA policy)
2. Launch a simulation environment (Isaac Sim, MuJoCo)
3. Run evaluation episodes (execute policy in sim, measure task success)
4. Return metrics (success rate, path efficiency, collision count)

The adapter would need to know:

- Model's action space and observation format (to feed sim observations to model)
- Environment/task specification (which sim scene, what success criterion)
- Robot embodiment (which URDF/MJCF, joint ordering)

**Gap for contracts**: The `JobSpec.model` field carries only a URL and name — no action space, observation format, or embodiment metadata. The `parameters` dict is opaque. A Physical AI evaluation adapter would need structured metadata about the model, environment, and robot to set up the evaluation correctly. This is exactly the cross-artifact compatibility metadata that contracts would provide.

---

## 3. Training Hub

**Source**: [Scale LLM fine-tuning article](https://developers.redhat.com/articles/2026/03/26/scale-llm-fine-tuning-training-hub-and-openshift-ai), [GRPO article](https://developers.redhat.com/articles/2026/08/26/reinforcement-learning-from-verifiable-rewards-with-training-hub-on-red-hat-openshift-ai), [GitHub](https://github.com/Red-Hat-AI-Innovation-Team/training_hub)

### Purpose

Python package providing standardized fine-tuning algorithms for LLMs. Preinstalled in Ray CUDA runtime images in RHOAI 3.5. Supports air-gapped environments.

### Supported algorithms

- **SFT** — Supervised Fine-Tuning
- **OSFT** — Orthogonal Subspace Fine-Tuning (continual learning)
- **LoRA / QLoRA** — Low-Rank Adaptation
- **LORA_GRPO** — LoRA with Group Relative Policy Optimization (RL from verifiable rewards)

### API surface

**Local/notebook API** (simple function calls):

```python
from training_hub import sft
result = sft(
    model_path="Qwen/Qwen2.5-1.5B-Instruct",
    data_path="/path/to/data.jsonl",
    ckpt_output_dir="/path/to/checkpoints",
    num_epochs=3,
    effective_batch_size=8,
    learning_rate=1e-5,
    max_seq_len=256,
)
```

**Kubeflow Trainer API** (distributed on OpenShift):

```python
from kubeflow.trainer import TrainerClient, TrainingHubTrainer, TrainingHubAlgorithms

job_name = client.train(
    trainer=TrainingHubTrainer(
        algorithm=TrainingHubAlgorithms.LORA_GRPO,
        func_args=params,
        env={"HF_HOME": "..."},
        resources_per_node={"cpu": 8, "memory": "64Gi", "nvidia.com/gpu": 1},
    ),
    options=[pod_template_overrides],
    runtime=th_runtime,
)
```

### Data formats

- **Models**: HuggingFace identifiers or local paths. Standard HF model directory layout.
- **Datasets**: JSONL files. No schema enforcement beyond what the training algorithm expects.
- **Checkpoints**: Standard HuggingFace checkpoint format (SafeTensors + config.json).

### Physical AI relevance

Training Hub is **LLM-focused**. The algorithms (SFT, LoRA, GRPO) are applicable to VLA models (pi0.5 is trained via SFT on LeRobot data), but:

- No support for LeRobot's video-based dataset format (Parquet + MP4)
- No support for normalization computation (quantile stats, etc.)
- No awareness of action spaces, camera configurations, or embodiment
- The `data_path` parameter expects JSONL, not episode-structured robot data

For Physical AI training, either:

1. Training Hub would need LeRobot-aware extensions (video data loading, normalization, action space handling)
2. Or LeRobot's `lerobot-train` would be used directly (as in lerobot-vla-finetuning), bypassing Training Hub

**Gap for contracts**: Training Hub passes model and data paths as opaque strings. It has no mechanism to validate that the dataset's features match the model's expected inputs, or that normalization stats are available in the right format. A contract system could add a validation step before `client.train()`.

---

## 4. Model Registry / Model Catalog

**Source**: [RHOAI docs v2.25](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/2.25/html/enabling_the_model_registry_component/overview-of-model-registries_model-registry-config), [Kubeflow REST API v1alpha3](https://www.kubeflow.org/docs/components/hub/reference/rest-api/), [model-catalog-bridge](https://github.com/redhat-ai-dev/model-catalog-bridge), [MaaS architecture](https://developers.redhat.com/articles/2026/08/18/architecting-the-red-hat-openshift-ai-dashboard-for-models-as-a-service)

### Data model

Based on Kubeflow Model Registry (v1alpha3 REST API):

| Entity | Key fields | Custom properties |
| --- | --- | --- |
| **RegisteredModel** | `name`, `owner`, `description`, `state` | `Map<string, MetadataValue>` |
| **ModelVersion** | `name`, `author`, `description`, `state` | `Map<string, MetadataValue>` |
| **ModelArtifact** | `name`, `uri`, `modelFormatName`, `modelFormatVersion`, `storageKey`, `storagePath` | `Map<string, MetadataValue>` |

Relationships: RegisteredModel 1↔*ModelVersion, ModelVersion 0..1↔* Artifact.

### Custom properties

`MetadataValue` is a union type supporting: `MetadataStringValue`, `MetadataIntValue`, `MetadataDoubleValue`, `MetadataBoolValue`, `MetadataStructValue` (JSON object, added later), `MetadataProtoValue`.

**Key limitation**: Most model metadata must be stored as custom properties (key-value pairs). The model-catalog-bridge project notes: "At this time, most of the model metadata expressed by the Model Catalog Schema does not have corresponding fields in the Kubeflow-based Model Registry component's RegisteredModel and ModelVersion data types." ([source](https://github.com/redhat-ai-dev/model-catalog-bridge))

### Model Catalog (RHOAI 2.25+)

A curated catalog of validated models (Red Hat, IBM, Meta, NVIDIA, Mistral, Google). Models are defined via ConfigMaps (`model-catalog-sources`). Separate from the Model Registry — the catalog is for discovery, the registry is for lifecycle management.

### Models-as-a-Service (MaaS)

REST API pattern for publishing, governing, and consuming models through a gateway. Components: K8s controller, REST API for subscriptions/API keys, gateway API ingress, dashboard. Models are served via vLLM behind the gateway.

### Physical AI relevance

The Model Registry's `ModelArtifact` has `modelFormatName` and `modelFormatVersion` — designed for formats like "onnx" / "1.13" or "safetensors" / "0.4". For Physical AI, additional metadata would need to go into custom properties:

- Action space (type, dimensions, semantics)
- Observation format (camera slots, state vector dimensions)
- Normalization strategy (baked/external, which stats format)
- Embodiment compatibility (which robots this model works with)
- Checkpoint loading semantics

Since custom properties support `MetadataStructValue` (JSON objects), structured Physical AI metadata could be stored — but with no schema enforcement. The registry would accept any JSON; it wouldn't validate that the action space declaration is well-formed or consistent with the dataset used for training.

**Gap for contracts**: The Model Registry can *store* Physical AI metadata in custom properties, but cannot *validate* or *query* it structurally. A contract schema could define the expected structure for Physical AI custom properties, enabling both validation at registration time and structured queries ("find all models compatible with Franka Panda 7-DOF arm").

---

## 5. vLLM-Omni

**Source**: [vLLM-Omni docs](https://docs.vllm.ai/projects/vllm-omni/en/latest/), [Red Hat blog](https://next.redhat.com/2026/06/30/expanding-ai-beyond-text-vllm-omni-extension-for-multimodal-models/), [vLLM blog announcement](https://vllm.ai/blog/2025-11-30-vllm-omni), [arXiv:2602.02204](https://arxiv.org/pdf/2602.02204)

### Purpose

Open-source framework extending vLLM to support omni-modality model inference: text, image, audio, video, and **action** output. Supports non-autoregressive architectures (Diffusion Transformers). Apache 2.0. Red Hat contributor (Ricardo Noriega, OCTOET-1203).

### Supported robot policy / VLA models

As of v0.26.0 (August 2026):

| Model | Type |
| --- | --- |
| **π0** (pi-zero) | VLA (vision-language-action) |
| **GR00T-N1.7** | VLA (humanoid) |
| **DreamZero-DROID** | VLA (manipulation) |
| **InternVLA-A1** | VLA |

Also supports world models (Cosmos3) and diffusion models (HunyuanImage, BAGEL, etc.).

### Architecture

- **Diffusion Engine**: handles non-autoregressive generation (diffusion policies, image/video generation)
- **Multistage pipeline**: decomposes models into pipelined stages (vision encoding → reasoning → action generation)
- **Fully disaggregated serving**: OmniConnector for dynamic GPU resource allocation across stages
- **OpenAI-compatible API**: extends the standard `/v1/chat/completions` endpoint for multimodal inputs

### Physical AI relevance

vLLM-Omni is the **serving layer** for Physical AI models on RHOAI. It can serve VLA policies (pi0, GR00T) that take camera images + text instructions + proprioception and return action vectors.

**What vLLM-Omni needs to know about a model** (to serve it correctly):

- Input modalities (which camera images, text format, proprioceptive state vector)
- Output format (action vector dimensions, action chunking)
- Whether the model uses autoregressive or diffusion-based action generation
- Normalization (does the model expect normalized inputs? are outputs normalized?)

**Current state**: Model-specific details are currently encoded in the model's config files (HuggingFace `config.json`). vLLM-Omni reads these at model load time. There is no external metadata standard for declaring a VLA model's serving interface.

**Gap for contracts**: When deploying a VLA model to vLLM-Omni and connecting it to a robot (via ROS 2 topics or OpenPI client), the system needs to validate that:

1. The model's expected camera inputs match the robot's available cameras
2. The model's action output format matches the robot's control interface
3. Normalization is handled correctly (model outputs → robot commands)

Currently, this validation is done by the person writing the deployment config. A contract system could automate it.

---

## 6. Cross-Hub Data Flows

### Current flow (LLM fine-tuning on RHOAI)

```
SDG Hub → (JSONL) → Training Hub → (HF checkpoint) → Model Registry → (model URI) → vLLM → EvalHub
```

Metadata passed between stages:

- SDG Hub → Training Hub: **file path only** (no schema, no feature description)
- Training Hub → Model Registry: **model format name, URI** (no input/output schema)
- Model Registry → vLLM: **model URI, format** (vLLM reads config.json from checkpoint)
- EvalHub → vLLM: **model URL** (evaluation framework handles the rest)

### Hypothetical Physical AI flow

```
Sim (Isaac Sim) → (episodes) → Dataset Store → Training (LeRobot) → Model Registry → vLLM-Omni → Robot (ROS 2)
         ↓                            ↑
    Domain randomization        Real episodes (ROSbag/OTel)
```

Metadata needed at each boundary:

- **Sim → Dataset Store**: action space semantics, camera config, embodiment, normalization stats, episode structure
- **Dataset Store → Training**: compatibility check (does dataset match model's expected inputs?)
- **Training → Model Registry**: model manifest (action space, observation format, normalization strategy, embodiment compatibility)
- **Model Registry → vLLM-Omni**: serving config (input modalities, output format, diffusion vs. AR)
- **vLLM-Omni → Robot**: action format validation (model output ↔ robot control interface)
- **Robot → Dataset Store**: episode format, sensor calibration, camera mapping

**None of these metadata transfers are currently standardized.** Each boundary is a potential interoperability failure point.

---

## 7. Gap Summary for Physical AI

| RHOAI Component | What it knows about models/data | What it needs for Physical AI |
| --- | --- | --- |
| SDG Hub | Column names | Action space, observation format, embodiment, sim asset references |
| Training Hub | File path, algorithm name | Dataset-model compatibility, normalization strategy, camera mapping |
| EvalHub | Model URL, benchmark ID | Model action space, environment spec, robot embodiment, success criteria |
| Model Registry | Format name, URI, k-v properties | Structured Physical AI metadata (action space, normalization, embodiment) |
| vLLM-Omni | Model config.json (at load time) | Validated serving interface (input modalities, output format, normalization) |
| KFP Pipelines | Artifact type (Model/Dataset), URI | Typed artifact schemas (what's inside this Dataset? what does this Model expect?) |

### Where contracts would plug in

1. **Registration time**: When registering a model in the Model Registry, contract metadata (action space, observation format, normalization, embodiment compatibility) is validated and stored as structured custom properties.

2. **Training job submission**: Before Training Hub launches a fine-tuning job, contracts validate that the dataset's features match the model's expected inputs.

3. **Evaluation setup**: Before EvalHub launches a Physical AI evaluation, contracts validate that the model, environment, and robot are compatible.

4. **Serving deployment**: Before deploying a model to vLLM-Omni for a specific robot, contracts validate the action format and normalization chain.

5. **Dataset ingestion**: When episodes arrive from simulation or real robots, contracts validate the data format and attach metadata for downstream consumers.
