# Descriptor Survey — CheckpointDescriptor & InferenceServerDescriptor

**Date**: 2026-09-21
**Scope**: How existing communities represent model checkpoint metadata and inference server configuration. Maps to our `CheckpointDescriptor` (what a trained model expects/produces) and `InferenceServerDescriptor` (what a running server accepts/provides).
**Method**: Source code analysis, documentation review, existing survey cross-references.

---

## 1. HuggingFace Hub — Model Cards & SafeTensors

### Model Cards (YAML frontmatter + README.md)

**What they call it**: Model Card metadata — structured YAML frontmatter in `README.md`.

**Explicitly declared fields** ([source](https://huggingface.co/docs/hub/en/model-cards)):

| Field | Type | Purpose | Populated for VLAs? |
| --- | --- | --- | --- |
| `pipeline_tag` | string | Task type (e.g., `robotics`, `text-generation`) | Yes — `robotics` for LeRobot models |
| `library_name` | string | Framework (e.g., `lerobot`, `transformers`) | Yes |
| `base_model` | string or list | Parent model ID(s), with inferred relationship type (`finetune`, `adapter`, `quantized`, `merge`) | Yes for fine-tunes |
| `datasets` | list of strings | Training dataset HF repo IDs | Sometimes |
| `license` | string | SPDX identifier | Yes |
| `tags` | list of strings | Free-form keywords | Yes |
| `model-index` | structured | Evaluation results (task, dataset, metrics) | Rare for robotics |

**What is NOT declared**:

- Input/output tensor specs (shapes, types, semantics)
- Action space, observation space, camera configuration
- Normalization strategy or statistics
- Embodiment or robot compatibility
- Action chunk horizon or control frequency

**Format/location**: YAML frontmatter in `README.md` at repo root. Parsed by HF Hub for search/filtering. The `ModelCardData` Python dataclass allows arbitrary extra keys, but only recognized fields are indexed.

### SafeTensors `__metadata__`

**What they call it**: `__metadata__` — optional string-to-string key-value map in the JSON header of `.safetensors` files.

**What's stored in practice for VLA models**: Minimal. Typical VLA checkpoint SafeTensors files contain per-tensor metadata (name, dtype, shape, byte offsets) but the `__metadata__` map is either absent or contains only `format` (e.g., `"pt"` for PyTorch origin). A convention proposed by Stability AI adds `modelspec.architecture`, `modelspec.title`, and `modelspec.sai_model_spec` for image generation models, but this has not been adopted for robotics/VLA models ([source](https://github.com/safetensors/safetensors)).

**What is NOT stored**: Any robotics-specific metadata — action space, observation format, normalization, embodiment, camera configuration. SafeTensors is deliberately minimal — a pure tensor storage format. All higher-level metadata is delegated to companion files (`config.json`, model card, processor files).

### Gaps relative to CheckpointDescriptor

- **No input/output format declaration**: model cards say *what* task a model performs but not *how* — what observations it expects or what action format it produces.
- **No normalization metadata**: neither the model card nor SafeTensors headers declare normalization strategy or statistics.
- **No embodiment metadata**: `pipeline_tag: robotics` is a flat label with no structured sub-fields for robot type, DOF, or sensor configuration.
- **Discoverability without operability**: you can *find* a model on the Hub, but you cannot programmatically determine whether it's compatible with your dataset and robot.

---

## 2. OpenPI — Physical Intelligence Inference Server

### Concept name

No explicit "checkpoint descriptor" or "server descriptor" concept. Model metadata is distributed across Python config objects, JSON files, and adapter classes.

### Checkpoint metadata (config.json + norm_stats.json)

**config.json** — stored in the checkpoint directory. For pi0/pi0.5, key fields include:

| Field | Example value | Purpose |
| --- | --- | --- |
| `type` | `"pi0"`, `"pi05"` | Policy architecture identifier |
| `n_obs_steps` | `1` | Number of observation history steps |
| `n_action_steps` | `10` | Action chunk horizon |
| `chunk_size` | `50` | Maximum action chunk length |
| `max_state_dim` | `32` | Maximum proprioception dimensions (zero-padded) |
| `max_action_dim` | `32` | Maximum action dimensions (zero-padded) |
| `normalization_mapping` | `{"VISUAL": "IDENTITY", "STATE": "MEAN_STD", "ACTION": "MEAN_STD"}` | Per-modality normalization strategy |
| `input_features` | dict of feature name → `{shape, type}` | Expected input features |
| `output_features` | dict of feature name → `{shape, type}` | Expected output features |
| `resize_imgs_with_padding` | `true` | Image preprocessing flag |
| `empty_cameras` | `1` | Number of unused camera slots (filled with padding) |

Source: [HF pi0.5 docs](https://huggingface.co/docs/lerobot/pi05), [data-formats survey section 2](data-formats-at-interfaces.md)

**norm_stats.json** — per-feature normalization statistics. Format:

```json
{
  "observation.state": {
    "mean": [float, ...],
    "std": [float, ...],
    "min": [float, ...],
    "max": [float, ...],
    "q01": [float, ...],
    "q99": [float, ...]
  },
  "action": { ... }
}
```

Used by the server to normalize incoming state observations and denormalize outgoing actions. Loaded from the dataset that was used for training. The presence of `q01`/`q99` determines whether quantile normalization is available.

### Server configuration

**Not declarative metadata** — server configuration is a Python `TrainConfig` object (not a standalone file). Specifies:

- Checkpoint path (local or GCS)
- Policy config class name (e.g., `pi0_fast_droid`, `pi0_fast_libero`)
- Per-robot adapter class references (e.g., `DroidInputs`, `LiberoInputs`)
- Dataset config for normalization stats

Server started via CLI: `uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_fast_droid --policy.dir=<path>`

Source: [OpenPI config.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py), [OpenPI remote inference docs](https://github.com/Physical-Intelligence/openpi/blob/main/docs/remote_inference.md)

### Introspection / metadata API

**None.** The OpenPI server exposes a WebSocket endpoint on port 8000. There is no metadata or capability discovery endpoint. Clients must know the exact observation field names for the configured robot. If the client sends wrong field names or shapes, the server raises a runtime error.

Source: [OpenPI Issue #1046](https://github.com/Physical-Intelligence/openpi/issues/1046)

### Implicit properties (embedded in adapter code, not declared)

The per-robot adapter classes (`DroidInputs`, `AlohaInputs`, `LiberoInputs`, etc.) contain hardcoded:

- Camera name remapping (robot-native → model-native keys)
- Image format conversion rules
- State vector assembly order
- Joint sign flip masks
- Gripper space conversion constants
- Output action slicing dimensions

These are the most operationally critical metadata for interoperability, and they exist only as Python code constants — not as queryable metadata.

Source: [OpenPI policies directory](https://github.com/Physical-Intelligence/openpi/tree/main/src/openpi/policies)

### Gaps relative to CheckpointDescriptor / InferenceServerDescriptor

- **No action semantics**: config.json declares `max_action_dim` but not what each dimension means (EE delta? joint position? which joints?).
- **No camera role metadata**: camera slot names are in config.json but their physical roles (wrist, world, overhead) are only in adapter code.
- **No server introspection**: a client cannot discover what the server expects without prior knowledge.
- **No embodiment declaration**: which robots this checkpoint/server supports is encoded only in the Python config class name and adapter code selection.

---

## 3. LeRobot — Policy Configuration

### Concept name

Policy config — a Python dataclass (e.g., `Pi0Config`, `DiffusionConfig`, `ACTConfig`) serialized as `config.json` in checkpoint directories.

### Explicitly declared fields

For `Pi0Config` (pi0/pi0.5 policies):

| Field | Type | Purpose |
| --- | --- | --- |
| `type` | string | Policy architecture (`"pi0"`, `"pi05"`) |
| `n_obs_steps` | int | Observation history length |
| `n_action_steps` | int | Action chunk horizon |
| `chunk_size` | int | Maximum chunk size |
| `max_state_dim` | int | Max proprioception dimensions (zero-padded to this) |
| `max_action_dim` | int | Max action dimensions (zero-padded to this) |
| `input_features` | dict | Feature name → `{shape: list, type: str}` |
| `output_features` | dict | Feature name → `{shape: list, type: str}` |
| `normalization_mapping` | dict | Modality → strategy (`"MEAN_STD"`, `"QUANTILE"`, `"IDENTITY"`) |
| `resize_imgs_with_padding` | bool | Whether images get aspect-preserving resize with padding |
| `empty_cameras` | int | Number of empty camera slots |

For `DiffusionConfig`:

| Field | Type | Purpose |
| --- | --- | --- |
| `input_shapes` | dict | Feature name → shape (e.g., `{"observation.image": [3, 96, 96]}`) |
| `output_shapes` | dict | Feature name → shape (e.g., `{"action": [2]}`) |
| `n_obs_steps` | int | Observation history length |
| `horizon` | int | Action prediction horizon |
| `n_action_steps` | int | Steps actually executed per chunk |

For `ACTConfig`:

| Field | Type | Purpose |
| --- | --- | --- |
| `input_shapes` | dict | Feature name → shape |
| `output_shapes` | dict | Feature name → shape |
| `chunk_size` | int | Action chunk length |
| `n_obs_steps` | int | Observation history |

Source: [LeRobot policy configs](https://github.com/huggingface/lerobot/tree/main/lerobot/common/policies)

### Serialized processing pipelines (preprocessor / postprocessor)

LeRobot checkpoints (as of mid-2026) ship serialized processing pipelines alongside the model weights. These are the most complete declarative inference-time metadata in any surveyed ecosystem.

**`policy_preprocessor.json`** — an ordered list of named processing steps applied to observations before model forward pass:

| Step (registry_name) | Purpose | Config fields |
| --- | --- | --- |
| `rename_observations_processor` | Remap client observation keys to model-expected keys | `rename_map` (dict) |
| `to_batch_processor` | Convert single observation to batch format | (none) |
| `relative_actions_processor` | Convert absolute actions to relative (if enabled) | `enabled`, `exclude_joints`, `action_names` |
| `normalizer_processor` | Normalize inputs using learned statistics | `features` (name → {type, shape}), `norm_map` (modality → strategy), `eps`. Companion `state_file` (safetensors) with quantile/mean/std values. |
| `pi05_prepare_state_tokenizer_processor_step` | Architecture-specific state preparation | (none) |
| `tokenizer_processor` | Tokenize language instruction | `tokenizer_name`, `max_length`, `padding`, `truncation` |
| `device_processor` | Move tensors to target device | `device`, `float_dtype` |

**`policy_postprocessor.json`** — reverse pipeline for model outputs:

| Step (registry_name) | Purpose | Config fields |
| --- | --- | --- |
| `unnormalizer_processor` | Denormalize action outputs using learned statistics | `features` (action shape/type), `norm_map`, `eps`. Companion `state_file` (safetensors). |
| `absolute_actions_processor` | Convert relative actions back to absolute (if enabled) | `enabled` |
| `device_processor` | Move to CPU | `device`, `float_dtype` |

Each step has a `registry_name` (resolved to a Python class at runtime) and a `config` dict. Steps that carry learned state (normalization statistics) reference a companion safetensors file via `state_file`.

**Concrete example** from [`execbat/pi05-robot-finetuned`](https://huggingface.co/execbat/pi05-robot-finetuned): 7-step preprocessor (rename → batch → relative_actions[disabled] → normalize[QUANTILES for state/action, IDENTITY for images] → pi05_state_prep → tokenize[PaliGemma] → device[cuda]), 3-step postprocessor (unnormalize[QUANTILES] → absolute_actions[disabled] → device[cpu]).

**Significance**: This is the only ecosystem where the full inference-time processing chain is **declaratively described and shipped with the checkpoint**. OpenPI, vLLM-Omni, and SGLang all reconstruct equivalent pipelines from hardcoded server code. A server that understands LeRobot's pipeline format can reproduce the exact preprocessing without model-specific code — addressing the "training config is richer than checkpoint metadata" gap.

**Limitations**: The `registry_name` values require a LeRobot Python runtime to resolve — they're not a standalone specification. Architecture-specific steps (e.g., `pi05_prepare_state_tokenizer_processor_step`) embed model-family knowledge. The format is a LeRobot implementation detail, not a published standard.

Source: [execbat/pi05-robot-finetuned](https://huggingface.co/execbat/pi05-robot-finetuned), [LeRobot inference docs](https://huggingface.co/docs/lerobot/main/inference)

### LeRobot PolicyServer (async inference)

LeRobot also provides a **gRPC-based PolicyServer** for distributed inference, introduced with SmolVLA. The server starts empty; the client configures it via a `SendPolicyInstructions` handshake that sends a `RemotePolicyConfig` (policy_type, pretrained path, device, actions_per_chunk, features, rename_map). The server then loads the checkpoint and its processing pipelines.

**gRPC service** (`AsyncInference`): 4 RPCs — `Ready` (health check), `SendPolicyInstructions` (configure server), `SendObservations` (stream observations), `GetActions` (receive action chunks). All data fields use **pickle serialization** over gRPC `bytes` — the protobuf messages are opaque byte blobs, not structured.

**Action chunking features**: The client maintains a local action queue and merges overlapping chunks using configurable aggregation (weighted_average: 0.3×old + 0.7×new, latest_only, average, conservative). Real-Time Chunking (RTC) adds a guidance term during flow-matching denoising for chunk-to-chunk consistency.

**Security**: Pickle-over-gRPC is a known critical vulnerability ([CVE-2026-25874](https://github.com/huggingface/lerobot/issues/3047), CVSS 9.3/9.8, filed 2026-02-27). Three of four `AsyncInference` RPC handlers pass attacker-controlled bytes into `pickle.loads()` with `# nosec` markers. No authentication, no TLS by default. The [strands-labs/robots](https://github.com/strands-labs/robots/issues/4257) project cannot build a thin gRPC client because the wire format is pickle, not structured protobuf. They tested v0.6.1 and confirmed pickle is still the wire format on both directions.

**Planned replacement** ([PR #3048](https://github.com/huggingface/lerobot/pull/3048)): A `safe_serialization.py` module replaces pickle with a binary wire format: `[4-byte JSON length][JSON metadata][safetensors tensor data]`. Three serialization pairs: policy config (JSON-only), observations (safetensors + JSON metadata preserving numpy/torch types), actions (safetensors + JSON for timestamps/timesteps). The gRPC service definition and protobuf `bytes` fields remain unchanged — only the payload encoding inside them changes. This is a security fix, not an interoperability fix: the wire format becomes safe to deserialize but remains structurally opaque (a length-prefixed binary blob, not typed protobuf fields). A client still cannot inspect the proto schema to understand what the server expects.

**Status**: PR #3048 has been open and unmerged for ~7 months (since 2026-02-27). v0.6.0 (July 2026) shipped without it. The [v0.7.0 roadmap](https://github.com/huggingface/lerobot/issues/3832) lists it as team item "RUN-04: Secure and converge remote inference" — not yet started.

**Contrast with OpenPI/vLLM-Omni/SGLang**: LeRobot's PolicyServer is designed for single-robot, single-client deployment. No batching, no multi-model, no multiplexing. The processing pipeline comes from the checkpoint's serialized JSON files, not from server-side adapter code. The other servers use HTTP/WebSocket with msgpack- or JSON-encoded payloads — language-agnostic and inspectable, though with their own interoperability issues (three msgpack numpy dialects, see [deployment-topology.md Section 3.1](deployment-topology.md)).

Source: [LeRobot async inference docs](https://huggingface.co/docs/lerobot/en/async), [HF Blog — Async Robot Inference](https://huggingface.co/blog/async-robot-inference), [LeRobot RTC docs](https://huggingface.co/docs/lerobot/rtc), [CVE-2026-25874](https://github.com/huggingface/lerobot/issues/3047), [PR #3048](https://github.com/huggingface/lerobot/pull/3048), [v0.7.0 roadmap](https://github.com/huggingface/lerobot/issues/3832), [strands-labs/robots #4257](https://github.com/strands-labs/robots/issues/4257)

### Normalization details

Normalization is stored in two layers:

1. **Strategy declaration** in `config.json`: `normalization_mapping` maps feature types (`VISUAL`, `STATE`, `ACTION`) to strategies (`IDENTITY`, `MEAN_STD`, `QUANTILE`). Also declared in `policy_preprocessor.json` step config (`norm_map`).
2. **Statistics values** in companion safetensors files:
   - `policy_preprocessor_step_N_normalizer_processor.safetensors` — normalization stats for inputs
   - `policy_postprocessor_step_0_unnormalizer_processor.safetensors` — denormalization stats for outputs
   - OR `norm_stats.json` for OpenPI-compatible checkpoints

### Validation at load time

- `validate_visual_features_consistency()` checks dataset camera names against policy expectations. Suggests `--rename_map` on mismatch.
- No action dimension validation at load time — mismatch discovered at first forward pass.
- No normalization compatibility check — wrong strategy silently degrades.
- No embodiment check — `robot_type` is a free-form string with no validation.

Source: [LeRobot v3 dataset docs](https://huggingface.co/docs/lerobot/lerobot-dataset-v3), [data-formats survey section 1](data-formats-at-interfaces.md)

### Gaps relative to CheckpointDescriptor / InferenceServerDescriptor

- **No action semantics**: `output_features` declares shape `[7]` but not what each dimension means.
- **No camera role annotations**: camera slot names (e.g., `base_0_rgb`, `left_wrist_0_rgb`) follow a naming convention but lack formal role metadata.
- **No embodiment reference**: nothing in the config declares which robot(s) this checkpoint was trained on. Robot identity only appears in HF model card tags (e.g., `robot:Franka-Panda`).
- **No fine-tuning lineage in config**: `base_model` lives in the HF model card, not in `config.json`.
- **Schema varies by architecture**: `Pi0Config`, `DiffusionConfig`, and `ACTConfig` have different field sets with no common base schema.
- **Pipeline steps are runtime-bound**: `registry_name` values require LeRobot Python to resolve. Not a standalone spec.
- **No server introspection**: the PolicyServer has no capability discovery endpoint. The client must know what policy to load.
- **Wire format is opaque**: pickle-serialized gRPC payloads prevent cross-language clients and pose security risks (CVE-2026-25874). The planned replacement ([PR #3048](https://github.com/huggingface/lerobot/pull/3048)) fixes security but keeps the format opaque — safetensors+JSON inside `bytes` fields, not typed protobuf messages. Still unmerged after 7 months.

---

## 4. vLLM / vLLM-Omni — Model Registry & Transforms

### Concept name

Model registration — VLA models are registered in the model registry with associated processor and transform classes.

### vLLM (mainline) — minimal VLA support

Mainline vLLM supports a single VLA model: **OpenVLA** (`OpenVLAForActionPrediction`), registered in `_MULTIMODAL_MODELS` in `vllm/model_executor/models/registry.py`.

**`OpenVLAConfig`** (in `vllm/transformers_utils/configs/openvla.py`):

| Field | Default | Purpose |
| --- | --- | --- |
| `n_action_bins` | 256 | Number of discretized action bins |
| `image_sizes` | `[224, 224]` | Input resolution per backbone |
| `image_token_index` | 32000 | Vocab token ID for image placeholder |
| `timm_model_ids` | list | Vision backbone IDs (DINOv2 + SigLIP) |
| `use_fused_vision_backbone` | bool | Architecture flag |

OpenVLA discretizes actions into token IDs — vLLM treats it as a standard token-generation model. No `action_dim`, no action space bounds, no normalization stats, no camera names in the config. No robotics-specific API endpoints — served through standard completion endpoints.

Source: [vLLM model registry](https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/models)

### vLLM-Omni — extended VLA support

vLLM-Omni adds continuous-action VLA support (pi0, pi0.5) with three dedicated runtime modules (AR, Diffusion, Generation) connected via a stage graph.

Each VLA model requires:

1. **Model class** in the model registry (e.g., `Pi0ForConditionalGeneration`) — architecture definition.
2. **Processor class** (e.g., `processor_pi0.py`) — handles image preprocessing: RGB conversion, HWC→CHW, resize with padding to 224x224, normalization to [-1, 1].
3. **Per-robot Transform class** (e.g., `DroidTransform`, `AlohaTransform`) — handles robot-specific observation remapping.

Source: [vLLM-Omni RFC #6524](https://github.com/vllm-project/vllm-omni/issues/6524), [Pi0.5 PR #6950](https://github.com/vllm-project/vllm-omni/pull/6950)

### Metadata in processor configs

Processor configs contain:

- Image resolution (target size for resize)
- Normalization parameters (mean, std for [-1, 1] range)
- Tokenizer reference (for language inputs)

These are stored as HuggingFace processor config files alongside the checkpoint, following the `transformers` library conventions.

### Per-robot transforms (independently reimplemented)

`DroidTransform` and other transform classes are **independently reimplemented** from OpenPI's adapter classes. They perform the same logical operations (camera remapping, joint sign flips, gripper conversion) but as separate codebases. Known divergences:

- vLLM-Omni's `DroidTransform` stitches all camera views into a single composite image (352x640), while OpenPI keeps views as separate tensors.
- Quantile normalization initially returned `None` for pi0.5 — same checkpoint, different outputs.
- Observation key whitelisting silently drops unrecognized fields from the client.

Source: [deployment-topology survey section 3](deployment-topology.md)

### Server introspection

**No metadata API.** The `/v1/realtime/robot/openpi` endpoint accepts observations and returns action chunks. No capability discovery, no schema exchange. The client must know the model's expected observation format a priori.

### Gaps relative to CheckpointDescriptor / InferenceServerDescriptor

- **Adapter logic duplicated as code**: the same robot-specific transforms exist in OpenPI, vLLM-Omni, and SGLang as three independent Python implementations with no shared declarative description.
- **No server introspection**: cannot programmatically discover what the server expects.
- **No robot coverage declaration**: which robots are supported is determined by which `*Transform` classes exist in the codebase, not by metadata.

---

## 5. SGLang — Checkpoint-Config-Driven Adapters

### Concept name

Checkpoint config + action metadata endpoint — SGLang is the only server with an explicit metadata introspection API.

### How VLA models are configured

SGLang takes a **checkpoint-config-driven** approach instead of hardcoded per-robot adapter classes. It resolves per-checkpoint camera names, state dimensions, and action dimensions from registered LeRobot checkpoint configs via `Pi05PipelineConfig`:

| Field | Source | Purpose |
| --- | --- | --- |
| `action_dim` | checkpoint `max_action_dim` | Max action dimensionality |
| `output_action_dim` | `output_features.action.shape[0]` | Actual output action dimensions |
| `action_horizon` | checkpoint `chunk_size` | Steps per action chunk |
| `n_action_steps` | checkpoint config | Steps to execute |
| `state_dim` | checkpoint `max_state_dim` | Max proprioception dimensions |
| `image_size` | `image_resolution` | Image height x width |
| `image_keys` | input features where `type == "VISUAL"` | Camera name tuple |
| `empty_cameras` | checkpoint config | Placeholder camera slots |
| `default_num_inference_steps` | config | Diffusion denoising steps |
| `max_token_len` | tokenizer config | Tokenizer max length |
| `paligemma_variant` | config | VLM backbone variant |
| `action_expert_variant` | config | Action head variant |

This is lighter-weight than OpenPI's handwritten `*Inputs` classes — the adapter behavior is derived from the checkpoint config rather than maintained as a separate codebase per robot.

Source: [SGLang pi0.5 cookbook](https://docs.sglang.io/cookbook/vla/OpenPI/Pi0.5), [SGLang multimodal_gen source](https://github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen)

### API endpoints

SGLang exposes four endpoints:

| Endpoint | Transport | Purpose |
| --- | --- | --- |
| `/v1/actions/generations` | HTTP POST | Primary action generation (client sends model-native keys) |
| `/v1/actions/metadata` | HTTP GET | **Metadata introspection** — returns model configuration |
| `/v1/actions/realtime` | WebSocket | Real-time streaming actions |
| `/openpi/policy` | WebSocket | OpenPI-compatible endpoint (accepts robot-native keys) |

### `/v1/actions/metadata` response

This endpoint returns a rich JSON object about the loaded model. Response structure:

```json
{
  "object": "action.metadata",
  "model": "<model-name>",
  "model_path": "<HF-repo-or-path>",
  "policy_family": "<family>",
  "input": {
    "image_keys": ["base_0_rgb", "left_wrist_0_rgb", ...],
    "image_size": [224, 224],
    "state_dim": 32
  },
  "output": {
    "action_type": "continuous",
    "action_horizon": 10,
    "action_dim": 7,
    "padded_action_dim": 32,
    "dtype": "float32"
  },
  "runtime": {
    "materialize_dtype": "...",
    "enable_autocast": true,
    "cuda_graph": { ... },
    "parallelism": { ... }
  },
  "defaults": {
    "num_inference_steps": 10,
    "prefix_cache": true,
    "cuda_graph": true
  },
  "capabilities": {
    "exact_prefix_cache": true,
    "cuda_graph": true,
    "realtime_websocket": true,
    "openpi_websocket": true,
    "batch_inputs": true,
    "multiple_candidates": true
  }
}
```

This is the **only introspection API** across all surveyed servers. It allows a client to programmatically discover what the server expects without prior knowledge. Notable: it distinguishes `action_dim` (actual) from `padded_action_dim` (zero-padded maximum), declares `action_type` ("continuous"), and exposes server capabilities (which API surfaces are available).

The `/v1/actions/generations` response also includes structured output metadata: `data[].action.type`, `data[].action.dtype`, `data[].action.shape` `[horizon, dim]`, `data[].action.values`, plus `usage` with `batch_size`, `action_horizon`, `action_dim`, `denoise_steps`.

Source: [SGLang pi0 issue #18266](https://github.com/sgl-project/sglang/issues/18266), [SGLang multimodal_gen source](https://github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen), [RLinf SGLang adapter docs](https://rlinf.readthedocs.io/en/latest/rst_source/extending/sglang_embodied_model.html)

### Gaps relative to InferenceServerDescriptor

- **No action semantics in metadata**: `action_dim` and `action_type: "continuous"` tell you the dimensionality and that actions are continuous (not discretized), but not what each dimension means (EE delta? joint position? which joints?).
- **No normalization metadata exposed**: the client doesn't learn what normalization the server applies or what stats it uses. Normalization is baked into the pipeline.
- **No embodiment information**: which robots are supported is not in the metadata response. The checkpoint determines this implicitly.
- **No base model lineage**: fine-tuning provenance is not exposed.
- **Limited transform description**: the metadata says what keys to send but not what preprocessing the server performs on images or state vectors.

---

## 6. LEAPP / Triton — ONNX Bundle Metadata

### Concept name

LEAPP (Learning, Evaluation, and Adaptation for Physical Policies) uses **tensor semantics** and **modality configs** as its metadata system. Triton Inference Server uses a `config.pbtxt` model configuration.

### LEAPP tensor semantics

LEAPP is the **only ecosystem with declarative action semantics metadata**. The `TensorSemantics` system provides:

| Property | Example | Purpose |
| --- | --- | --- |
| `kind` | `JOINT_POSITION`, `JOINT_VELOCITY`, `GRIPPER_POSITION` | Semantic type annotation per tensor |
| `element_names` | `["hip_l", "knee_l", "ankle_l", ...]` | Per-dimension joint/element names |
| `extra.frame` | `"base_link"` | Reference frame (extensible field) |
| `extra.units` | `"radians"` | Physical units (extensible field) |

Source: [LEAPP tensor semantics docs](https://nvidia-isaac.github.io/leapp/semantics/usage.html)

### ONNX export — what metadata survives

LEAPP traces the full PyTorch pipeline (preprocessing + backbone + action head) and exports it as an ONNX bundle. At export time:

- `--embodiment_tag` and `--joint_config` parameters determine which robot-specific preprocessing and output formatting is baked in.
- Normalization operations are baked into the graph as Sub/Div/BatchNormalization ops.
- The resulting ONNX model is self-contained for deployment — no external norm_stats needed.

**What's in the ONNX bundle**: Input/output tensor names, shapes, and dtypes. Standard ONNX `metadata_props` (string→string key-value pairs) can carry arbitrary metadata, but there is no standard schema for robotics-specific properties. LEAPP's tensor semantics are attached to the Python-side pipeline objects, not to the exported ONNX graph itself.

Source: [LEAPP docs](https://nvidia-isaac.github.io/leapp/), [ONNX IR spec](https://github.com/onnx/onnx/blob/main/docs/IR.md)

### Triton model config (config.pbtxt)

Triton Inference Server uses a Protocol Buffers text format for model configuration:

```
name: "gr00t_model"
platform: "onnxruntime_onnx"
input [
  { name: "observations" data_type: TYPE_FP32 dims: [-1, 103] }
]
output [
  { name: "actions" data_type: TYPE_FP32 dims: [-1, 29] }
]
instance_group [{ kind: KIND_GPU }]
```

This declares tensor names, shapes, and dtypes — structural metadata only. No semantic annotations, no normalization metadata, no robot or embodiment information.

Source: [Triton model configuration docs](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html)

### Gaps relative to CheckpointDescriptor / InferenceServerDescriptor

- **Tensor semantics don't survive ONNX export**: LEAPP's `TensorSemantics` exist at the Python API level but are not serialized into the ONNX graph metadata.
- **No camera/observation semantics**: LEAPP tensor semantics focus on action outputs, not on what observation inputs mean.
- **Triton config is purely structural**: tensor shapes and types, no semantic metadata.
- **`extra` fields have no guaranteed schema**: `frame` and `units` are conventions, not validated fields.
- **NVIDIA-specific**: not a community standard, not adopted outside NVIDIA's stack.

---

## 7. ONNX — Graph Metadata

### Concept name

Model metadata — `ModelProto` fields and `metadata_props` key-value pairs.

### Explicitly declared metadata

| Field | Type | Purpose |
| --- | --- | --- |
| `opset_import` | list | Operator set versions used |
| `ir_version` | int | ONNX IR version |
| `producer_name` / `producer_version` | string | Exporting framework |
| `domain` | string | Model domain |
| `doc_string` | string | Free-text documentation |
| `metadata_props` | list of `{key: string, value: string}` | Arbitrary key-value metadata |
| Input/output `ValueInfoProto` | structured | Tensor name, element type (28 dtypes), shape (with symbolic dims) |
| Type denotations | enum | `TENSOR`, `IMAGE`, `AUDIO`, `TEXT` — 4 broad categories |
| Dimension denotations | enum | `DATA_BATCH`, `DATA_CHANNEL`, `DATA_FEATURE` |
| Image metadata | string properties | `Image.BitmapPixelFormat`, `Image.ColorSpaceGamma`, `Image.NominalPixelRange` |

Source: [ONNX IR spec](https://github.com/onnx/onnx/blob/main/docs/IR.md), [ONNX type denotations](https://github.com/onnx/onnx/blob/main/docs/TypeDenotation.md)

### What the WBC pipeline stores in practice

The nvidia-industrial-wbc-pipeline exports ONNX models with:

- Input tensor: `observations` (float32, shape `[-1, 103]` for flat terrain)
- Output tensor: `actions` (float32, shape `[-1, 29]` for 29-DOF humanoid)
- Normalization baked into graph as Sub/Div/BatchNormalization ops
- `metadata_props`: typically only `producer_name` and `producer_version`

The graph itself becomes a normalization spec — the Sub/Div ops encode the mean/std values — but there's no metadata *declaring* this pattern. A consumer has no way to know whether the model expects pre-normalized or raw inputs without inspecting graph structure.

### Gaps relative to CheckpointDescriptor

- **No robotics vocabulary**: type denotations are limited to 4 broad categories. No controlled vocabulary for joint positions, end-effector deltas, camera roles, or proprioception.
- **No normalization metadata**: whether normalization is baked into the graph or external is undeclared.
- **`metadata_props` is unstructured**: string→string with no schema enforcement or community conventions for robotics.
- **No embodiment or training provenance**: which robot, which dataset, which training configuration.
- **A 2022 proposal** ([onnx/onnx#3958](https://github.com/onnx/onnx/issues/3958)) for RDF-based extensible metadata remains open with no implementation.

---

## 8. NVIDIA NIM — Model Manifests

### Concept name

Model manifest — `model_manifest.yaml` shipped inside NIM containers.

### Explicitly declared metadata

| Field | Purpose | Example |
| --- | --- | --- |
| Profile tags: `backend` | Inference engine | `vllm`, `sglang`, `trtllm` |
| Profile tags: `precision` | Weight precision | `bf16`, `fp8`, `nvfp4` |
| Profile tags: `tp` / `pp` | Tensor/pipeline parallelism | `tp2`, `pp1` |
| Profile tags: `lora` | LoRA support | `lora` (presence = supported) |
| Memory annotations | Estimated VRAM per GPU | `gpu_memory: {high: 40GB, low: 20GB}` |
| Hardware matching | GPU device IDs for auto-selection | Device ID list |
| Profile priority | Selection order | Priority-ordered list |

Source: [NIM model profiles docs](https://docs.nvidia.com/nim/large-language-models/latest/deployment/model-profiles-and-selection.html)

### Scope limitation

NIM manifests are **scoped entirely to LLM/VLM deployment**. They solve the problem of "which hardware profile should I use for this LLM?" — not model-dataset compatibility or action space validation.

- No input/output tensor specs
- No action space, observation space, or normalization metadata
- No embodiment or robot configuration
- No robotics models distributed as NIMs — GR00T and Isaac models use separate deployment mechanisms
- Schema is not published as an open spec

### Conceptual relevance

The **profile selection pattern** (matching model requirements to available hardware via tagged profiles with priority ordering) is conceptually interesting. A similar pattern could match model requirements to available robots or datasets. But the format itself is not applicable to Physical AI checkpoint or server metadata.

### Gaps relative to CheckpointDescriptor / InferenceServerDescriptor

- **Wrong domain entirely**: NIM manifests describe LLM deployment hardware profiles, not model I/O contracts.
- **No model capability metadata**: beyond precision and parallelism, no description of what the model does.
- **Proprietary**: not adopted outside NVIDIA's NIM ecosystem.

---

## Cross-Cutting Analysis

### Coverage matrix

| Property needed | HF Cards | OpenPI config | LeRobot config | LeRobot PolicyServer | vLLM-Omni | SGLang | LEAPP | ONNX | NIM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Architecture / model family** | `pipeline_tag` | `type` | `type` | From checkpoint | model registry | checkpoint config | N/A | `producer_name` | N/A |
| **Base model / lineage** | `base_model` | N/A | N/A (in card) | N/A | N/A | N/A | N/A | N/A | N/A |
| **Input tensor shapes** | No | `input_features` | `input_shapes` | From checkpoint `input_features` | processor config | checkpoint config | ONNX graph | Yes | No |
| **Output tensor shapes** | No | `output_features` | `output_shapes` | From checkpoint `output_features` | processor config | `action_dim` | ONNX graph | Yes | No |
| **Action chunk horizon** | No | `n_action_steps` | `n_action_steps` / `horizon` | Client-set `actions_per_chunk` | N/A | `action_horizon` | N/A | N/A | No |
| **Action semantics (what dims mean)** | No | No | No | No | No | No | **Yes** (`kind`, `element_names`) | No | No |
| **Normalization strategy** | No | `normalization_mapping` | `normalization_mapping` | **Yes** (`policy_preprocessor.json` step config) | code | code | Baked into graph | No | No |
| **Normalization stats** | No | `norm_stats.json` | processor safetensors | **Yes** (processor safetensors, loaded by pipeline) | `_build_norm_buffers` | code | Baked into graph | No | No |
| **Processing pipeline** | No | Code | Code | **Yes** (`policy_preprocessor.json` + `policy_postprocessor.json`) | Code | Code | Baked into graph | No | No |
| **Camera slot names** | No | `input_features` | `input_features` | From checkpoint + client `rename_map` | processor | `camera_names` | N/A | No | No |
| **Camera roles (wrist/world)** | No | Code only | Naming convention | Naming convention | Code only | Naming convention | N/A | No | No |
| **Supported embodiments** | No | Code (class name) | No | No | Code (class name) | No | `--embodiment_tag` | No | No |
| **Server introspection API** | N/A | **No** | N/A | **No** (client configures server) | **No** | **Yes** (`/v1/actions/metadata`) | Via Triton | N/A | N/A |
| **Deployment hardware profile** | No | No | No | No | No | No | No | No | **Yes** |

### Key findings

1. **LeRobot's `config.json` + processing pipelines are the richest existing checkpoint metadata** — `config.json` contains normalization strategy, feature shapes, action chunk parameters, and camera slot names. The `policy_preprocessor.json` / `policy_postprocessor.json` files go further, serializing the full inference-time processing chain as ordered step lists with companion safetensors for learned statistics. But there is no formal schema, no action semantics, no embodiment reference, and the pipeline step names are bound to LeRobot's Python runtime.

2. **SGLang's `/v1/actions/metadata` is the only server introspection API** — returning structured metadata including `input` (image_keys, image_size, state_dim), `output` (action_type, action_horizon, action_dim, padded_action_dim, dtype), `capabilities` (which API surfaces are available), and runtime config. This is the closest existing implementation to an InferenceServerDescriptor. But it omits action semantics (what each dimension means), normalization details, and supported embodiments. LeRobot's PolicyServer has no introspection — it inverts the pattern (client configures server).

3. **LEAPP's tensor semantics are the only action semantics metadata** — `kind`, `element_names`, and extensible `extra` fields. But they're NVIDIA-specific, don't survive ONNX export, and don't cover observation/camera semantics.

4. **Adapter transforms are the biggest gap across all communities** — the robot-specific transforms (camera remapping, joint sign flips, gripper conversion) that determine whether a checkpoint actually works with a given robot are hardcoded as Python constants in every server implementation, with no declarative representation anywhere. LeRobot's serialized pipelines partially address this for normalization and tokenization, but camera remapping and joint reordering are still handled via the client-provided `rename_map` (a flat dict, not a rich descriptor).

5. **No community declares embodiment compatibility** — which robot(s) a checkpoint supports is determined by which adapter classes exist in server source code, not by metadata attached to the checkpoint. LeRobot model cards may carry tags (e.g., `robot:Franka-Panda`) but these are free-form strings with no validation or structured querying.

6. **Normalization is now handled four different ways**: LeRobot checkpoints ship serialized processor pipelines with safetensors stats; OpenPI stores external `norm_stats.json`; LEAPP/WBC bake normalization into the ONNX graph; vLLM-Omni reimplements normalization in server code. No metadata convention declares which approach a checkpoint uses.

7. **LeRobot's serialized processing pipelines are a step toward TransformDescriptor** — the `policy_preprocessor.json` / `policy_postprocessor.json` format (ordered step list with registry names, configs, and state files) is conceptually close to the TransformDescriptor in our data model. Key gap: step names are runtime-bound (`registry_name` requires LeRobot Python to resolve), not a standalone specification that other servers could consume.
