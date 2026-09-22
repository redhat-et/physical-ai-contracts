# Deployment Topology & Adapter Architecture

**Date**: 2026-09-21 (updated; originally 2026-09-20)
**Purpose**: Document how the logical functions from our architecture diagrams (Policy Server, Embodiment Adapter, Robot Controller) map to actual deployment units across ecosystems. Examines where adapter logic lives, what it does, and why the current fragmentation matters for interoperability.
**Input**: Source code analysis of OpenPI, vLLM-Omni, and GR00T/LEAPP; web research on deployment patterns.

---

## 1. The Logical vs. Physical Architecture

Our architecture diagrams (01-logical-blocks.svg) show three distinct functions in the inference path:

1. **Policy Server** — runs the VLA model, produces action chunks
2. **Embodiment Adapter** — converts between model I/O semantics and robot-specific conventions
3. **Robot Controller** — low-level control (ros2_control), hardware interface

In practice, these functions are deployed differently depending on the ecosystem. The Embodiment Adapter is not a standalone component — it's embedded in the Policy Server (OpenPI, vLLM-Omni) or baked into the model artifact (LEAPP).

---

## 2. Deployment Patterns by Ecosystem

### 2.1 OpenPI: Adapter embedded in server, off-robot

**Architecture**: GPU workstation runs Policy Server + Embodiment Adapter as a single Python process. Robot runs `openpi-client` (thin WebSocket client) + ros2_control.

```
[GPU Workstation]                    [Robot]
┌──────────────────────┐             ┌──────────────────┐
│ OpenPI Server        │  WebSocket  │ openpi-client    │
│ ┌──────────────────┐ │◄───────────►│ (thin: resize    │
│ │ Policy (pi0.5)   │ │             │  images, send    │
│ ├──────────────────┤ │             │  raw obs dict)   │
│ │ DroidInputs /    │ │             ├──────────────────┤
│ │ AlohaInputs      │ │             │ ros2_control     │
│ │ (adapter classes)│ │             │ (hw interface)   │
│ ├──────────────────┤ │             └──────────────────┘
│ │ Normalization    │ │
│ │ (norm_stats.json)│ │
│ └──────────────────┘ │
└──────────────────────┘
```

The client sends **robot-native** observation keys (e.g., `observation/exterior_image_1_left` for DROID). The server-side adapter class remaps them to model-expected keys (e.g., `base_0_rgb`).

**Source**: [OpenPI remote inference docs](https://github.com/Physical-Intelligence/openpi/blob/main/docs/remote_inference.md) — "This is useful for running inference on more powerful GPUs off-robot, and also helps keep the robot and policy environments separate."

**On-device exception**: On Jetson-class devices, all three functions can run on-device (server on `localhost:8000`), but this is uncommon.

### 2.2 vLLM-Omni: Partial adapter, production serving features

**Architecture**: Same topology as OpenPI (GPU server + thin client), with OpenPI-compatible endpoint at `/v1/realtime/robot/openpi`. Adds production serving features (batching, disaggregated prefill/decode, multi-GPU).

vLLM-Omni **does** have per-robot adapter code (e.g., `DroidTransform`), independently reimplemented from OpenPI's `DroidInputs`. For supported robots (DROID, ALOHA), the client can send the same robot-native observation dict to either server.

**Not a full drop-in replacement** — see Section 3.

**Sources**: [vLLM-Omni RFC #6524](https://github.com/vllm-project/vllm-omni/issues/6524), [Pi0.5 PR #6950](https://github.com/vllm-project/vllm-omni/pull/6950)

### 2.3 GR00T/LEAPP: Adapter baked into model artifact

**Architecture**: Fundamentally different. LEAPP traces the full PyTorch pipeline (preprocessing, backbone, action head) and exports it as an ONNX bundle. The adapter logic is baked into the artifact at export time.

```
[Export time]                         [Deploy time]
┌─────────────────────┐               ┌─────────────────────┐
│ LEAPP export        │               │ Triton Server       │
│ --embodiment_tag    │──► ONNX ──►  │ (runs ONNX stages)  │
│ --joint_config      │    bundle     ├─────────────────────┤
│                     │               │ isaac_ros_deploy    │
│ Bakes in:           │               │ (ROS 2 bridge)      │
│ - preprocess_state  │               ├─────────────────────┤
│ - normalization     │               │ ros2_control        │
│ - action_head       │               │ (hw interface)      │
└─────────────────────┘               └─────────────────────┘
```

**LEAPP tensor semantics** provide partial metadata: kind annotations (`JOINT_POSITION`, `JOINT_VELOCITY`), per-dimension element names (`hip_l`, `knee_l`), and extensible fields for `frame` and `units`. This is the only ecosystem with any declarative action semantics metadata.

For humanoids, a two-tier system applies: GR00T VLA at ~10 Hz produces latent action tokens, SONIC whole-body controller at 50-120 Hz decodes them into motor commands. SONIC has its own C++/TensorRT deployment stack.

**Sources**: [LEAPP docs](https://nvidia-isaac.github.io/leapp/), [LEAPP tensor semantics](https://nvidia-isaac.github.io/leapp/semantics/usage.html), [isaac_ros_deploy](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_deploy), [GR00T E2E deployment guide](https://docs.nvidia.com/learning/physical-ai/gr00t-e2e-workflow/latest/real-robot-workflow/real-deployment.html)

### 2.4 SGLang: Client-side adapter, optimized VLA runtime

**Architecture**: Same remote-inference topology, but with two key differences from OpenPI/vLLM-Omni:

1. **Checkpoint-config-driven adapters (not hardcoded classes).** SGLang resolves per-checkpoint camera names, state dimensions, and action dimensions from registered LeRobot checkpoint configs. On its native API (`/v1/actions/generations`), the client sends model-native keys (e.g., `base_0_rgb`). On its OpenPI-compatible endpoint (`/openpi/policy`), it accepts robot-native keys. This is a lighter-weight approach than OpenPI's handwritten `*Inputs` classes or vLLM-Omni's `*Transform` classes.

2. **Dedicated VLA runtime.** SGLang's `multimodal_gen` path skips the LLM sampler, logits processor, and paged decode KV cache for VLA workloads (flow matching is not a token decode workload). This avoids the inter-stage coordination overhead that affects vLLM-Omni (see Section 2.5).

3. **Multiple API surfaces.** SGLang offers four endpoints: `/v1/actions/generations` (HTTP), `/v1/actions/metadata` (HTTP), `/v1/actions/realtime` (WebSocket), and `/openpi/policy` (WebSocket, OpenPI-compatible). The OpenPI endpoint makes SGLang a near-drop-in replacement for OpenPI clients.

**General comparison with vLLM-Omni:**

| | SGLang | vLLM-Omni |
| --- | --- | --- |
| **Origin** | LMSYS (Stanford/Berkeley) → RadixArk ($400M) | UC Berkeley → vLLM Inc. |
| **Scale** | 400K+ GPUs (xAI Grok), trillions tok/day | Default backend for most cloud APIs |
| **KV cache** | RadixAttention (prefix tree, ~29% higher throughput at 8B) | PagedAttention |
| **VLA architecture** | Dedicated `multimodal_gen`, single-process | AR + Diffusion + Generation runtimes, stage graph |
| **Per-robot adapters** | Checkpoint-config-driven | Server-side `DroidTransform` etc. |
| **OpenPI compatibility** | `/openpi/policy` WebSocket + own native API | `/v1/realtime/robot/openpi` endpoint |

**Sources**: [SGLang pi0.5 docs](https://docs.sglang.io/cookbook/vla/OpenPI/Pi0.5), [SGLang pi0 issue #18266](https://github.com/sgl-project/sglang/issues/18266), [RLinf SGLang adapter docs](https://rlinf.readthedocs.io/en/latest/rst_source/extending/sglang_embodied_model.html)

### 2.5 vLLM-Omni: Corrected architecture characterization

vLLM-Omni does NOT route VLA through an AR token decode path. It has three dedicated runtime modules — AR, Diffusion, and Generation — connected via a stage graph. Pi0/pi0.5 is decomposed into an AR VLM stage + a flow-matching action head stage, each served by the appropriate runtime.

The overhead identified by the [Robion paper](https://arxiv.org/abs/2609.12075) (4.4× lower SLO attainment vs monolithic) comes from **inter-stage buffer inspection** in the disaggregated architecture — coordination cost at each denoising step that's disproportionate for VLA's millisecond-scale steps. This is a disaggregation tax, not a wrong-pipeline problem.

**Source**: [vLLM-Omni architecture](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/architecture_overview/), [Robion paper](https://arxiv.org/abs/2609.12075)

---

## 3. OpenPI vs. vLLM-Omni: Near Drop-In, Not Drop-In

For supported robots, the OpenPI client can point at either server. Both accept robot-native observation dicts and return `float[horizon × action_dim]` action chunks. But there are concrete interoperability gaps:

| Issue | Impact |
| --- | --- |
| **Independently reimplemented transforms** | OpenPI's `DroidInputs` and vLLM-Omni's `DroidTransform` are separate codebases performing the same camera remapping, joint sign flips, gripper conversion. Any subtle difference produces silently wrong robot actions. |
| **Normalization bugs** | vLLM-Omni's quantile normalization initially returned `None` for pi0.5. Same checkpoint, silently different outputs. |
| **Observation key whitelisting** | vLLM-Omni whitelists accepted keys; unrecognized fields from the client are silently dropped. |
| **Parameter forwarding** | vLLM-Omni's OpenPI endpoint initially only forwarded `robot_obs`, `session_id`, `reset`. Parameters like `num_inference_steps` were silently ignored. |
| **Robot coverage** | OpenPI supports any robot via user-written `*Inputs`/`*Outputs` classes. vLLM-Omni's model registry is less extensible — unsupported robots require contributing to the vLLM-Omni codebase. |

**Sources**: [vLLM-Omni RFC #6524](https://github.com/vllm-project/vllm-omni/issues/6524), [vLLM-Omni quantile normalization fix](https://github.com/vllm-project/vllm-omni/pull/6950)

### 3.1 Wire Format: Three msgpack numpy Dialects (Verified)

The msgpack numpy serialization incompatibility is real and well-documented. Three dialects exist for encoding numpy arrays over msgpack:

| Dialect | Marker fields | `kind` field | Used by |
| --- | --- | --- | --- |
| **openpi-client** | `{__ndarray__, dtype, shape, data}` | Absent | OpenPI client/server |
| **vLLM-native** | `{nd, type, kind, shape, data}` | dtype kind char (e.g., `'f'`) | vLLM-Omni internal |
| **msgpack-numpy package** | `{nd, type, kind, shape, data}` | `b''` (empty bytes) | `msgpack-numpy` PyPI package |

vLLM-Omni's OpenPI endpoint initially only accepted vLLM-native markers, rejecting arrays packed by the `msgpack-numpy` package or OpenPI's own client. PR [#6051](https://github.com/vllm-project/vllm-omni/pull/6051) fixed this by making the decoder accept all three formats inbound. Outbound responses continue using openpi-client markers.

The `kind` field is the structural differentiator — its presence distinguishes numpy array markers from ordinary user dictionaries. This means a client that auto-detects server type (as the robotics-playground does) must inspect response payloads for `__ndarray__` vs. `nd` markers.

**Implication for interoperability**: A client that packs observations using one dialect may fail on a server expecting another. The playground's vendored `msgpack_numpy.py` with auto-detection is a correct — if fragile — workaround. A standard wire format for numpy arrays in robotic observations would eliminate this class of bug.

**Sources**: [vLLM-Omni PR #6051](https://github.com/vllm-project/vllm-omni/pull/6051), [msgpack-numpy package](https://github.com/lebedov/msgpack-numpy)

### 3.2 Image Preprocessing: Server-Side Everywhere (Verified)

All three servers accept raw observations from clients and perform image preprocessing server-side. However, the preprocessing pipelines differ in important ways.

**OpenPI** — two-stage server-side preprocessing:

1. **Adapter class** (e.g., `DroidInputs._parse_image`): converts float32 CHW → uint8 HWC only. No resize, no normalization.
2. **Model pipeline** (`preprocess_observation` in `model.py`): resizes to 224×224 via `resize_with_pad` (aspect-preserving, bilinear, symmetric padding with fill=-1.0), normalizes uint8 [0,255] → float32 [-1,1].

**vLLM-Omni** — two parallel preprocessing paths:

1. **Processor** (`processor_pi0.py`): RGB conversion, HWC→CHW, resize with pad to 224×224, normalize to [-1,1]. Closely mirrors OpenPI's model pipeline (same normalization, same pad fill value).
2. **Per-robot transform** (`DroidTransform`): center crop (95%), resize each view to 176×320, then **stitches all views into a single 352×640 composite image** (wrist repeated 2× on top, two exterior views on bottom). This is architecturally different from OpenPI/SGLang, which keep camera views as separate tensors.

**SGLang** — single-stage server-side preprocessing (`pi05_preprocess.py`):
RGB conversion → HWC→CHW → scale to [0,1] → resize with pad to 224×224 → normalize to [-1,1]. Closely matches OpenPI's model pipeline. No per-robot transforms on the server — the client sends model-native camera keys (e.g., `base_0_rgb`). SGLang resolves per-checkpoint camera/state config from registered LeRobot checkpoints.

| | OpenPI | vLLM-Omni | SGLang |
| --- | --- | --- | --- |
| **Client sends** | Robot-native obs keys | Robot-native obs keys | Model-native obs keys (or robot-native via `/openpi/policy`) |
| **Server-side per-robot adapter** | Yes (`DroidInputs` etc.) | Yes (`DroidTransform` etc.) | Checkpoint-config driven (camera names, state dim) |
| **Image resize** | 224×224 (server) | 224×224 (processor) or 176×320 (DroidTransform) | 224×224 (server) |
| **View handling** | Separate tensors | Stitched composite (DroidTransform) or separate (processor) | Separate tensors |
| **Normalization** | uint8 → float32 [-1,1] | uint8 → float32 [-1,1] | uint8 → float32 [-1,1] |
| **Missing cameras** | fill -1.0 + mask | fill -1.0 | fill -1.0 + mask |

**Key finding**: The `DroidTransform` view-stitching in vLLM-Omni is a fundamentally different image representation than OpenPI's separate-tensor approach. For the same DROID robot and same pi0.5 checkpoint, the model receives different input depending on which server processes the observation. This suggests `DroidTransform` is specific to a different model variant (video-diffusion-based) rather than the standard pi0/pi0.5 pipeline, which uses the processor path.

**SGLang OpenPI compatibility**: SGLang provides an `/openpi/policy` WebSocket endpoint that accepts the same observation format as OpenPI. Combined with its own `/v1/actions/generations` HTTP API and `/v1/actions/realtime` WebSocket, SGLang offers the most API surface of any server. However, the client must still use model-native camera keys on the native API (the OpenPI endpoint handles remapping).

**Sources**: [OpenPI source — model.py, droid_policy.py](https://github.com/Physical-Intelligence/openpi/tree/main/src/openpi), [vLLM-Omni — processor_pi0.py, droid.py](https://github.com/vllm-project/vllm-omni), [SGLang — pi05_preprocess.py, Pi0.5 cookbook](https://docs.sglang.io/cookbook/vla/OpenPI/Pi0.5)

---

## 4. What the Adapter Classes Actually Do

OpenPI's per-robot adapter classes (`DroidInputs`, `AlohaInputs`, `LiberoInputs`, etc.) are pure Python data transforms. They have zero robotics dependencies — no URDF parsing, no ROS2, no kinematic libraries.

### Concrete transforms performed

| Transform | Example (DroidInputs) | Example (AlohaInputs) |
| --- | --- | --- |
| **Camera name remapping** | `observation/exterior_image_1_left` → `base_0_rgb`, `observation/wrist_image_left` → `left_wrist_0_rgb` | `cam_high` → `base_0_rgb`, `cam_left_wrist` → `left_wrist_0_rgb` |
| **Image format conversion** | float32 CHW → uint8 HWC | Same |
| **State vector assembly** | Concatenate `joint_position` [7] + `gripper_position` [1] | Concatenate bimanual joint positions [14] |
| **Joint sign flips** | None | Hardcoded mask: `[1,-1,-1,1,1,1,1,1,-1,-1,1,1,1,1]` |
| **Gripper space conversion** | None | Aloha linear → pi0 angular space (hardcoded min/max) |
| **Padding** | Zero-pad missing camera slots + `image_mask` | Same |
| **Output slicing** | Take first 8 of 32-dim padded output | Take 14-dim output |

Adding a new robot means writing a new `*Inputs`/`*Outputs` class with hardcoded constants.

**Source**: [OpenPI source — droid_policy.py, aloha_policy.py, libero_policy.py](https://github.com/Physical-Intelligence/openpi/tree/main/src/openpi/policies)

---

## 5. No Ecosystem Uses URDF for Policy Serving

| Ecosystem | URDF/robot description in serving? | What drives per-robot behavior |
| --- | --- | --- |
| OpenPI | No | Hardcoded Python classes per robot |
| vLLM-Omni | No | Hardcoded Python transforms per robot |
| GR00T/LEAPP | No (`--joint_config` + `--embodiment_tag` at export) | Modality config (Python/JSON), baked into ONNX |
| ros2_control | Yes (loads URDF for joint names, limits, HW interfaces) | URDF + controller YAML config |

The Robot Controller (ros2_control) is the only component that actually consumes URDF. The Embodiment Adapter, despite being the function most in need of robot-specific knowledge (joint ordering, sign conventions, sensor mapping), operates on hardcoded constants instead.

---

## 6. Why Adapter and Controller Aren't Bundled

Both are embodiment-specific, but they serve different concerns with different deployment constraints:

| Property | Embodiment Adapter | Robot Controller |
| --- | --- | --- |
| **Converts between** | Model I/O semantics ↔ robot-generic commands | Generic commands ↔ hardware signals (PWM, CAN, EtherCAT) |
| **Frequency** | 5-20 Hz (model inference rate) | 50-1000 Hz (control loop rate) |
| **Compute needs** | GPU (co-located with model for efficiency) | Real-time CPU (latency-critical) |
| **Changes when you swap** | The model (different action semantics) | The robot (different actuators) |
| **Model-agnostic?** | No — depends on model's I/O convention | Yes — doesn't know what model is running |

Bundling adapter + controller would force the GPU onto the robot. The current split keeps GPU-heavy work on powerful hardware and only the thin control loop on constrained robot hardware.

---

## 7. Interoperability Implications

### The adapter is the fragmentation point

The Embodiment Adapter is the function most critical for interoperability and the one with the least standardization. Its logic is:

- **Duplicated** across server implementations (OpenPI, vLLM-Omni, SGLang — each reimplements the same transforms)
- **Hardcoded** as magic numbers in Python (joint sign flip masks, gripper conversion constants, camera name mappings)
- **Not derived** from any declarative descriptor (despite URDF containing much of the needed information)
- **Not associated** with the checkpoint (despite being determined by the checkpoint's training data)

### What a declarative adapter descriptor could enable

If the adapter transforms were driven by a declarative descriptor — associated with the checkpoint and/or the robot — rather than baked into server source code:

1. **Server portability**: Move a model from OpenPI to vLLM-Omni to SGLang without reimplementing per-robot transform code in each server's codebase.
2. **Robot portability**: Deploy the same checkpoint to a new robot by providing a new robot descriptor + EmbodimentMapping, without writing Python adapter classes.
3. **Correctness**: Eliminate the class of bugs where two servers implement the same transform differently (the vLLM-Omni quantile normalization bug, potential joint sign flip mismatches).

### LEAPP tensor semantics as partial prior art

NVIDIA's LEAPP is the only ecosystem with any declarative action semantics metadata (kind annotations, element names, frame/units). However:

- The `extra` fields have no guaranteed schema
- Camera/observation semantics aren't explicitly covered
- It's NVIDIA-specific, not a community standard
- It describes the model's output, not the mapping *to* a specific robot

**Source**: [LEAPP tensor semantics](https://nvidia-isaac.github.io/leapp/semantics/usage.html)

---

## 8. The Two-Protocol Reality

The inference stack bridges two ecosystems with fundamentally different transport models:

| | ML Ecosystem (above split) | Robotics Ecosystem (below split) |
| --- | --- | --- |
| **Language** | Python (PyTorch, JAX) | C++ (real-time safe) |
| **Transport** | HTTP/WebSocket, gRPC | DDS (pub/sub, discovery) |
| **Deployment** | Cloud/GPU servers, containers | On-robot, bare-metal, RT kernel |
| **Rate** | 5-20 Hz (inference) | 50-1000 Hz (control) |
| **Latency concern** | Throughput (batch inference) | Determinism (jitter) |

The split point is at the Robot Controller. **OpenPI chose WebSocket over ROS 2** for practical reasons: no need for ROS 2 on GPU servers, DDS discovery is painful across subnets/firewalls, and ML engineers don't use ROS. The Robot Controller uses ROS 2 because it needs real-time guarantees and hardware abstraction (ros2_control).

### NVIDIA eliminates both protocol overheads

In NVIDIA's on-device stack (isaac_ros_deploy), the `InferenceController` is a ros2_control controller that:

1. Reads joint state from ros2_control state interfaces (shared memory)
2. Runs the full LEAPP pipeline via Triton (preprocessing + inference + postprocessing)
3. Writes targets to ros2_control command interfaces (shared memory)

This collapses Policy Engine + Robot Controller into one process, eliminating: the WebSocket/HTTP overhead, the OpenPI-to-ROS2 bridge, and the ROS 2 topic overhead between adapter and controller. The only remaining ROS 2 boundary is ros2_control's hardware abstraction layer (in-process shared memory).

**Trade-off**: the model must run on-device (Jetson Thor or similar). No powerful GPU on the robot → must fall back to remote inference with all its protocol overhead.

### The missing middle: ros2_control controller with remote inference client

Nobody has built a ros2_control controller that embeds a remote-inference client (WebSocket to OpenPI/vLLM-Omni/SGLang) and directly reads state/writes command interfaces. This would mirror NVIDIA's `InferenceController` pattern but with remote GPU inference, eliminating the glue code and intermediate ROS 2 topic layer. Existing approaches:

- **MoveIt Pro ExecutePolicy** (PickNik): closest thing — Inference Server → Adapter Node (ROS 2 service) → ExecutePolicy → ros2_control. Cleanly integrated but still has an intermediate ROS 2 service layer. Source: [PickNik docs](https://docs.picknik.ai/how_to/vla/connect_a_vla_policy/)
- **GSoC 2026 trajectory upsampling**: enhances JointTrajectoryController to handle sparse 20 Hz VLA outputs via cubic spline interpolation to 1000 Hz hardware rate. Prerequisite for embedding a policy client, but policy still external. Source: [Open Robotics Discourse](https://discourse.openrobotics.org/t/gsoc-2026-physical-ai-inference-and-trajectory-upscaling-for-ros2-control/58037)
- **VLAgents/RobotControlStack**: bypasses ROS 2 entirely, using synchronous Gymnasium-style APIs. Achieves 220 Hz / 0.3 ms delay. Argues ROS 2's async pub/sub is fundamentally wrong for tight policy-control loops. Source: [VLAgents, arXiv:2601.11250](https://arxiv.org/abs/2601.11250)

---

## 9. Other Interop Gaps Beyond the Adapter

Ranked by severity, based on analysis of every interface in the architecture diagram:

| Interface | Severity | Nature |
| --- | --- | --- |
| **Policy Server ↔ Adapter** | **High** | Adapter logic duplicated across 3+ servers as hardcoded Python |
| **Client ↔ Server wire format** | **High** | Three incompatible msgpack numpy serialization dialects (`openpi-client`, `vLLM-native`, `msgpack-numpy` package). Clients that pack arrays in one dialect fail on servers expecting another. vLLM-Omni had to patch its decoder ([PR #6051](https://github.com/vllm-project/vllm-omni/pull/6051)) to accept all three. |
| **Data Collector → Dataset Store** | **High** | 6+ independent ROSbag-to-LeRobot converters, each with different resampling strategies. No metadata records which strategy was used. Source: [LeRobot ROS 2 RFC #4368](https://github.com/huggingface/lerobot/issues/4368) |
| **Image preprocessing pipeline** | **Medium** | All servers preprocess images server-side, but vLLM-Omni's `DroidTransform` stitches views into a composite image while OpenPI/SGLang keep views separate. Same robot + same checkpoint → different model input depending on server. |
| **Model Registry → Policy Server** | **Medium** | SafeTensors converging as weight format, but sidecar loading (config.json, norm_stats, processor files) diverges. Subtle pitfalls: BF16↔FP16 coercion, attention weight interleaving order. |
| **Controller ↔ Simulator** | **Medium** | Controller code portable (ros2_control), but asset formats incompatible: Gazebo/SDF, MuJoCo/MJCF, Isaac Sim/OpenUSD. No standard conversion. Source: [ros2_control Simulator Integrations](https://control.ros.org/jazzy/doc/simulators/simulators.html) |
| **Adapter ↔ Controller** | **Low** | ROS 2 message types converged. Gap is semantic (which joint = which dim), not transport. |
| **Controller ↔ Hardware** | **Low** | ros2_control hardware interface well-defined. |

---

## 10. Open Questions

1. Could the adapter transforms be mechanically derived from a combination of (a) the checkpoint's ActionSemantics/ObsSemantics and (b) the robot's URDF + sensor config? Or is there irreducible manual knowledge (e.g., which gripper convention to use)?
2. Is GR00T's `embodiment_tag` + modality config evolving toward a community standard, or is it NVIDIA-internal?
3. Could Rosetta's YAML contract (from the robotics-rl repo) serve as a starting point for a declarative adapter descriptor?
4. How does the two-tier SONIC architecture (latent action tokens → motor commands) change the adapter boundary? Is SONIC itself an adapter, a controller, or something in between?
5. Is a ros2_control controller with embedded remote-inference client (WebSocket to OpenPI/vLLM-Omni) a viable community contribution? What are the real-time implications of making WebSocket calls from a ros2_control update loop?
6. Is the VLAgents/RobotControlStack argument — that ROS 2's async pub/sub is fundamentally wrong for synchronous policy-control loops — gaining traction in the community?
