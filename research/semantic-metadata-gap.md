# Semantic Metadata Gap: Actions and Observations

Status: Problem definition (no solution proposals).
Last updated: 2026-09-19.

## Summary

The storage/serialization layer for Physical AI artifacts is converging (SafeTensors, LeRobot v3, MP4, HF Hub). The **semantic layer** — what the data *means* — has no standard. This document captures what is missing, what partial solutions exist, and why the gap matters.

---

## 1. The Problem

Interoperability between Physical AI components requires answering compatibility questions that no existing metadata can answer:

- Can this **dataset** be used to fine-tune this **model** for this **robot**?
- Does this **checkpoint's** action output match what the **Embodiment Adapter** expects?
- Does this **dataset's** observation format match what the **policy** requires as input?

Today, engineers answer these questions by reading source code, READMEs, and config files across multiple repos, then writing manual Python glue. Errors (wrong joint ordering, mismatched normalization, incorrect camera mapping) are caught only at runtime or — worse — cause silent training failures.

The root cause: metadata describes **shape and type** (a `float32` array of shape `[7]`) but not **semantics** (7-DOF end-effector delta in meters, base frame, with quaternion in `(x,y,z,w)` order).

---

## 2. Action Semantics

### What needs to be described

For each action dimension or action group in a dataset or checkpoint:

| Property | Example values | Why it matters |
| ---------- | --------------- | ---------------- |
| **Representation type** | EE delta, EE absolute pose, joint position, joint velocity, joint effort | Determines what conversion the Embodiment Adapter must perform |
| **Dimensionality & grouping** | dims 0-2 = position, dims 3-6 = orientation, dim 7 = gripper | A flat `float[8]` is meaningless without this |
| **Units** | meters, radians, degrees, normalized [0,1], m/s | Silent unit mismatch causes subtle failures |
| **Coordinate frame** | base_link, ee_link, world, camera_frame | Frame mismatch = wrong direction of motion |
| **Orientation convention** | quaternion (x,y,z,w) vs (w,x,y,z), euler XYZ vs ZYX, rotation matrix | ROS uses (x,y,z,w), Isaac/MuJoCo use (w,x,y,z) — silent data corruption |
| **Action group identity** | `right_arm`, `left_arm`, `locomotion`, `gripper` | pi0.5 fine-tunes action groups independently; composition requires knowing which group a dataset covers |
| **Normalization** | quantile (q01/q99), z-score (mean/std), identity, baked-into-checkpoint | Mismatch causes scale errors; strategy split across dataset stats and policy config |
| **Absolute vs. delta** | absolute position target, delta from current, delta from chunk start | Same shape, completely different control behavior |
| **Gripper convention** | binary open/close, continuous 0-1, force target | Varies across datasets with no declaration |

### What exists today (partial solutions)

**OXE 7-DOF normalization**: Converts all datasets to a common 7-dim EE delta representation, zero-padded to 32 dims. The only cross-embodiment action "standard" to date.

- Limitation: Highly opinionated — forces everything into EE deltas. Cannot represent joint-space control, humanoid locomotion, dexterous hands, or any action space beyond 7 DOF. Not a descriptive vocabulary but a conversion pipeline.
- Status: Momentum has shifted from RLDS/OXE to LeRobot. New projects do not standardize on OXE.
- Source: [OXE paper](https://arxiv.org/abs/2310.08864), [RT-X paper](https://arxiv.org/abs/2310.08864)

**GR00T `modality.json`**: Maps index ranges in flat arrays to named semantic fields (e.g., `"joint_position": {"start":0, "end":6}`).

- Limitation: NVIDIA-specific, not a community standard. Splits arrays but doesn't declare representation type, units, or coordinate frame.
- Source: [GR00T data preparation](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/data_preparation.md)

**LeRobot `info.json`**: Feature names, shapes, dtypes. No action semantics. A feature named `action` with shape `[7]` carries no semantic information.

- Source: [LeRobot v3 docs](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)

**DROID multi-representation recording**: Records joint velocity, Cartesian delta, and Cartesian velocity simultaneously. Defers the choice to training time.

- Limitation: Multiple representations exist in HDF5 structure but with no standardized metadata declaring which is which.
- Source: [DROID paper](https://arxiv.org/abs/2403.12945)

**Research proposals** (not adopted standards):

- CalibAll ([arXiv:2511.17001](https://arxiv.org/abs/2511.17001)): unifies actions in camera frame using extrinsics
- UniAct ([arXiv:2501.10105](https://arxiv.org/abs/2501.10105)): learned 256-entry vector-quantized universal action codebook
- FAST/FAST+ (used by pi0/pi0.5): universal pretrained action tokenizer — learned representation, not declarative vocabulary
- Latent Action Diffusion ([arXiv:2506.14608](https://arxiv.org/abs/2506.14608)): learned latent action space

### Key insight: describe, don't convert

OXE's approach (convert everything to one canonical format) is fragile: it cannot accommodate new action types (dexterous hands, whole-body humanoids, multi-agent) without redesigning the canonical format. The alternative is to **describe** what each dataset/checkpoint uses, in a vocabulary that's extensible, and let the Embodiment Adapter handle conversion. This is the difference between a rigid schema and a descriptive metadata standard.

---

## 3. Observation Semantics

### What needs to be described

The same semantic gap exists on the observation (input) side. A policy requires specific observation modalities; a dataset provides specific recordings. Matching them requires knowing more than feature names and shapes.

**Camera observations:**

| Property | Example values | Why it matters |
| ---------- | --------------- | ---------------- |
| **Semantic role** | world/external, wrist/gripper, head, shoulder | Policy expects specific viewpoints; free-form names prevent matching |
| **Mono vs. stereo** | mono, stereo (left/right pair) | Stereo policy can't use mono data without architecture change |
| **Depth** | RGB only, RGB-D, depth-only | Depth availability changes what the policy can learn |
| **Resolution** | 224x224, 640x480, 1280x720 | Can be resized, but aspect ratio changes may lose information |
| **Color encoding** | RGB, BGR, grayscale, YUV | OpenCV defaults to BGR; many ML pipelines assume RGB |
| **Frame rate** | 5 Hz, 15 Hz, 30 Hz | Affects action chunking, temporal consistency |
| **Mounting / extrinsics** | Fixed position relative to base, attached to EE, head-mounted | Determines what the camera sees as the robot moves |
| **Intrinsics** | FOV, focal length, distortion model | Needed for 3D reconstruction, sim-to-real transfer |

**Proprioception observations:**

| Property | Example values | Why it matters |
| ---------- | --------------- | ---------------- |
| **Joint ordering** | Which joint is dim 0? URDF order ≠ USD order ≠ MJCF order | Silent permutation error |
| **Joint types** | revolute, prismatic, continuous, fixed | Affects value interpretation |
| **State representation** | position, velocity, effort, or combination | A `float[14]` could be 7 positions + 7 velocities, or 14 positions |
| **Units** | radians, degrees, meters, N·m | No declaration in any format |
| **Coordinate frame** | base_link, world | Determines reference for joint positions |

**Other sensor modalities:**

| Modality | Key properties to describe |
| ---------- | --------------------------- |
| **IMU** | Axes convention (NED vs ENU), sample rate, accelerometer + gyroscope ranges |
| **Force/torque** | Sensor location (wrist, fingertip), frame, units (N, N·m) |
| **LiDAR** | Point density, range, scan pattern, coordinate frame |
| **Tactile** | Sensor type (GelSight, BioTac), spatial resolution, data format |

### What exists today (partial solutions)

**LeRobot `info.json`**: Has `video_info` with fps, codec, pixel channels, height/width. No semantic role, no mounting info, no camera intrinsics/extrinsics.

**LeRobot `rename_map`**: Acknowledges the camera name mapping problem but solves it by manual user specification at training time. Not standardized role names in metadata.

**ROS 2 `sensor_msgs/CameraInfo`**: Has intrinsics (K matrix, distortion model) and `frame_id` for mounting. But this is runtime message data, not dataset metadata. Lost during recording-to-dataset conversion.

**URDF `<sensor>` elements**: Can describe camera mounting position and some optical parameters. But URDF sensor support is incomplete and inconsistent across parsers.

**GR00T `modality.json`**: Has `video_key` remapping but no camera semantics beyond name mapping.

### Key insight: camera extrinsics/intrinsics are lost during dataset conversion

Camera mounting information (parent link, relative position/rotation in the parent's coordinate frame, focal length, FOV, distortion model) exists at recording time — in the URDF, the ROS TF tree, or the simulator scene description. This information is **discarded** during conversion to training datasets. The result: an engineer trying to reproduce a dataset's camera setup on their own robot (e.g., mounting a wrist camera on a Franka Panda gripper) must reverse-engineer the extrinsics experimentally — trial and error to avoid obstructed views and out-of-distribution frames.

This is worse than a naming problem. Even if camera roles were standardized, without extrinsics metadata you cannot:

- Reproduce the camera setup on a different physical robot
- Judge whether your camera placement is close enough to the training data's
- Automatically configure a simulator to match the real setup (or vice versa for sim-to-real)

### Camera role naming is a secondary (but high-frequency) pain point

From our survey and LeRobot issue tracker: camera name mismatches between datasets and models are the most common integration error. A dataset with `observation.images.top` and a model expecting `observation.images.exterior_image_1_left` — the same physical camera, different names, no way to automatically match them.

---

## 4. The Relationship Between Action and Observation Semantics

Action and observation semantics are not independent:

1. **Training determines both**: A dataset recorded with a specific camera setup and action convention produces a checkpoint that expects the same observation format and outputs the same action format.
2. **The checkpoint inherits both**: A fine-tuned model implicitly requires the observation modalities it was trained on and outputs actions in the format of its training data.
3. **The Embodiment Adapter bridges both**: It must map robot sensor streams → policy observation format (input side) AND policy action output → robot joint commands (output side).
4. **Compatibility is bidirectional**: Selecting a dataset for fine-tuning requires matching BOTH observation modalities (does the dataset have the cameras the policy needs?) AND action semantics (does the dataset's action format match the base model's expectation, or can it be converted?).

### Artifacts that need semantic metadata

| Artifact | What it declares |
| ---------- | ----------------- |
| **Dataset** | "I contain episodes with these observation modalities [camera roles, proprioception format] and these action groups [representation type, units, dims]" |
| **Checkpoint** | "I expect these observation inputs [inherited from training data] and produce these action outputs [inherited from training data]" |
| **Embodiment Adapter** | "I convert between [checkpoint's action/observation format] and [robot's physical interfaces]" |
| **Robot Description** | "I have these joints [names, types, limits], these sensors [cameras, IMU, F/T], these control interfaces [position, velocity, effort]" |

---

## 5. What This Is NOT

This document describes the **problem and the gap**. It does not propose:

- A specific schema or vocabulary (that belongs in the design phase)
- An extension to LeRobot's `info.json` (that may be one implementation, but we haven't validated it)
- A new spec (existing communities may address this; we need to check)

### Open questions for further research

1. Is the LeRobot community receptive to adding semantic metadata to `info.json`? (Check issues, PRs, maintainer discussions)
2. Is anyone in the ROS / Gazebo community working on observation metadata standards for ML datasets?
3. Does Gymnasium's `observation_space` / `action_space` API (Box, Dict, etc.) provide a usable vocabulary, or is it too coarse?
4. Could Rosetta's YAML contract be extended to cover action/observation semantics, not just feature-to-topic mapping?
5. Are the research proposals (CalibAll, UniAct) converging on a shared vocabulary even if their methods differ?
