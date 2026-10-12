# Observation Descriptor Survey

**Date**: 2026-09-21
**Scope**: How different Physical AI communities represent observation semantics — camera descriptions, proprioception, and composite observation spaces.

**Descriptor model under investigation**:

- **ObservationDescriptor**: full observation space (cameras[], proprioception, language, additional modalities)
- **CameraDescriptor**: single camera (role, resolution, color encoding, stereo, extrinsics, intrinsics, preprocessing state)
- **ProprioceptionDescriptor**: proprioceptive state vector (joint positions, velocities, efforts, gripper — per-dimension semantics)

---

## 1. OpenPI

**Source**: [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) (`src/openpi/models/model.py`, `examples/droid/`, `examples/aloha/`, `examples/libero/`, `src/openpi/transforms.py`)

### What they call the concept

No unified observation descriptor. The model-level `Observation` dataclass in `model.py` defines the canonical internal structure. Per-robot observation conventions are encoded in example scripts and transform pipelines.

### Model-level Observation dataclass

```python
@dataclass
class Observation:
    images: dict[str, Float[*b h w c]]      # named camera images
    image_masks: dict[str, Bool[*b]]         # per-camera validity masks
    state: Float[*b s]                       # proprioceptive state vector
    tokenized_prompt: Int[*b l] | None       # tokenized language instruction
    tokenized_prompt_mask: Bool[*b l] | None
    token_ar_mask, token_loss_mask           # FAST-specific
```

Constants: `IMAGE_RESOLUTION = (224, 224)`, `IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")`.

### Camera naming conventions

The model defines abstract camera slots (`base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb`). Each robot maps its physical cameras to these slots via `RepackTransform` (regex-based key remapping):

| Robot | Physical camera keys | Mapped to |
| ------- | --------------------- | ----------- |
| DROID | `exterior_image_1_left`, `wrist_image_left` | `base_0_rgb`, `left_wrist_0_rgb` |
| ALOHA real | `cam_high`, `cam_low`, `cam_left_wrist`, `cam_right_wrist` | `base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb` |
| ALOHA sim | `pixels["top"]` | `base_0_rgb` |
| LIBERO | `agentview_image`, `robot0_eye_in_hand_image` (180deg rotated) | `base_0_rgb`, `left_wrist_0_rgb` |

### Proprioception

Structure varies entirely by robot:

| Robot | State composition | Dimension |
| ------- | ------------------ | ----------- |
| DROID | `joint_position` (7) + `gripper_position` (1) | 8 |
| ALOHA | `qpos` (14: 7 joints x 2 arms) | 14 |
| LIBERO | `robot0_eef_pos` (3) + axisangle(quat) (3) + `robot0_gripper_qpos` (1) | 7 |

### Explicitly declared vs implicit

| Property | Status |
| ---------- | -------- |
| Camera resolution | Implicit — all resized to 224x224 by `ResizeImages` transform |
| Camera role | Implicit — encoded in key names, mapped by `RepackTransform` |
| Color encoding | Implicit — assumed RGB float32 in [-1, 1] after preprocessing |
| Intrinsics/extrinsics | Not represented |
| State semantics | Implicit — per-dimension meaning hardcoded per robot |
| Normalization | Explicit — `norm_stats.json` with z-score or quantile stats |

### Key gaps relative to our model

- No CameraDescriptor: camera role, resolution, intrinsics, extrinsics, stereo, preprocessing state are all implicit
- No ProprioceptionDescriptor: state vector semantics are per-robot convention
- Camera slot mapping (`RepackTransform`) is exactly the function our MappingDescriptor would formalize
- `image_masks` (per-camera validity) is a concept not in our current CameraDescriptor

---

## 2. LeRobot

**Source**: [huggingface/lerobot](https://github.com/huggingface/lerobot) (`lerobot/common/datasets/dataset_metadata.py`, `lerobot/common/datasets/feature_utils.py`); example: [lerobot/aloha_sim_insertion_human info.json](https://huggingface.co/datasets/lerobot/aloha_sim_insertion_human/blob/main/meta/info.json)

### What they call the concept

Observations are declared as **features** in `meta/info.json`, keyed by dot-separated names following the convention `observation.<modality>.<name>`.

### Feature declaration in info.json

```json
{
  "features": {
    "observation.images.top": {
      "dtype": "video",
      "shape": [480, 640, 3],
      "names": ["height", "width", "channel"],
      "info": {
        "video.fps": 50.0,
        "video.codec": "av1",
        "video.pix_fmt": "yuv420p",
        "video.is_depth_map": false,
        "has_audio": false
      }
    },
    "observation.images.left_wrist": {
      "dtype": "video",
      "shape": [480, 640, 3],
      "names": ["height", "width", "channel"]
    },
    "observation.state": {
      "dtype": "float32",
      "shape": [14],
      "names": {
        "motors": ["left_waist", "left_shoulder", "left_elbow",
                    "left_forearm_roll", "left_wrist_angle",
                    "left_wrist_rotate", "left_gripper", ...]
      }
    },
    "action": {
      "dtype": "float32",
      "shape": [14],
      "names": { "motors": ["left_waist", ...] }
    }
  }
}
```

### Camera metadata coverage

| Property | Coverage |
| ---------- | ---------- |
| Resolution | Yes — from `shape` ([H, W, C]) |
| FPS | Yes — `video.fps` in `info` |
| Codec | Yes — `video.codec` (av1, h264) |
| Pixel format | Partial — `video.pix_fmt` is codec-level (yuv420p), not tensor color space |
| Depth map flag | Yes — `video.is_depth_map` boolean |
| Camera role | **No** — implicit in key name (`observation.images.top`, `observation.images.left_wrist`) |
| Camera intrinsics | **No** |
| Camera extrinsics | **No** |
| Stereo | **No** |
| Preprocessing state | **No** |

### Proprioception

`observation.state` provides per-dimension naming via `names: {"motors": [...]}`. Example: `["left_waist", "left_shoulder", ..., "right_gripper"]`. No units, ranges, coordinate frames, or representation type (joint_position vs ee_delta).

### Key gaps relative to our model

- Camera roles are convention-encoded in key names, not explicit metadata
- No camera calibration data (intrinsics, extrinsics, distortion)
- No preprocessing state tracking (resize, normalize, crop)
- `pix_fmt` describes the video codec format, not the decoded tensor color space
- Proprioception has named dimensions but no units, ranges, or representation type
- No composite observation descriptor — each feature is independent, no grouping concept

---

## 3. ROS 2

**Source**: [ros2/common_interfaces](https://github.com/ros2/common_interfaces) — `sensor_msgs/msg/CameraInfo.msg`, `sensor_msgs/msg/Image.msg`, `sensor_msgs/msg/JointState.msg`, `sensor_msgs/msg/Imu.msg`, `sensor_msgs/msg/PointCloud2.msg`

### What they call the concept

Individual message types for each sensor modality. No composite "observation" concept — composition happens at the application level via topic subscription.

### CameraInfo (intrinsics + calibration)

```
Header header              # timestamp + frame_id
uint32 height, width       # image dimensions
string distortion_model    # "plumb_bob", "rational_polynomial", etc.
float64[] D                # distortion coefficients
float64[9] K               # 3x3 intrinsic matrix (fx, fy, cx, cy)
float64[9] R               # 3x3 rectification matrix (stereo)
float64[12] P              # 3x4 projection matrix (includes stereo baseline)
uint32 binning_x, binning_y
RegionOfInterest roi
```

The most complete camera calibration format among all communities surveyed. Uncalibrated cameras detected via `K[0] == 0.0`.

### Image

```
Header header
uint32 height, width
string encoding            # "rgb8", "bgr8", "mono8", "32FC1", etc.
uint8 is_bigendian
uint32 step                # row stride in bytes
uint8[] data
```

### JointState (proprioception)

```
Header header
string[] name              # per-joint names
float64[] position         # rad (revolute) or m (prismatic)
float64[] velocity         # rad/s or m/s
float64[] effort           # Nm or N
```

All arrays same length or empty; any can be omitted. Units are implicit per joint type.

### Extrinsics via TF tree

Every message carries `header.frame_id` naming its coordinate frame (e.g., `camera_optical_frame`, `base_link`). The TF tree (published separately as transforms) provides spatial relationships between all frames, giving camera extrinsics without embedding them in the message. This is the most complete extrinsics system of any community surveyed.

### Explicitly declared vs implicit

| Property | Status |
| ---------- | -------- |
| Camera intrinsics | **Explicit** — full K matrix, distortion model + coefficients |
| Camera extrinsics | **Explicit** — via frame_id + TF tree |
| Image encoding | **Explicit** — encoding string (rgb8, bgr8, mono8, depth) |
| Resolution | **Explicit** — height, width |
| Stereo | **Explicit** — R matrix, P matrix with baseline |
| Joint names | **Explicit** — string array |
| Joint units | **Implicit** — convention based on joint type |
| Camera role | **Implicit** — frame_id names the frame but doesn't declare "wrist" vs "world" |
| Preprocessing state | **Not represented** |
| Observation composition | **Not represented** — no grouping of sensors into observation space |

### What ROS provides that ML frameworks lack

- Full camera calibration (intrinsic matrix, distortion model, stereo rectification)
- Coordinate frame system (frame_id + TF tree for spatial relationships)
- Per-joint naming with explicit name array
- Uncertainty (IMU covariance matrices)
- Self-describing point clouds (PointField channel descriptors)

### What ROS lacks for ML observation descriptors

- No preprocessing/normalization state
- No camera role semantics beyond frame naming
- No observation space composition (grouping cameras + state + language)
- No ML tensor metadata (CHW vs HWC, batch dimensions, dtype conventions)

---

## 4. OpenUSD

**Source**: [OpenUSD API — UsdGeomCamera](https://openusd.org/docs/api/class_usd_geom_camera.html), UsdRenderProduct

### What they call the concept

Camera properties live on `UsdGeomCamera` prims. Resolution and output format live on `UsdRenderProduct` (bound to a camera).

### UsdGeomCamera properties

**Intrinsics:**

- `projection` — `perspective` or `orthographic`
- `focalLength` — in tenths of scene unit (default 50, i.e. 50mm at cm stage)
- `horizontalAperture` / `verticalAperture` — filmback size (default 20.955 = 35mm spherical)
- `horizontalApertureOffset` / `verticalApertureOffset` — lens shift
- FOV derived: `hFOV = 2 * atan(horizontalAperture / (2 * focalLength))`
- `clippingRange` — near/far planes

**Depth of field:** `fStop`, `focusDistance`

**Exposure:** `exposure`, `exposure:iso`, `exposure:time` — mirrors real camera parameters

**Stereo:** `stereoRole` — `mono`, `left`, `right`

**Extrinsics:** Inherited from `UsdGeomXformable` — full transform op stack. Cameras look down -Z, +X right, Y-up. `ComputeLocalToWorldTransform(time)` gives world-space pose.

### Resolution and output via UsdRenderProduct

`UsdGeomCamera` itself has **no resolution or pixel format**. These live on `UsdRenderProduct`:

- Resolution — `GfVec2i` (width, height)
- Pixel aspect ratio
- Ordered RenderVars (color, depth, normals, etc.)
- Product type — `raster` or `deepRaster`

### Coverage vs CameraDescriptor

| CameraDescriptor field | OpenUSD coverage |
| ------------------------ | ------------------ |
| role (wrist/world/head) | **Not defined** — no semantic role concept |
| resolution | On UsdRenderProduct, not camera prim |
| color_encoding | **Not defined** — deferred to renderer schemas |
| stereo | `stereoRole` (mono/left/right) |
| extrinsics | Full transform hierarchy via UsdGeomXformable |
| intrinsics (focal, aperture) | Complete for physical optics (no distortion model) |
| distortion model | **Not defined** — USD assumes ideal lenses |
| preprocessing_state | **Not applicable** — USD describes scene, not ML pipeline |

### Key gaps for ML use cases

- No distortion model (barrel/pincushion) — real cameras always have distortion
- No pixel format / color encoding on the camera itself
- No semantic camera role
- No preprocessing state concept
- Resolution decoupled from camera — requires navigating to RenderProduct binding

---

## 5. URDF / SDF / Gazebo

**Source**: [URDF sensor spec](http://wiki.ros.org/urdf/XML/sensor), [SDF camera spec](http://sdformat.org/spec?ver=1.11&elem=sensor#camera_camera), [Gazebo camera plugin](https://classic.gazebosim.org/tutorials?tut=ros_gzplugins#Camera)

### What they call the concept

URDF itself has a minimal `<sensor>` element (REP-0120 proposal, limited adoption). In practice, camera sensors are declared via **Gazebo plugin extensions** inside `<gazebo>` blocks in the URDF, or natively in **SDF** (Simulation Description Format) which has first-class sensor support.

### URDF sensor element (REP-0120)

```xml
<sensor name="head_camera" type="camera" update_rate="30">
  <parent link="head_link"/>
  <origin xyz="0.05 0 0.1" rpy="0 0 0"/>
  <camera>
    <image width="640" height="480" format="R8G8B8" hfov="1.3962634"/>
    <clip near="0.02" far="300"/>
  </camera>
</sensor>
```

Properties: `name`, `type` (camera, ray, imu, etc.), `update_rate`, parent link, 6-DOF origin pose. Camera sub-element: `image` (width, height, format, hfov), `clip` (near, far).

**Adoption note**: REP-0120 was proposed but never widely adopted. Most URDF files use the Gazebo extension approach instead.

### SDF camera sensor (native)

SDF provides richer camera description:

```xml
<sensor name="camera" type="camera">
  <update_rate>30</update_rate>
  <camera>
    <horizontal_fov>1.047</horizontal_fov>
    <image>
      <width>640</width>
      <height>480</height>
      <format>R8G8B8</format>
    </image>
    <clip>
      <near>0.1</near>
      <far>100</far>
    </clip>
    <noise>
      <type>gaussian</type>
      <mean>0.0</mean>
      <stddev>0.007</stddev>
    </noise>
    <distortion>
      <k1>0.0</k1><k2>0.0</k2><k3>0.0</k3>
      <p1>0.0</p1><p2>0.0</p2>
      <center>0.5 0.5</center>
    </distortion>
    <lens>
      <type>stereographic</type>
      <scale_to_hfov>true</scale_to_hfov>
      <intrinsics>
        <fx>277.1</fx><fy>277.1</fy>
        <cx>160</cx><cy>120</cy>
        <s>0</s>
      </intrinsics>
    </lens>
  </camera>
</sensor>
```

SDF adds: noise model (gaussian/custom), distortion coefficients (k1-k3, p1-p2, center), lens type (gnomonical, stereographic, equidistant, etc.), and explicit intrinsics (fx, fy, cx, cy, skew). Depth cameras use `type="depth"` with the same structure.

### Gazebo plugin extension in URDF

```xml
<gazebo reference="camera_link">
  <sensor type="camera" name="head_camera">
    <update_rate>30.0</update_rate>
    <camera name="head">
      <horizontal_fov>1.3962634</horizontal_fov>
      <image>
        <width>800</width>
        <height>600</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.02</near>
        <far>300</far>
      </clip>
      <noise>
        <type>gaussian</type>
        <mean>0.0</mean>
        <stddev>0.007</stddev>
      </noise>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
      <alwaysOn>true</alwaysOn>
      <updateRate>0.0</updateRate>
      <cameraName>head_camera</cameraName>
      <imageTopicName>image_raw</imageTopicName>
      <cameraInfoTopicName>camera_info</cameraInfoTopicName>
      <frameName>camera_optical_frame</frameName>
    </plugin>
  </sensor>
</gazebo>
```

The plugin bridges simulation to ROS topics, publishing `sensor_msgs/Image` and `sensor_msgs/CameraInfo`.

### Explicitly declared vs implicit

| Property | Status |
| ---------- | -------- |
| Resolution | **Explicit** -- `<image>` width, height |
| Pixel format | **Explicit** -- `<format>` (R8G8B8, L8, B8G8R8, etc.) |
| FOV | **Explicit** -- `<horizontal_fov>` in radians |
| Clip planes | **Explicit** -- `<clip>` near, far |
| Update rate | **Explicit** -- `<update_rate>` in Hz |
| Noise model | **Explicit** (SDF/Gazebo) -- type, mean, stddev |
| Distortion | **Explicit** (SDF 1.6+) -- k1-k3, p1-p2, center |
| Intrinsics | **Explicit** (SDF 1.6+) -- fx, fy, cx, cy, skew via `<lens><intrinsics>` |
| Extrinsics | **Explicit** -- parent link + `<origin>` pose (URDF) or `<pose>` (SDF) |
| Camera role | **Not represented** -- no semantic role concept; only link name convention |
| Stereo | **Partial** -- modeled as two separate camera sensors with known baseline |
| Preprocessing state | **Not represented** -- describes physical sensor, not ML pipeline |
| Depth sensing | **Explicit** -- separate sensor `type="depth"` |

### What URDF/SDF provides that ML frameworks lack

- Noise model for simulation fidelity (gaussian noise on pixel values)
- Distortion coefficients (SDF) matching the Brown-Conrady model
- Explicit lens type (stereographic, equidistant, equisolid for fisheye)
- Physical mounting via parent link and 6-DOF pose offset
- Update rate for temporal characteristics

### What URDF/SDF lacks for ML observation descriptors

- No camera role semantics (wrist, world, head, ego)
- No preprocessing/normalization state
- No ML tensor format metadata (HWC vs CHW, dtype, batch)
- No observation composition concept
- URDF's native sensor support is minimal; real capability requires SDF or Gazebo plugins

---

## 6. Gymnasium (Farama)

**Source**: [Farama-Foundation/Gymnasium](https://github.com/Farama-Foundation/Gymnasium) — `gymnasium/spaces/dict.py`, `gymnasium/spaces/box.py`, `gymnasium/core.py`

### What they call the concept

**`observation_space`** — a `Space` object (typically `Dict`, `Box`, `MultiBinary`, or `Discrete`) that defines the structure and bounds of valid observations. GoalEnv adds structured keys.

### Observation space types

**Box**: continuous space with bounds.

- Properties: `low` (ndarray), `high` (ndarray), `shape` (tuple), `dtype` (numpy dtype)
- No dimension names, no units, no frame info, no semantic labels

**Dict**: composite space mapping string keys to sub-spaces.

- Supports nesting (Dict of Dict)
- Keys provide naming but no formal ontology
- Used for goal-conditioned envs: `{"observation": Box(...), "desired_goal": Box(...), "achieved_goal": Box(...)}`

### GoalEnv observation structure

Goal-conditioned environments use a Dict observation space with three standard keys:

- `observation` — current state observation
- `desired_goal` — target state to achieve
- `achieved_goal` — current state projected into goal space

This is the closest Gymnasium comes to structured observation semantics — but only for goal-conditioned tasks.

### Explicitly declared vs implicit

| Property | Status |
| ---------- | -------- |
| Shape | **Explicit** — on every Space |
| Dtype | **Explicit** — on every Space |
| Bounds (low/high) | **Explicit** — on Box |
| Dimension names | **Not represented** |
| Units | **Not represented** |
| Coordinate frames | **Not represented** |
| Modality labels | **Not represented** |
| Temporal context (frequency) | **Not represented** |
| Semantic grouping | Partial — Dict keys provide names, GoalEnv provides structure |

### Key gaps relative to our model

- Spaces define shape/dtype/bounds but never what each dimension means semantically
- No camera descriptor concept — images are just Box spaces with shape [H, W, C]
- No proprioception semantics — joint states are flat arrays with positional indexing
- `render_mode` ("rgb_array") controls visual output but is not connected to observation_space
- The `metadata` dict can hold arbitrary info but has no standard observation keys

---

## 7. GR00T

**Source**: [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) — `gr00t/data/types.py`, `gr00t/configs/data/embodiment_configs.py`, `gr00t/data/interfaces.py`, `gr00t/data/state_action/state_action_processor.py`

### What they call the concept

**`ModalityConfig`** — per-modality configuration specifying which data keys to load and how to sample them. Organized by **embodiment tag** (e.g., `oxe_droid_relative_eef_relative_joint`, `unitree_g1_sonic`).

**`VLAStepData`** — the core observation dataclass:

```python
@dataclass
class VLAStepData:
    images: dict[str, list[np.ndarray]]    # view_name -> temporal stack
    states: dict[str, np.ndarray]          # state_name -> array
    actions: dict[str, np.ndarray]         # action_name -> array
    masks: dict[str, list[np.ndarray]] | None  # view_name -> masks
    text: str | None                       # task description
    embodiment: EmbodimentTag              # cross-embodiment tag
    metadata: dict[str, Any]               # extensible
```

### ModalityConfig structure

```python
@dataclass
class ModalityConfig:
    delta_indices: list[int]           # temporal sampling offsets
    modality_keys: list[str]           # data keys to load
    sin_cos_embedding_keys: list[str] | None   # keys for sin/cos normalization
    mean_std_embedding_keys: list[str] | None  # keys for mean/std normalization
    action_configs: list[ActionConfig] | None   # per-key action representation
```

### Camera conventions per embodiment

Organized in `MODALITY_CONFIGS` dict, keyed by embodiment tag value:

| Embodiment tag | Video keys | State keys |
| --------------- | ------------ | ------------ |
| `oxe_droid_relative_eef_relative_joint` | `exterior_image_1_left`, `wrist_image_left` | `eef_9d`, `gripper_position`, `joint_position` |
| `unitree_g1_sonic` | `ego_view` | `left_leg`, `right_leg`, `waist`, `left_arm`, `right_arm`, `left_hand`, `right_hand`, `projected_gravity` |
| `unitree_g1_full_body_*` | `ego_view` | `left_leg`, `right_leg`, `waist`, `left_arm`, `right_arm`, `left_hand`, `right_hand` |
| `libero_sim` | `image`, `wrist_image` | `x`, `y`, `z`, `roll`, `pitch`, `yaw`, `gripper` |
| `simpler_env_widowx` | `image_0` | `x`, `y`, `z`, `roll`, `pitch`, `yaw`, `pad`, `gripper` |
| `simpler_env_google` | `image` | `x`, `y`, `z`, `rx`, `ry`, `rz`, `rw`, `gripper` |

### Explicitly declared vs implicit

| Property | Status |
| ---------- | -------- |
| Camera keys | **Explicit** — in `modality_keys` per embodiment |
| Camera resolution | **Implicit** — handled by VLM backbone ("supports flexible resolution") |
| Camera role | **Implicit** — encoded in key name (`ego_view`, `wrist_image`) |
| Camera intrinsics | **Not represented** |
| State key names | **Explicit** — per-dimension keys (`x`, `y`, `z`, `roll`, ...) |
| State representation | **Explicit** — `ActionConfig` with `rep` (relative/delta/absolute), `type` (eef/non_eef), `format` (xyz+rot6d, etc.) |
| Normalization | **Explicit** — `sin_cos_embedding_keys`, `mean_std_embedding_keys`, statistics dict |
| Temporal sampling | **Explicit** — `delta_indices` (e.g., `[-15, 0]` for history) |
| Embodiment | **Explicit** — `EmbodimentTag` enum |

### Key gaps relative to our model

- No CameraDescriptor: video keys are strings with no resolution, intrinsics, or role metadata
- Per-dimension state keys (`x`, `y`, `z`, `roll`) are more granular than our ProprioceptionDescriptor
- `ActionConfig` with `rep`/`type`/`format` is close to our ActionDescriptor's `representation_type`
- Temporal sampling (`delta_indices`) is a concept not in our current model
- The embodiment tag acts as an indirection layer — our model would make this explicit per-descriptor

---

## 8. LEAPP / RDP (Robot Deployment Package)

**Source**: [say-paul/robotics-rl](https://github.com/say-paul/robotics-rl) (unified_launcher branch) — `configs/robots/g1_groot_wbc_unified.yaml`. LEAPP itself is not publicly documented; observations here come from the RDP (Robot Deployment Package) which uses LEAPP-style patterns.

### What they call the concept

The RDP YAML defines model inputs/outputs with explicit **tensor shapes, dtypes, and semantic names**. Sensor configuration is under `hardware.sensors`. There is no standalone observation descriptor — observation semantics are distributed across `hardware`, `models`, and `execution` sections.

### Model input declarations

```yaml
models:
  - name: "groot_vla_n16"
    input:
      - name: "rgb_image"
        shape: [1, 3, 480, 640]
        dtype: "float32"
      - name: "language_prompt"
        shape: [1, 512]
        dtype: "int32"
      - name: "robot_state"
        shape: [1, 43]      # joint pos/vel + base state
        dtype: "float32"
```

### Sensor (camera) configuration

```yaml
hardware:
  sensors:
    - name: "head_camera_rgb"
      type: "camera"
      modality: "rgb"
      topic: "/camera/rgb/image_raw"
      resolution: [640, 480]
      rate_hz: 30
    - name: "torso_imu"
      type: "imu"
      topic: "/imu/data"
      frame_id: "torso_link"
      rate_hz: 500
```

### Explicitly declared vs implicit

| Property | Status |
| ---------- | -------- |
| Camera resolution | **Explicit** — `resolution: [640, 480]` on sensor |
| Camera modality | **Explicit** — `modality: "rgb"` |
| Camera rate | **Explicit** — `rate_hz: 30` |
| Camera intrinsics | **Not represented** |
| Camera role | **Partial** — `name: "head_camera_rgb"` is naming convention, not formal role |
| IMU frame | **Explicit** — `frame_id: "torso_link"` |
| Model input shapes | **Explicit** — tensor shape + dtype per input |
| State semantics | **Implicit** — `robot_state` shape [1, 43] with no per-dimension labels |
| Preprocessing | **Implicit** — model expects [1, 3, 480, 640] but sensor provides [640, 480]; adaptation unstated |

### Key gaps relative to our model

- Sensor config and model input config are separate with no explicit bridge
- No per-dimension state semantics (43-dim vector is opaque)
- Camera type/modality declared but no intrinsics, extrinsics, or distortion
- The shape mismatch between sensor output (HWC) and model input (NCHW) has no explicit transform descriptor
- Joint ordering defined by URDF reference, not inline

---

## 9. vLLM / SGLang

**Source**: [sgl-project/sglang](https://github.com/sgl-project/sglang) — `python/sglang/multimodal_gen/runtime/vla/observation.py`, `python/sglang/multimodal_gen/configs/pipeline_configs/pi05.py`, `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/pi05_preprocess.py`; [vllm-project/vllm](https://github.com/vllm-project/vllm) — `vllm/model_executor/models/openvla.py`

### What they call the concept

**`VLAObservationBatch`** (SGLang) — the observation container for VLA inference. **`Pi05PipelineConfig`** — the pipeline config declaring image keys, resolution, and normalization.

### SGLang VLAObservationBatch

```python
@dataclass
class VLAObservationBatch:
    prompt: list[str]
    images: dict[str, torch.Tensor]       # camera_name -> tensor
    image_masks: dict[str, torch.Tensor]  # camera_name -> validity
    state: torch.Tensor | None
    noise: torch.Tensor | None            # diffusion noise
    tokens: torch.Tensor
    token_masks: torch.Tensor
    batch_size: int
    metadata: dict[str, Any]              # includes "camera_order"
```

### SGLang Pi05PipelineConfig

```python
image_keys: tuple[str, ...] = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
empty_cameras: int = 0
image_size: tuple[int, int] = (224, 224)
image_normalization_mean: tuple[float, ...] = (0.5, 0.5, 0.5)
image_normalization_std: tuple[float, ...] = (0.5, 0.5, 0.5)
action_dim: int = 32
state_dim: int = 32
max_token_len: int = 200
action_horizon: int = 50
tokenizer_name: str = "google/paligemma-3b-pt-224"
```

### Image preprocessing pipeline (SGLang pi05_preprocess.py)

1. Accept torch.Tensor (CHW or HWC), PIL.Image, or numpy array
2. Auto-detect channel ordering and convert to CHW float32
3. Scale byte images to [0, 1], detect if already normalized to [-1, 1]
4. `resize_with_pad` to target resolution (224x224), preserving aspect ratio
5. Map to [-1, 1] range: `tensor * 2.0 - 1.0`
6. Proprioception: discretize state into 256 bins over [-1, 1], embed as text tokens: `"Task: {prompt}, State: {discretized_state};\nAction: "`

### vLLM OpenVLA

```python
_OPENVLA_IMAGE_SIZE = 224
_OPENVLA_PATCH_SIZE = 14
```

Single image per request, 6 channels (dual-backbone ViT), 224x224 resolution hardcoded. Image preprocessing delegated to HuggingFace processor.

### Explicitly declared vs implicit

| Property | Status |
| ---------- | -------- |
| Camera keys | **Explicit** — `image_keys` tuple in config |
| Image resolution | **Explicit** — `image_size = (224, 224)` in config |
| Normalization | **Explicit** — `image_normalization_mean/std` in config |
| Camera role | **Implicit** — encoded in key names (`base_0_rgb`, `left_wrist_0_rgb`) |
| Camera intrinsics | **Not represented** |
| Color encoding | **Implicit** — assumed RGB, auto-detected in preprocessor |
| State encoding | **Explicit** — discretized into 256 bins, embedded as text |
| Number of cameras | **Implicit** — derived from `len(image_keys) - empty_cameras` |

### Key gaps relative to our model

- Camera keys follow OpenPI convention but resolution/normalization are server config, not per-camera
- No per-camera resolution — all cameras share `image_size`
- No intrinsics, extrinsics, or distortion
- Preprocessing pipeline is auto-detecting (byte vs float, CHW vs HWC) rather than declared
- State dimensionality (`state_dim = 32`) is a flat number with no per-dimension semantics
- The `camera_order` metadata in `VLAObservationBatch` shows ordering matters but isn't formalized

---

## Cross-Community Comparison

### Camera descriptor coverage

| Property | OpenPI | LeRobot | ROS 2 | OpenUSD | Gymnasium | GR00T | URDF/Gazebo | RDP | SGLang |
| ---------- | -------- | --------- | ------- | --------- | ----------- | ------- | ------------- | ----- | -------- |
| Resolution | implicit | shape | explicit | RenderProduct | shape | implicit | **explicit** | explicit | config |
| Color encoding | implicit | pix_fmt (codec) | encoding | not defined | not defined | implicit | **format** | modality | implicit |
| Role (wrist/world) | key convention | key convention | frame_id | not defined | not defined | key convention | name attr | name convention | key convention |
| Intrinsics | -- | -- | **full K matrix** | focal+aperture | -- | -- | derived from FOV | -- | -- |
| Distortion | -- | -- | **full model** | -- | -- | -- | **k1-k3, p1-p2** | -- | -- |
| Extrinsics | -- | -- | **frame_id + TF** | **transform stack** | -- | -- | **origin xform** | frame_id | -- |
| Stereo | -- | -- | **R, P matrices** | **stereoRole** | -- | -- | two sensors | -- | -- |
| Preprocessing | -- | -- | -- | -- | -- | -- | -- | -- | mean/std |
| FPS/rate | -- | video.fps | -- | -- | -- | -- | **update_rate** | rate_hz | -- |
| Noise model | -- | -- | -- | -- | -- | -- | **gaussian** | -- | -- |
| Clipping planes | -- | -- | -- | **clippingRange** | -- | -- | **near/far** | -- | -- |

### Proprioception coverage

| Property | OpenPI | LeRobot | ROS 2 | Gymnasium | GR00T | RDP |
| ---------- | -------- | --------- | ------- | ----------- | ------- | ----- |
| Per-dimension names | -- | **names dict** | **name[]** | -- | **modality_keys** | -- |
| Units | -- | -- | implicit | -- | -- | -- |
| Representation type | per-robot | -- | -- | -- | **ActionConfig.rep** | -- |
| Velocity | per-robot | -- | **velocity[]** | -- | -- | -- |
| Effort/torque | -- | -- | **effort[]** | -- | -- | -- |
| Normalization | norm_stats | stats.json | -- | bounds | statistics dict | -- |

### Key findings

1. **Camera role is universally implicit.** Every community encodes camera role in key/frame names by convention. No community has an explicit `role` field with controlled vocabulary (wrist, world, head, ego).

2. **ROS 2 is the gold standard for camera calibration.** CameraInfo provides the most complete camera model (intrinsics, distortion, stereo, extrinsics via TF). No ML framework approaches this level of detail.

3. **OpenUSD complements ROS 2 for simulation.** USD covers physical optics (focal length, aperture, stereo) and transform hierarchy but lacks distortion models and pixel-level metadata.

4. **URDF/SDF bridges simulation and ROS calibration.** SDF (from version 1.6) includes distortion coefficients, lens intrinsics, and noise models that approach ROS 2 CameraInfo fidelity. Gazebo plugins publish this data as ROS CameraInfo messages, making SDF the only format that feeds directly into the ROS calibration pipeline.

5. **ML frameworks trade calibration for preprocessing metadata.** OpenPI, SGLang, and GR00T care about resolution, normalization range, and pixel format — concepts absent from ROS 2, OpenUSD, and URDF/SDF.

6. **Proprioception naming is divergent.** LeRobot uses `{"motors": [...]}`, GR00T uses per-key modality configs (`"left_arm"`, `"right_arm"`), ROS 2 uses `name[]` string array. No common schema.

7. **GR00T's ModalityConfig is the most structured.** It explicitly declares temporal sampling, normalization strategy per key, and action representation type — concepts other frameworks leave implicit.

8. **Observation composition is unsolved.** No community has a formal descriptor that groups cameras + proprioception + language into a single typed observation with compatibility validation. OpenPI's `Observation` dataclass is closest but it's a runtime type, not a metadata schema.
