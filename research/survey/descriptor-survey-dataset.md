# Descriptor Survey: Dataset & Training Metadata

**Scope**: How different Physical AI communities represent dataset-level metadata (what we call "DatasetDescriptor" and "EpisodeDescriptor") and training configuration (what we call "TrainingConfigDescriptor" and "DataConversionDescriptor").

**DatasetDescriptor** — describes a dataset's structure, features, embodiment, and provenance.

**EpisodeDescriptor** — describes per-episode metadata within a dataset (task, success, length, etc.).

**TrainingConfigDescriptor** — describes a training pipeline's data consumption, transforms, and produced checkpoint.

**DataConversionDescriptor** — describes how to convert between dataset formats or between a recording and a dataset.

---

## 1. LeRobot v3.0

**Source**: [HF docs](https://huggingface.co/docs/lerobot/lerobot-dataset-v3), [blog](https://huggingface.co/blog/lerobot-datasets-v3), [paper arXiv:2602.22818](https://arxiv.org/abs/2602.22818) (ICLR 2026)

### DatasetDescriptor equivalent

`meta/info.json` is the canonical metadata file. Known fields:

| Field | Type | Description |
| --- | --- | --- |
| `codebase_version` | string | LeRobot format version (e.g., "v3.0") |
| `robot_type` | string | Free-form robot name (e.g., "so101_follower") |
| `fps` | int | Recording frame rate |
| `features` | dict | Feature name → {dtype, shape, names} |
| `total_episodes` | int | Number of episodes |
| `total_frames` | int | Total timesteps across episodes |
| `splits` | dict | Train/val/test split specification |
| `data_path` | string | Path template for parquet shards |
| `video_path` | string | Path template for video shards |

Feature naming uses dot-notation: `observation.state`, `observation.images.<camera_key>`, `action`. Each feature has `dtype` (float32, int64, video), `shape` (e.g., `[7]`), `names` (dimension labels, e.g., joint names).

Additional metadata files:

| File | Content |
| --- | --- |
| `meta/stats.json` | Per-feature: mean, std, min, max (optionally q01, q99) |
| `meta/tasks.jsonl` | Natural-language task descriptions → integer IDs |
| `meta/episodes/` | Chunked Parquet: per-episode lengths, task assignments, data/video offsets |

### EpisodeDescriptor equivalent

Episode metadata in `meta/episodes/*.parquet`:

- `episode_index` — integer ID
- `length` — number of timesteps
- `task_index` — maps to `tasks.jsonl` entry
- Data/video byte offsets for streaming access

Per-timestep columns in `data/*.parquet` include `episode_index`, `frame_index`, `timestamp`, plus all feature columns.

### TrainingConfigDescriptor equivalent

Policy configuration (e.g., `Pi0Config`, `DiffusionConfig`, `ACTConfig`) in Python dataclasses. Key training-relevant fields:

- `normalization_mapping` — per-modality strategy (IDENTITY, MEAN_STD, QUANTILE)
- `input_features` / `output_features` — dict of feature name → {shape, type}
- `chunk_size` / `n_obs_steps` — temporal structure
- Image transforms configured at training time (brightness, contrast, sharpness augmentation)

### DataConversionDescriptor equivalent

Conversion scripts (`convert_dataset_v21_to_v30`, `push_dataset_to_hub`) with hardcoded logic. `rename_map` parameter allows camera key remapping at training time. No declarative conversion descriptor.

### What's implicit

- Action space semantics (joint position vs. EE delta vs. joint velocity)
- Camera roles (wrist vs. world vs. overhead)
- Embodiment structure (joint ordering, limits, kinematic chain)
- Normalization strategy designed for (only stats are stored, not strategy)
- Cross-artifact compatibility (which models this dataset works with)

### Community adoption

Dominant format as of 2026. 16K+ datasets, 2.2K+ contributors on HF Hub. Used by LeRobot, SmolVLA, XVLA. GR00T adopts as base format. `community_dataset_v3` aggregates 791 datasets across 46 robot types.

---

## 2. NVIDIA GR00T LeRobot Flavor

**Source**: [data preparation guide](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/data_preparation.md)

### DatasetDescriptor equivalent

Adds `meta/modality.json` on top of LeRobot v2 `info.json`:

| Field | Type | Description |
| --- | --- | --- |
| `state.joint_position` | {start, end} | Index range for joint positions in state array |
| `state.gripper_position` | {start, end} | Index range for gripper state |
| `action.joint_position` | {start, end} | Index range for joint positions in action array |
| `action.gripper_position` | {start, end} | Index range for gripper action |
| `video.<original_key>` | string | Remapped camera name (e.g., `"cam_high"` → `"video.ego"`) |
| `annotation.human.language_instruction` | string | Language instruction source field |

This supplements LeRobot's flat arrays with semantic field splitting — what GR00T calls "modality mapping."

Additional stats files:

- `stats.json` — standard stats (same format as LeRobot)
- `relative_stats.json` — per-horizon-step stats for relative action representations

### What GR00T adds that LeRobot lacks

- Index-based semantic array splitting (which indices are arm vs. gripper)
- Camera key remapping (original → standardized names)
- Per-group action representation metadata (`ActionRepresentation` enum: absolute/delta)
- Multiple annotation channel support

### What's still implicit

- Action semantics per dimension (unit, frame, sign convention)
- Camera roles (remapping is name→name, not name→role)
- Joint ordering rationale

---

## 3. RLDS / Open X-Embodiment (OXE)

**Source**: [RLDS GitHub](https://github.com/google-research/rlds), [OXE paper](https://arxiv.org/abs/2310.08864)

### DatasetDescriptor equivalent

Schema defined in Python code (TFDS `DatasetBuilder`), not a declarative metadata file.

Core data model:

- **Dataset** → collection of **Episodes**
- **Episode** → ordered sequence of **Steps**
- **Step** → `{observation, action, reward, discount, is_first, is_last, is_terminal}`

OXE adds conventions on top of RLDS:

- Observation dict: image fields, state vector, `language_instruction`
- Both `raw_action` (robot-specific) and normalized `action` (7-DOF EE) can coexist
- Actions normalized to standard 7-DOF EE representation across 60+ embodiments

OXE dataset registry (`rtx.py`) provides per-dataset metadata:

- `get_description()` — natural language summary
- `get_citation()` — BibTeX
- `get_homepage()` — project URL
- `get_relative_dataset_location()` — storage path + version

### EpisodeDescriptor equivalent

RLDS episodes have implicit structure through `is_first`/`is_last`/`is_terminal` step flags. No explicit per-episode metadata beyond these boundaries — no task label, success flag, or scene identifier at the episode level.

### Cross-embodiment metadata

**Implicit, not explicit.** The RTX dataset registry is a thin Python registry pointing to pre-converted datasets. No shared cross-embodiment metadata schema — no robot type enum, no action dimensionality spec, no sensor modality tags. Cross-embodiment alignment happens in the RLDS conversion process, not in metadata.

### What's implicit

- Robot type / embodiment (in documentation, not structured metadata)
- Action space semantics (OXE normalizes to 7-DOF but original semantics lost)
- Camera naming (varies per source dataset)
- Normalization (pipeline step, not metadata)

### Community adoption

Foundation format for cross-embodiment research (2023–2025). 60+ datasets, 22 robot platforms. However, momentum has shifted to LeRobot format as of 2025–2026.

---

## 4. Hugging Face Dataset Cards

**Source**: [HF docs](https://huggingface.co/docs/hub/datasets-cards)

### DatasetDescriptor equivalent

YAML frontmatter in README.md. Machine-readable fields:

| Field | Type | Description |
| --- | --- | --- |
| `license` | string | Data license (e.g., "cc-by-4.0") |
| `language` | list | Language codes |
| `tags` | list | Free-form tags |
| `task_categories` | list | HF task taxonomy (not robotics-specific) |
| `size_categories` | string | Order-of-magnitude size bin |
| `dataset_info` | object | Feature types, splits, download/dataset sizes |
| `configs` | list | Named subsets/configurations |
| `paperswithcode_id` | string | Link to Papers With Code |

### What it doesn't describe

- No robotics-specific fields (robot_type, action space, embodiment, camera roles)
- No compatibility metadata (which models or environments work with this dataset)
- Documentation-oriented: primarily for human discovery (search, filtering), not machine validation

### Community adoption

Universal on HF Hub — every dataset has a card. For robotics datasets, structured metadata is minimal: no fields for robot type, action space, observation space, or camera setup. In practice, most LeRobot dataset cards contain almost no free-text description either — the auto-generated cards just dump the `info.json` content into the README (e.g., `lerobot/aloha_sim_transfer_cube_human` has `robot_type: "aloha"` and no prose; `lerobot/xarm_push_medium` has `robot_type: "unknown"` despite being an xArm dataset). Robotics-specific metadata effectively lives only in LeRobot's `info.json`, which is a LeRobot-internal convention (defined by a `DatasetInfo` dataclass in `lerobot/datasets/utils.py`), not a broader HF standard — there is no JSON Schema, no HF-side validation on upload, and unknown keys are silently ignored.

---

## 5. Croissant (MLCommons)

**Source**: [Croissant Spec 1.1](https://docs.mlcommons.org/croissant/docs/croissant-spec-1.1.html), [MLCommons](https://mlcommons.org/working-groups/data/croissant/)

### DatasetDescriptor equivalent

JSON-LD metadata format layered on schema.org. Four layers:

1. **Metadata**: name, description, license, provenance (v1.1 adds W3C PROV-O)
2. **Resources**: `FileObject` and `FileSet` describing source data files
3. **Structure**: `RecordSet` with `Field` definitions — types, shapes, joins, transforms
4. **Semantics**: ML-specific types (`cr:Split`, `cr:BoundingBox`, `cr:Label`, `cr:SegmentationMask`)

Key `Field` properties:

- `dataType` — atomic (`sc:Integer`, `sc:Float`, `sc:Text`, `sc:DateTime`) or semantic (`sc:ImageObject`, `sc:VideoObject`)
- `source` — where data originates (FileObject, FileSet, or another RecordSet)
- `isArray` + `arrayShape` — multi-dimensional array support (v1.1)
- `references` — foreign-key relationships between RecordSets
- `subField` — nested structure

`DataSource` supports `extract` (column, jsonPath, fileProperty) and `transform` (separator, regex, readLines, unArchive) operations.

### What it doesn't describe

- **No temporal/sequential semantics**: no concept of episodes, trajectories, or time-series ordering
- **No robotics-specific types**: no action spaces, observation types, robot descriptions, or camera roles
- **No existing robotics extension** (extensions exist for RAI, geospatial, life sciences — not robotics)

### Extensibility for robotics

Croissant's extension mechanism allows linking fields to external ontologies via `@context` namespace declarations. A robotics extension could define semantic types for episodes, actions, observations, and embodiment. No one has started this work.

### Community adoption

700K+ datasets carry Croissant metadata. Integrated into HF, Kaggle, OpenML, Google Dataset Search. TensorFlow, JAX, and PyTorch load Croissant datasets natively.

---

## 6. BridgeData V2

**Source**: [project page](https://rail-berkeley.github.io/bridgedata/), [GitHub](https://github.com/rail-berkeley/bridge_data_v2)

### DatasetDescriptor equivalent

No structured metadata file. Dataset properties known from documentation:

| Property | Value |
| --- | --- |
| Robot | WidowX 250 6-DOF |
| Action dims | 7 (Cartesian delta EE xyz + rpy + gripper) |
| Control frequency | 5 Hz |
| Avg trajectory length | 38 timesteps |
| Trajectories | 60,096 (50,365 teleop + 9,731 scripted) |
| Environments | 24 |
| Skills | 13 |
| Image resolution | 640×480 |

Camera naming: "over-the-shoulder" (primary RGBD), two "randomized" RGB cameras, "wrist" (wide-angle). No formal camera identifiers — described in documentation text.

### EpisodeDescriptor equivalent

Each trajectory has a natural language instruction label. No structured per-episode metadata file.

### What's implicit

Everything: action space type, camera roles, joint ordering, normalization. All metadata is in documentation or must be inferred from code.

---

## 7. DROID

**Source**: [project page](https://droid-dataset.github.io/), [paper](https://arxiv.org/abs/2403.12945)

### DatasetDescriptor equivalent

TFDS/RLDS format on Google Cloud Storage. Dataset-level metadata:

| Property | Value |
| --- | --- |
| Robot | Franka Panda 7-DOF |
| Trajectories | 76,000 |
| Interaction time | 350 hours |
| Scenes | 564 |
| Tasks | 86 |
| Data collectors | 50 |
| Institutions | 13 |
| Regions | North America, Asia, Europe |

Camera naming: `exterior_image_1_left`, `exterior_image_2_left`, `wrist_image_left` (stereo pairs via Zed 2 / Zed Mini). Naming pattern: `{position}_image_{N}_{stereo_side}`.

DROID records **multiple action representations** simultaneously: joint velocity, Cartesian delta, Cartesian velocity — enabling post-hoc selection. This is unique among surveyed datasets.

### EpisodeDescriptor equivalent

Per-step: `observation`, `action`, `language_instruction`. Updated language annotations (3 per episode for 75K episodes) and improved camera calibrations distributed separately via HuggingFace.

### What's implicit

Action space semantics per representation, camera roles (inferred from naming convention), scene/task taxonomy.

---

## 8. LIBERO

**Source**: [GitHub](https://github.com/Lifelong-Robot-Learning/LIBERO), [HuggingFace](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets)

### DatasetDescriptor equivalent

Task suites defined in Python code and BDDL (Behavior Description Definition Language) files:

| Suite | Tasks | Purpose |
| --- | --- | --- |
| LIBERO-Spatial | ~10 | Transfer of spatial/relational knowledge |
| LIBERO-Object | ~10 | Transfer of object-related knowledge |
| LIBERO-Goal | ~10 | Transfer of goal/procedural knowledge |
| LIBERO-100 | 100 | Transfer of entangled knowledge (split: LIBERO-90 pretrain + LIBERO-10 eval) |

Action space: 7-dim continuous (3D position + 3D rotation + 1 gripper). Observations: `robot0_eye_in_hand_image` (wrist), `agentview_image` (third-person), plus proprioception. Images at 128×128.

### EpisodeDescriptor equivalent

Tasks have: `task.name`, `task.language` (natural language instruction), `task.bddl_file` (formal task specification). Each task has fixed initial states for reproducibility. Success is binary (+1 reward on completion).

### What's notable

BDDL files provide **formal task descriptions** — a structured language for specifying goals, making LIBERO's task metadata richer than most. However, this is environment-specific (robosuite/robomimic), not a portable standard.

### What's implicit

Dataset-level metadata (no info.json equivalent), camera roles (inferred from observation key names), action semantics.

---

## Property Coverage Matrix

| Property | LeRobot v3 | GR00T | RLDS/OXE | HF Cards | Croissant | BridgeData2 | DROID | LIBERO |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Feature schema** (names, shapes, types) | ✓ info.json | ✓ info.json + modality.json | ✓ TFDS builder | Partial (dataset_info) | ✓ RecordSet/Field | ✗ | ✓ TFDS | ✗ |
| **Episode structure** | ✓ episodes/ parquet | ✓ (LeRobot-based) | ✓ Steps with is_first/is_last | ✗ | ✗ (no temporal) | ✗ | ✓ RLDS steps | ✗ |
| **Per-episode task label** | ✓ tasks.jsonl + task_index | ✓ annotation channels | ✗ (per-step only) | ✗ | ✗ | ✓ NL per trajectory | ✓ per-step NL | ✓ task.language + BDDL |
| **Per-feature statistics** | ✓ stats.json | ✓ stats.json + relative_stats | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **Robot type** | ✓ string label | ✓ string label | ✗ (in docs only) | ✗ | ✗ | ✗ (in docs) | ✗ (in docs) | ✗ |
| **Action dim / shape** | ✓ features.action.shape | ✓ modality.json ranges | ✓ (in TFDS spec) | ✗ | ✓ (arrayShape) | ✗ | ✓ (in TFDS) | ✗ |
| **Action semantics** | ✗ | Partial (index splitting) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **Camera naming** | ✓ feature keys | ✓ + video key remapping | ✓ feature keys | ✗ | ✗ | ✗ (in docs) | ✓ naming pattern | ✓ obs keys |
| **Camera roles** | ✗ | ✗ (remap, not roles) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **FPS / control frequency** | ✓ fps field | ✓ (LeRobot-based) | ✗ | ✗ | ✗ | ✗ (in docs: 5Hz) | ✗ | ✗ |
| **Splits** | ✓ splits dict | ✓ | ✓ cr:Split | Partial | ✓ cr:Split | ✗ | ✗ | ✓ (suite-based) |
| **License** | Via HF card | Via HF card | ✗ | ✓ | ✓ | CC-BY-4.0 (docs) | ✗ | ✗ |
| **Provenance / lineage** | ✗ | ✗ | ✗ | ✗ | ✓ PROV-O (v1.1) | ✗ | ✗ | ✗ |
| **Normalization strategy** | ✗ (stats only) | ✗ (stats only) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **Embodiment structure** | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **Cross-artifact compat** | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **Multi-action repr** | ✗ | ✗ | Partial (raw + normalized) | ✗ | ✗ | ✗ | ✓ (3 repr) | ✗ |
| **Formal task spec** | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ BDDL |
| **Machine-readable schema** | ✗ (convention) | ✗ (convention) | ✗ (code-based) | Partial (YAML) | ✓ JSON-LD | ✗ | ✗ | ✗ |

---

## Recommended Properties for DatasetDescriptor

Based on the survey, a DatasetDescriptor should include the following properties, citing best source:

### Widely available (formalize what exists)

| Property | Best source | Notes |
| --- | --- | --- |
| `features` — dict of name → {shape, dtype, modality} | LeRobot `info.json` | Universal need; LeRobot's dot-notation convention is practical |
| `fps` — recording frame rate | LeRobot `info.json` | Only LeRobot declares this explicitly |
| `episode_count` / `total_frames` | LeRobot `info.json` | Basic dataset sizing |
| `splits` | LeRobot + Croissant `cr:Split` | Standard ML concept |
| `robot_type` — string identifier | LeRobot `info.json` | Free-form; needs controlled vocabulary |
| `task_descriptions[]` | LeRobot `tasks.jsonl` | Natural language task labels |
| `stats` — per-feature mean/std/min/max/q01/q99 | LeRobot `stats.json` ≈ OpenPI `norm_stats.json` | Near-convergent format |
| `license` | HF Cards / Croissant | Standard metadata |
| `format_version` | LeRobot `codebase_version` | Version tracking |

### Partially available (extend from fragments)

| Property | Best source | Gap to fill |
| --- | --- | --- |
| `observation: ObservationDescriptor` | OpenPI `Observation` dataclass | No community has a composite; assemble from feature dict |
| `action: ActionChunkDescriptor` | LeRobot features + GR00T modality.json | Shape exists; semantics missing |
| `collection_method` — teleop / autonomous / simulation | DROID docs (multi-collector metadata) | Always in docs, never structured |
| `environment` — scene description | LIBERO BDDL, DROID scene count | Varies from string to formal spec |
| `provenance` — source datasets, conversion chain | Croissant PROV-O | Only Croissant has this |

### Missing from all (novel properties)

| Property | Justification |
| --- | --- |
| `embodiment: EmbodimentDescriptor` | No dataset declares robot structure (joint ordering, limits, kinematics) |
| `action_semantics: ActionDescriptor` | No dataset declares what action dimensions mean |
| `camera_roles[]` | No dataset declares camera roles with controlled vocabulary |
| `normalization_strategy` | Stats exist but intended strategy is never declared |
| `compatible_models[]` | Which checkpoints/architectures work with this dataset — nowhere declared |
| `data_quality` | Episode success rate, annotation quality, filtering criteria |

---

## Recommended Properties for EpisodeDescriptor

| Property | Best source | Status |
| --- | --- | --- |
| `task` — task label or language instruction | LeRobot `task_index` → `tasks.jsonl` | Available in LeRobot, DROID, LIBERO |
| `length` — number of timesteps | LeRobot `episodes/*.parquet` | Available |
| `success` — bool or success metric | LIBERO (binary reward) | Rarely available; DROID has partial |
| `embodiment_id` | (none) | Needed for multi-robot datasets |
| `environment_id` / `scene_id` | DROID scene metadata | Available in DROID, implicit elsewhere |
| `operator_id` | DROID collector metadata | Privacy-sensitive; DROID anonymizes |
| `collection_timestamp` | (none) | Absent from all surveyed formats |
| `annotations` — quality labels, failure modes | (none) | Absent; would help data curation |
| `formal_task_spec` | LIBERO BDDL | Only LIBERO has this |

---

## Recommended Properties for TrainingConfigDescriptor

No community has a declarative training config descriptor. Training configuration exists as Python dataclasses (LeRobot policy configs, OpenPI TrainConfig) or command-line arguments. Key properties to declare:

| Property | Source | Notes |
| --- | --- | --- |
| `input_dataset` — DatasetDescriptor reference | LeRobot config `dataset_repo_id` | String ref, not typed |
| `normalization_mapping` — per-modality strategy | LeRobot `normalization_mapping` | Best existing source |
| `action_representation` — type + relative/absolute | GR00T `ActionRepresentation` | GR00T most structured |
| `camera_keys[]` — which cameras from dataset | OpenPI adapter classes | Hardcoded in Python |
| `image_transforms[]` — resize, crop, augmentation | LeRobot `ImageTransformsConfig` | Well-structured in LeRobot |
| `architecture` — model family + hyperparameters | LeRobot policy configs | Varies per arch |
| `produces_checkpoint` — CheckpointDescriptor | (none) | Nowhere declared |
| `target_embodiment` — intended deployment robot | (none) | Nowhere declared |

---

## Recommended Properties for DataConversionDescriptor

Conversion logic exists as hardcoded scripts. Key properties to declare:

| Property | Source | Notes |
| --- | --- | --- |
| `source_format` / `target_format` | LeRobot conversion scripts | Implicit in script names |
| `feature_mapping` — source → target feature names | LeRobot `rename_map`, GR00T `modality.json` | Partially declarative |
| `camera_mapping` — source → target camera keys | GR00T `modality.json` video remapping | Best existing source |
| `action_mapping` — joint reordering, sign flips | (none — see MappingDescriptor survey) | Universally hardcoded |
| `transforms[]` — normalization, resizing | Rosetta YAML `apply` field | Rosetta covers image resize |
| `provenance_chain` — ordered conversions | Croissant PROV-O | Only Croissant has provenance |

---

## What's Missing from All Communities

1. **Action semantics on datasets** — no dataset format declares what action dimensions mean (joint position vs. EE delta, which joints, what units). LeRobot has shape and dimension names; GR00T has index ranges. Neither has semantic type, unit, or frame.

2. **Embodiment structure in dataset metadata** — `robot_type` is a free-form string everywhere. No dataset links to a URDF/MJCF/USD file or declares joint ordering, limits, or kinematics. This is the root cause of the joint ordering problem.

3. **Camera role vocabulary** — every dataset names cameras differently (`cam_high`, `exterior_image_1_left`, `agentview_image`, `front`). No controlled vocabulary for roles (wrist, world, head, overhead, ego).

4. **Normalization contract** — stats exist (LeRobot `stats.json`, OpenPI `norm_stats.json`) but the intended normalization strategy is never stored in the dataset. It's in the policy config, creating a cross-artifact gap.

5. **Cross-artifact compatibility** — no dataset declares which models, architectures, or embodiments it's compatible with. Compatibility is determined by trying and failing.

6. **Composite observation descriptor** — no dataset groups cameras + proprioception + language into a typed composite. Features are a flat dict; the modality structure must be inferred from naming conventions.

7. **Data quality metadata** — episode success rate, annotation quality, filtering criteria, domain randomization parameters (for sim data). LIBERO has binary success; most datasets have nothing.

8. **Conversion provenance** — when a dataset is converted from one format to another (e.g., RLDS → LeRobot, rosbag → LeRobot), what transforms were applied? Only Croissant's PROV-O addresses this, and no robotics dataset uses Croissant.

---

## Key Findings

**1. LeRobot v3 `info.json` + `stats.json` is the de facto standard for dataset metadata.** It's the only format with explicit feature schema, FPS, episode structure, and per-feature statistics. Everything else either has less (BridgeData, DROID — metadata in docs only) or is losing momentum (RLDS).

**2. GR00T's `modality.json` demonstrates the metadata gap.** NVIDIA had to add a supplementary file on top of LeRobot because the base format lacks semantic array splitting and camera remapping. This validates that the gap is real and felt by major players.

**3. RLDS/OXE proved cross-embodiment datasets work but solved it by normalizing, not by describing.** OXE's approach was to convert all datasets to a common 7-DOF format, losing original action semantics. A descriptor approach would preserve original semantics while enabling cross-embodiment compatibility checking.

**4. Croissant is the right layer for governance metadata but lacks robotics primitives.** Provenance, licensing, and data use policies are well-handled by Croissant. Temporal structure, action spaces, and embodiment are not. A Croissant robotics extension is theoretically possible but no one has started, and adoption by the robotics community is uncertain.

**5. DROID's multi-representation recording is an underappreciated pattern.** Recording multiple action representations simultaneously (joint velocity + Cartesian delta + Cartesian velocity) avoids the information loss of choosing one format at recording time. A DatasetDescriptor should support declaring multiple action representations per dataset.

**6. LIBERO's BDDL task specs are the richest task metadata.** Most datasets have only natural language instructions. BDDL provides formal, machine-readable task specifications. A DatasetDescriptor could reference formal task specs alongside natural language descriptions, though BDDL is environment-specific.

**7. The training config is the least described artifact.** Every community has some form of training configuration, but it's always in Python code (dataclasses, argparse, hydra configs), never in a standalone metadata format. TrainingConfigDescriptor is genuinely novel territory.
