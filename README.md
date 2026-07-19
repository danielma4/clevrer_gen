# clevrer-gen

Config-driven generation of CLEVRER-style collision videos. Objects, their
counts, initial velocities, accelerations, materials, and all CLEVRER
attributes are controlled directly from a YAML config — point it at a scenario
file to probe a specific dynamics setup.

Physics is simulated with **pybullet**; rendering uses **Blender 4.2 via the
`bpy` Python module** (no separate Blender binary). Output is written in the
**CLEVRER on-disk layout**, so it loads directly into downstream CLEVRER tooling
(e.g. SlotFormer) with no adapter.

## Install

```bash
conda create -n clevrergen python=3.11 -y
conda activate clevrergen
pip install "bpy==4.2.0" pybullet numpy scipy pyyaml tqdm \
            pycocotools opencv-python imageio imageio-ffmpeg
```

## Usage

Each run writes to `output/<name>/` (full CLEVRER layout inside), where `<name>`
defaults to the scenario name. Override with `--name`, the parent dir with
`--output_base`, or give an exact path with `--output_root`.

```bash
# defaults (configs/default.yaml): 5 random train videos -> output/default/
python generate_dataset.py

# a controlled scenario -> output/two_collide/
python generate_dataset.py --scenario configs/scenarios/two_collide.yaml \
    --num_videos 20 --set output.split=val seed=1

# pick the run name explicitly -> output/my_run/
python generate_dataset.py --scenario configs/scenarios/two_collide.yaml --name my_run
```

### Distribute across GPUs

The total `output.num_videos` is split round-robin across shards, one process per
GPU, no overlap:

```bash
# all MIG slices on this node, one process each -> output/two_collide/
scripts/launch_local.sh --scenario configs/scenarios/two_collide.yaml --num_videos 800

# or as a SLURM array (array size == shard count; one GPU per task)
sbatch scripts/launch_slurm.sbatch --scenario configs/scenarios/two_collide.yaml --num_videos 800
```

Manually, each process just needs `--num_shards N --shard_id k`.

### Bundled scenarios (`configs/scenarios/`)

| scenario | what it makes |
|----------|---------------|
| `one_ball.yaml` | one sphere, random position + (constant) velocity per video |
| `two_parallel.yaml` | two balls in separate lanes; random speed/direction; never collide |
| `two_collide.yaml` | two balls launched head-on (deterministic collision) |
| `many_fast.yaml` | 8-10 fast objects aimed at center (many collisions) |

Everything in `configs/default.yaml` is overridable by a scenario file (deep
-merged) or `--set key.sub=value`. Key knobs:

| field | meaning |
|-------|---------|
| `objects.num` | int or `[min,max]` count per video |
| `objects.speed` / `direction` | initial speed range; `random`/`toward_center`/`away_from_center`/`[vx,vy]` |
| `objects.accel` | constant acceleration applied to all objects |
| `objects.explicit` | per-object overrides (shape/color/material/size/position/velocity/...) |
| `physics.*` | gravity, friction, restitution, mass, damping, substeps |
| `video.*` | num_frames, fps, width, height |
| `render.*` | device (OPTIX/CUDA/CPU), samples, passes (segmentation, flow) |

## Top-down (orthographic overhead) view

A Miniworld-style bird's-eye view, available two ways. Either renders top-down
clip `i` frame-for-frame matched to the 3D clip `i`.

**Integrated (recommended):** toggle `render.views`. Both views are produced
from the *same single simulation/seed* in one run:

```bash
python generate_dataset.py --scenario configs/scenarios/two_collide.yaml \
    --num_videos 5 --set render.views.perspective=true render.views.topdown=true
# -> videos/...  (3D + annotations/masks)   and   topdown/...  (overhead)
```

Set `perspective=false` to emit only the top-down view.

**Standalone:** `render_topdown.py` is an independent pipeline (re-simulates from
`seed + index`, shares only `physics`/`materials`/`properties`, never touches the
main RGB code). Same flags as `generate_dataset.py`:

```bash
python render_topdown.py --scenario configs/scenarios/two_collide.yaml --num_videos 5
```

Both write `<root>/topdown/<split>/video_<a>-<b>/video_<i>.mp4`. Tunable under the
`topdown:` config block: `resolution` (square), `margin`, `ortho_scale`.

## Output layout (CLEVRER-strict)

Each run lives under `output/<name>/`, which is itself a complete CLEVRER-format
`data_root` (point SlotFormer straight at it):

```
output/<name>/
  videos/<split>/video_<a>-<b>/video_<i>.mp4                    # 3D (perspective) view
  annotations/<split>/annotation_<a>-<b>/annotation_<i>.json   # object_property, motion_trajectory, collisions
  derender_proposals/proposal_<i>.json                          # per-frame RLE masks + bboxes
  topdown/<split>/video_<a>-<b>/video_<i>.mp4                    # overhead view (if enabled)
  flow/<split>/flow_<i>.npy                                      # dense motion vectors (if enabled)
```

## Architecture

- `physics.py` — pybullet simulation → per-frame trajectory + collision events (standalone, no bpy).
- `render.py` — procedural Blender scene; RGB + object-index (segmentation) + vector (flow) passes.
- `annotate.py` — CLEVRER-strict output adapter (the internal schema is general; other emitters can be added here).
- `generate.py` — orchestrates physics → render → annotate per video.
