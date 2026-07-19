"""Orchestration: run physics -> render -> annotate for one video, in CLEVRER
layout. Frames are rendered to a temp dir, muxed to mp4, then annotations are
written. The whole thing is deterministic in (seed, index).
"""
import os
import shutil
import tempfile

import imageio.v2 as imageio
import numpy as np

from . import annotate
from .config import ensure_root, load_config, resolve_seed  # noqa: F401
from .physics import build_objects, simulate
from .render import Renderer


def _mux(frame_paths, out_path, fps):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with imageio.get_writer(out_path, fps=fps, codec='libx264',
                            quality=8, macro_block_size=16) as w:
        for fp in frame_paths:
            w.append_data(imageio.imread(fp))


def _collisions_ok(count, filt):
    """Check a collision count against a {min?, max?} filter spec."""
    return ('min' not in filt or count >= filt['min']) and \
           ('max' not in filt or count <= filt['max'])


def _simulate_filtered(cfg, index):
    """Re-roll the (cheap) physics until the collision count matches the filter,
    so we only ever render an accepted simulation. Returns (objects, sim) or
    (None, None) if no accepted roll is found within the budget."""
    filt = cfg['output'].get('collision_filter')
    max_tries = cfg['output'].get('collision_filter_max_tries', 50)
    for attempt in range(max_tries):
        rng = np.random.default_rng(cfg['seed'] + index + attempt * 100003)
        objects = build_objects(cfg, rng)
        sim = simulate(cfg, objects)
        if filt is None or _collisions_ok(len(sim['collisions']), filt):
            return objects, sim
    print(f'[clevrer-gen] video {index}: collision_filter {filt} not met '
          f'in {max_tries} tries; skipping', flush=True)
    return None, None


def _render_perspective(cfg, index, objects, sim):
    """Standard CLEVRER 3D view: mp4 + annotation + (optional) masks/flow."""
    root, split = cfg['output']['root'], cfg['output']['split']
    out_mp4 = annotate.video_path(root, split, index)
    renderer = Renderer(cfg, seed=cfg['seed'] + index)
    renderer.add_objects(sim['object_property'])
    renderer.animate(sim['trajectory'])

    do_flow = cfg['render']['passes']['flow']
    do_seg = cfg['render']['passes']['segmentation'] or do_flow
    tmp = tempfile.mkdtemp(prefix='clevrergen_')
    try:
        frame_paths, inside_view, frame_masks, flows = [], [], [], []
        for t, frame in enumerate(sim['trajectory']):
            renderer.set_pose(frame['objects'])
            fp = os.path.join(tmp, f'rgb_{t:05d}.png')
            renderer.render_frame(t, fp, tmp)
            frame_paths.append(fp)
            inside_view.append([renderer.project(o['location'])[3]
                                for o in frame['objects']])
            if do_seg:
                frame_masks.append(renderer.read_masks(t, len(objects), tmp))
            if do_flow:
                flows.append(renderer.compute_flow(t, sim['trajectory'], frame_masks))

        _mux(frame_paths, out_mp4, cfg['video']['fps'])
        if cfg['output']['save_frames']:
            _save_frames(frame_paths, root, split, index)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    annotate.write_annotation(root, split, index, os.path.basename(out_mp4),
                              sim, inside_view, seed=cfg['seed'])
    if do_seg:
        annotate.write_proposals(root, index, sim, frame_masks)
    if do_flow:
        flow_path = os.path.join(root, 'flow', split, f'flow_{index:05d}.npy')
        os.makedirs(os.path.dirname(flow_path), exist_ok=True)
        np.save(flow_path, np.stack(flows).astype(np.float16))
    return out_mp4


def _render_topdown_view(cfg, index, sim):
    """Orthographic overhead mp4 of the same simulation (imported lazily to
    avoid a circular import; rebuilds the bpy scene, so call it last)."""
    from .topdown import TopDownRenderer, topdown_path
    root, split = cfg['output']['root'], cfg['output']['split']
    out_mp4 = topdown_path(root, split, index)
    renderer = TopDownRenderer(cfg)
    renderer.add_objects(sim['object_property'])
    tmp = tempfile.mkdtemp(prefix='clevrergen_top_')
    try:
        frame_paths = []
        for t, frame in enumerate(sim['trajectory']):
            renderer.set_pose(frame['objects'])
            fp = os.path.join(tmp, f'{t:05d}.png')
            renderer.render_frame(t, fp)
            frame_paths.append(fp)
        _mux(frame_paths, out_mp4, cfg['video']['fps'])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out_mp4


def generate_video(cfg, index):
    """Generate the enabled view(s) for one scene from a single simulation.
    `render.views` toggles {perspective, topdown}; both share one seed+sim."""
    resolve_seed(cfg)
    ensure_root(cfg)
    root, split = cfg['output']['root'], cfg['output']['split']
    views = cfg['render'].get('views') or {}
    do_persp = views.get('perspective', True)
    do_top = views.get('topdown', False)
    overwrite = cfg['output']['overwrite']

    from .topdown import topdown_path
    persp_mp4 = annotate.video_path(root, split, index)
    top_mp4 = topdown_path(root, split, index)
    need_persp = do_persp and (overwrite or not os.path.exists(persp_mp4))
    need_top = do_top and (overwrite or not os.path.exists(top_mp4))
    if not (need_persp or need_top):
        return persp_mp4 if do_persp else top_mp4

    objects, sim = _simulate_filtered(cfg, index)
    if sim is None:
        return None

    # Perspective first: the top-down renderer rebuilds (wipes) the bpy scene.
    if need_persp:
        _render_perspective(cfg, index, objects, sim)
    if need_top:
        _render_topdown_view(cfg, index, sim)
    return persp_mp4 if do_persp else top_mp4


def _save_frames(frame_paths, root, split, index):
    frame_dir = os.path.join(root, 'frames', split, f'video_{index:05d}')
    os.makedirs(frame_dir, exist_ok=True)
    for t, fp in enumerate(frame_paths):
        shutil.copy(fp, os.path.join(frame_dir, f'{t:05d}.png'))


def generate_dataset(cfg, shard_id=0, num_shards=1):
    """Generate videos [start, start+num). With num_shards>1, this process only
    renders the round-robin subset `i % num_shards == shard_id`, so N processes
    (one per GPU) cover the range with balanced load and no overlap."""
    resolve_seed(cfg)   # resolve once so all shards/videos share the same seed
    ensure_root(cfg)
    start = cfg['output']['start_index']
    indices = [i for i in range(start, start + cfg['output']['num_videos'])
               if (i - start) % num_shards == shard_id]
    print(f'[clevrer-gen] shard {shard_id}/{num_shards}: {len(indices)} videos',
          flush=True)
    paths = []
    for n, i in enumerate(indices):
        p = generate_video(cfg, i)
        if p is not None:
            paths.append(p)
        print(f'[clevrer-gen] shard {shard_id} [{n + 1}/{len(indices)}] '
              f'video {i} -> {p}', flush=True)
    return paths
