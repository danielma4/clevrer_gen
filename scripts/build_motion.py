#!/usr/bin/env python
"""Build dense motion tensors for existing clips. No rendering, no re-simulation.

Reads the annotation (poses), the derender_proposals RLE (exact object-id map)
and the reconstructed per-clip camera, and writes

    <root>/motion/<split>/motion_XXXXX-YYYYY/motion_NNNNN.npy

as float16 [T, 3, H/cell, W/cell], channels (dx, dy, visibility), with
displacement in latent-cell units.

    python scripts/build_motion.py output/*_nocollide --workers 16
"""
import argparse
import glob
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from clevrer_gen import motion as M              # noqa: E402
from clevrer_gen.annotate import _group_dir      # noqa: E402
from clevrer_gen.camera import Camera            # noqa: E402
from clevrer_gen.config import load_config       # noqa: E402


def motion_path(root, split, index):
    d = os.path.join(root, 'motion', split, _group_dir(index, 'motion'))
    return os.path.join(d, f'motion_{index:05d}.npy')


def _one(job):
    root, split, path, cfg_sizes, W, H, base_seed, T, cell, overwrite = job
    try:
        with open(path) as f:
            anno = json.load(f)
        index = anno['scene_index']
        out = motion_path(root, split, index)
        if os.path.exists(out) and not overwrite:
            return ('skip', index, None)

        prop_path = os.path.join(root, 'derender_proposals',
                                 f'proposal_{index:05d}.json')
        if not os.path.exists(prop_path):
            return ('nomask', index, None)
        with open(prop_path) as f:
            prop = json.load(f)

        if len(anno['motion_trajectory']) < T:
            return ('short', index, len(anno['motion_trajectory']))

        for o in anno['object_property']:
            o.setdefault('size_scale', float(cfg_sizes[o['size']]))

        seed = (anno.get('seed') if anno.get('seed') is not None else base_seed) + index
        cam = Camera(seed, W, H)
        arr = M.clip_motion(cam, anno, prop, T, cell, H, W)

        os.makedirs(os.path.dirname(out), exist_ok=True)
        tmp = out + '.tmp.npy'
        np.save(tmp, arr.astype(np.float16))
        os.replace(tmp, out)
        return ('ok', index, float(np.abs(arr[:, :2]).max()))
    except Exception as exc:                      # keep one bad clip from killing the run
        return ('error', path, f'{type(exc).__name__}: {exc}')


def build(root, T, cell, workers, overwrite):
    name = os.path.basename(os.path.normpath(root))
    cfg = load_config(scenario=os.path.join(os.path.dirname(__file__), '..',
                                            'configs', 'scenarios', f'{name}.yaml'))
    W, H = cfg['video']['width'], cfg['video']['height']
    sizes, base_seed = cfg['properties']['sizes'], cfg['seed']

    jobs = []
    for split_dir in sorted(glob.glob(os.path.join(root, 'annotations', '*'))):
        split = os.path.basename(split_dir)
        for path in sorted(glob.glob(os.path.join(split_dir, '*', '*.json'))):
            jobs.append((root, split, path, sizes, W, H, base_seed, T, cell, overwrite))
    if not jobs:
        print(f'  {name}: no annotations'); return

    counts, mags, bad = {}, [], []
    with Pool(workers) as pool:
        for status, key, info in pool.imap_unordered(_one, jobs, chunksize=8):
            counts[status] = counts.get(status, 0) + 1
            if status == 'ok' and info is not None:
                mags.append(info)
            elif status in ('error', 'short', 'nomask'):
                bad.append((status, key, info))

    meta = {'shape': [T, 3, H // cell, W // cell],
            'dtype': 'float16', 'cell': cell, 'resolution': [W, H],
            'channels': ['dx', 'dy', 'visibility'],
            'units': 'displacement in latent-cell units; multiply by cell for pixels',
            'frames': f'index t holds displacement t -> t+1; '
                      f'frame {T - 1} is zeroed with visibility 0',
            'source': 'analytic: rigid poses + IndexOB masks + per-clip camera'}
    os.makedirs(os.path.join(root, 'motion'), exist_ok=True)
    with open(os.path.join(root, 'motion', 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f'  {name}: {counts} max|disp| {max(mags) if mags else 0:.2f} cells')
    for b in bad[:5]:
        print(f'    !! {b}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('roots', nargs='+')
    ap.add_argument('--num_frames', type=int, default=64)
    ap.add_argument('--cell', type=int, default=8)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    for root in args.roots:
        build(root, args.num_frames, args.cell, args.workers, args.overwrite)


if __name__ == '__main__':
    main()
