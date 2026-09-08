#!/usr/bin/env python
"""Add `camera` and numeric object geometry to existing annotation JSONs.

Neither can be recovered from the annotation alone: the camera is jittered per
clip from (seed, index), and `size` is only a label -- in this dataset both
"large" and "small" map to the same 0.35 radius, so the label carries no
geometry. Everything else in the file is left untouched.

These clips all predate the 2026-08-13 render fix, so render_half_extents is
written with the legacy (unscaled) geometry that actually produced their pixels,
while physics_half_extents records what pybullet collided with. The two differ
for cubes and cylinders, i.e. only in new_shapes_nocollide.

    python scripts/backfill_annotations.py output/one_ball_nocollide
    python scripts/backfill_annotations.py output/*_nocollide
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from clevrer_gen import properties as props    # noqa: E402
from clevrer_gen.camera import Camera          # noqa: E402
from clevrer_gen.config import load_config     # noqa: E402


def scenario_for(root):
    name = os.path.basename(os.path.normpath(root))
    path = os.path.join(os.path.dirname(__file__), '..',
                        'configs', 'scenarios', f'{name}.yaml')
    if not os.path.exists(path):
        raise SystemExit(f'no scenario yaml for {name}')
    return load_config(scenario=path)


def backfill(root, dry_run=False):
    cfg = scenario_for(root)
    sizes = cfg['properties']['sizes']
    W, H = cfg['video']['width'], cfg['video']['height']
    base_seed = cfg['seed']

    files = sorted(glob.glob(os.path.join(root, 'annotations', '*', '*', '*.json')))
    if not files:
        print(f'  skip {root}: no annotations')
        return 0

    cams = {}
    done = 0
    for path in files:
        with open(path) as f:
            anno = json.load(f)
        index = anno['scene_index']
        seed = (anno.get('seed') if anno.get('seed') is not None else base_seed) + index
        if seed not in cams:
            cams[seed] = Camera(seed, W, H).to_dict()

        for obj in anno['object_property']:
            s = float(sizes[obj['size']])
            obj['size_scale'] = s
            obj['render_half_extents'] = props.render_half_extents(
                obj['shape'], s, legacy=True)
            obj['physics_half_extents'] = props.render_half_extents(
                obj['shape'], s)
        anno['camera'] = cams[seed]

        if not dry_run:
            tmp = path + '.tmp'
            with open(tmp, 'w') as f:
                json.dump(anno, f)
            os.replace(tmp, path)
        done += 1
    print(f'  {root}: {done} annotations'
          f'{" (dry run)" if dry_run else ""}, {W}x{H}, seed base {base_seed}')
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('roots', nargs='+')
    ap.add_argument('--dry_run', action='store_true')
    args = ap.parse_args()
    total = sum(backfill(r, args.dry_run) for r in args.roots)
    print(f'total: {total} annotations')


if __name__ == '__main__':
    main()
