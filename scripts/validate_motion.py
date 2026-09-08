#!/usr/bin/env python
"""Validate motion labels on random clips per subset. Prints pass/fail per check.

  shapes    RGB frame count, motion T and len(motion_trajectory) all agree
  camera    intrinsics + extrinsics present in the annotation
  ids       id map holds exactly {0} u {object_id + 1}, integral only
            (this is the check that catches a lossily-compressed id map)
  project   annotated 3D location, through the recorded camera, lands in that
            object's mask
  surface   ray/shape hits lie on the analytic surface (spheres: |X| == r)
  rigid     re-placed material points move by exactly the pose delta
  roundtrip reprojecting a hit point returns the pixel it came from
  warp      RGB frame t+1 sampled at p + flow matches frame t on moving pixels

surface/rigid/roundtrip are what catch a wrong quaternion order, handedness or
camera convention -- failures that otherwise yield plausible but systematically
wrong labels. warp is the only check touching pixels the label pipeline never
saw; its threshold is loose because the mp4 is lossy and metal objects have
view-dependent highlights, which break brightness constancy whether or not the
flow is correct.
"""
import argparse
import glob
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from clevrer_gen import motion as M                    # noqa: E402
from clevrer_gen.camera import Camera                  # noqa: E402
from clevrer_gen.config import load_config             # noqa: E402

CHECKS = ['shapes', 'camera', 'ids', 'project', 'surface', 'roundtrip',
          'rigid', 'warp']


def motion_path(root, split, index):
    lo = (index // 1000) * 1000
    return os.path.join(root, 'motion', split,
                        f'motion_{lo:05d}-{lo + 1000:05d}',
                        f'motion_{index:05d}.npy')


def _read_video(path):
    import cv2
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return np.array(frames, dtype=np.float32)


def check_clip(root, split, index, cfg, T, cell):
    import cv2
    W, H = cfg['video']['width'], cfg['video']['height']
    res = {}

    anno = json.load(open(glob.glob(os.path.join(
        root, 'annotations', split, '*', f'annotation_{index:05d}.json'))[0]))
    prop = json.load(open(os.path.join(
        root, 'derender_proposals', f'proposal_{index:05d}.json')))
    mot = np.load(motion_path(root, split, index))
    vid = glob.glob(os.path.join(root, 'videos', split, '*',
                                 f'video_{index:05d}.mp4'))[0]

    n_rgb = int(cv2.VideoCapture(vid).get(cv2.CAP_PROP_FRAME_COUNT))
    res['shapes'] = (mot.shape == (T, 3, H // cell, W // cell)
                     and len(anno['motion_trajectory']) >= T and n_rgb >= T)

    res['camera'] = 'cam_to_world' in anno.get('camera', {})
    if not res['camera']:
        return res, 'no camera in annotation'
    if 'render_half_extents' not in anno['object_property'][0]:
        return res, 'no render_half_extents in annotation'

    cam = Camera(anno['seed'] + index, W, H)
    maps = M.id_maps(prop, T, H, W)
    geoms = [M.object_geometry(o) for o in anno['object_property']]
    traj = anno['motion_trajectory']

    expected = {0} | {o['object_id'] + 1 for o in anno['object_property']}
    res['ids'] = (set(np.unique(maps).tolist()).issubset(expected)
                  and maps.dtype.kind in 'iu')

    proj_ok = proj_tot = 0
    surf, rt, rigid = [0.0], [0.0], [0.0]
    for t in range(0, T - 1, max(1, (T - 1) // 8)):
        for j, (geom, o) in enumerate(zip(geoms, anno['object_property'])):
            sel = maps[t] == j + 1
            if sel.sum() < 20:
                continue
            ys, xs = np.nonzero(sel)
            a, b = traj[t]['objects'][j], traj[t + 1]['objects'][j]
            c0, R0 = np.asarray(a['location']), M.quat_to_matrix(a['orientation'])
            c1, R1 = np.asarray(b['location']), M.quat_to_matrix(b['orientation'])

            # Only meaningful while the centre is on screen: an object leaving
            # frame keeps a visible sliver after its centre is gone, which is
            # what inside_camera_view records, not a projection error.
            p, _ = cam.project(c0)
            pxi, pyi = int(round(p[0])), int(round(p[1]))
            if 0 <= pxi < W and 0 <= pyi < H:
                proj_tot += 1
                proj_ok += int(maps[t][pyi, pxi] == j + 1)

            dirs = cam.pixel_rays(xs + 0.5, ys + 0.5)
            o_loc = np.broadcast_to((cam.location - c0) @ R0, dirs.shape)
            d_loc = dirs @ R0
            t_hit, _ = M.intersect_local(o_loc, d_loc, *geom)
            xl = o_loc + t_hit[:, None] * d_loc
            if o['shape'] == 'sphere':
                surf.append(float(np.abs(np.linalg.norm(xl, axis=1)
                                         - geom[1][0]).max()))
            xw = c0 + xl @ R0.T
            pr, _ = cam.project(xw)
            rt.append(float(np.abs(pr - np.stack([xs + 0.5, ys + 0.5], -1)).max()))
            if max(np.abs(R0 - np.eye(3)).max(), np.abs(R1 - np.eye(3)).max()) < 1e-9:
                rigid.append(float(np.abs(((c1 + xl @ R1.T) - xw) - (c1 - c0)).max()))

    # An on-screen centre may still be occluded by a nearer object, so allow a
    # small shortfall only where that is possible.
    floor = 1.0 if len(geoms) == 1 else 0.98
    res['project'] = proj_tot > 0 and proj_ok / proj_tot >= floor
    res['surface'] = max(surf) < 1e-6
    res['roundtrip'] = max(rt) < 1e-2
    res['rigid'] = max(rigid) < 1e-9

    frames = _read_video(vid)
    yy, xx = np.mgrid[0:H, 0:W]
    we, ze = [], []
    for t in range(0, min(T - 1, len(frames) - 1), max(1, (T - 1) // 5)):
        poses = [((np.asarray(traj[t]['objects'][j]['location']),
                   M.quat_to_matrix(traj[t]['objects'][j]['orientation'])),
                  (np.asarray(traj[t + 1]['objects'][j]['location']),
                   M.quat_to_matrix(traj[t + 1]['objects'][j]['orientation'])))
                 for j in range(len(geoms))]
        flow, valid = M.dense_motion(cam, poses, geoms, maps[t], maps[t + 1])
        moving = valid & (np.hypot(flow[..., 0], flow[..., 1]) > 0.5)
        if moving.sum() < 50:
            continue
        warped = cv2.remap(frames[t + 1], (xx + flow[..., 0]).astype(np.float32),
                           (yy + flow[..., 1]).astype(np.float32),
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        we.append(np.abs(warped - frames[t])[moving].mean())
        ze.append(np.abs(frames[t + 1] - frames[t])[moving].mean())
    # No moving pixels means nothing to test, not a failure: new_velos draws
    # speed from (0,1) u [7,10], so some clips move well under a pixel a frame.
    if not we:
        res['warp'] = None
        res['explained'] = float('nan')
    else:
        res['warp'] = np.mean(we) < 0.4 * np.mean(ze)
        res['explained'] = 100 * (1 - np.mean(we) / np.mean(ze))
    return res, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('roots', nargs='+')
    ap.add_argument('--clips', type=int, default=5)
    ap.add_argument('--num_frames', type=int, default=64)
    ap.add_argument('--cell', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    all_ok = True
    for root in args.roots:
        name = os.path.basename(os.path.normpath(root))
        cfg = load_config(scenario=os.path.join(
            os.path.dirname(__file__), '..', 'configs', 'scenarios', f'{name}.yaml'))
        paths = sorted(glob.glob(os.path.join(root, 'motion', '*', '*', '*.npy')))
        if not paths:
            print(f'\n{name}: NO MOTION TENSORS'); all_ok = False; continue
        picks = random.Random(args.seed).sample(paths, min(args.clips, len(paths)))

        print(f'\n{name}')
        agg = {k: [] for k in CHECKS}
        expl = []
        for p in picks:
            split = p.split(os.sep)[-3]
            index = int(os.path.splitext(os.path.basename(p))[0].split('_')[-1])
            try:
                res, err = check_clip(root, split, index, cfg,
                                      args.num_frames, args.cell)
            except Exception as exc:
                print(f'  clip {index:05d}: ERROR {type(exc).__name__}: {exc}')
                all_ok = False
                continue
            if err:
                print(f'  clip {index:05d}: {err}'); all_ok = False; continue
            line = ' '.join(
                f'{k}:' + ('n/a' if res[k] is None else 'PASS' if res[k] else 'FAIL')
                for k in CHECKS)
            note = ('static clip' if np.isnan(res['explained'])
                    else f'warp explains {res["explained"]:.0f}%')
            print(f'  clip {index:05d} {line} ({note})')
            for k in CHECKS:
                if res[k] is not None:
                    agg[k].append(res[k])
            if not np.isnan(res['explained']):
                expl.append(res['explained'])
        for k in CHECKS:
            if agg[k] and not all(agg[k]):
                all_ok = False
        print('  SUMMARY ' + ' '.join(f'{k}:{sum(agg[k])}/{len(agg[k])}'
                                      for k in CHECKS if agg[k])
              + f'  mean warp explained {np.mean(expl) if expl else float("nan"):.0f}%')

    print(f'\n{"ALL CHECKS PASSED" if all_ok else "FAILURES PRESENT"}')
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
