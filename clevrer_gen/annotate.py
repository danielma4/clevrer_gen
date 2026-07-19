"""CLEVRER-strict output adapter.

Writes the three artifacts SlotFormer reads, given the general internal
representation (sim result + per-frame projection + per-frame masks):
    videos/{split}/video_{a:05d}-{b:05d}/video_{i:05d}.mp4
    annotations/{split}/annotation_{a:05d}-{b:05d}/annotation_{i:05d}.json
    derender_proposals/proposal_{i:05d}.json
The internal schema is kept separate so other emitters can be added later.
"""
import json
import os

import numpy as np
from pycocotools import mask as mask_utils


def _group_dir(index, prefix):
    lo = (index // 1000) * 1000
    return f'{prefix}_{lo:05d}-{lo + 1000:05d}'


def video_path(root, split, index):
    d = os.path.join(root, 'videos', split, _group_dir(index, 'video'))
    return os.path.join(d, f'video_{index:05d}.mp4')


def annotation_path(root, split, index):
    d = os.path.join(root, 'annotations', split, _group_dir(index, 'annotation'))
    return os.path.join(d, f'annotation_{index:05d}.json')


def proposal_path(root, index):
    return os.path.join(root, 'derender_proposals', f'proposal_{index:05d}.json')


def encode_mask(binary):
    """[H, W] {0,1} array -> COCO RLE with a JSON-serialisable str `counts`."""
    rle = mask_utils.encode(np.asfortranarray(binary.astype(np.uint8)))
    rle['counts'] = rle['counts'].decode('ascii')
    return rle


def write_annotation(root, split, index, video_fn, sim, inside_view, seed=None):
    """Ground-truth physics annotation (object_property, motion, collisions).

    `inside_view[t][j]` is the bool from camera projection for frame t, object j.
    """
    motion = []
    for t, frame in enumerate(sim['trajectory']):
        objs = []
        for j, st in enumerate(frame['objects']):
            objs.append({
                'object_id': j,
                'location': st['location'],
                'orientation': st['orientation'],
                'velocity': st['velocity'],
                'angular_velocity': st['angular_velocity'],
                'inside_camera_view': bool(inside_view[t][j]),
            })
        motion.append({'frame_id': t, 'objects': objs})

    obj_prop = [{'object_id': o['id'], 'shape': o['shape'], 'color': o['color'],
                 'material': o['material'], 'size': o['size']}
                for o in sim['object_property']]
    collisions = [{'frame_id': c['frame'], 'object_ids': c['object_ids'],
                   'location': c['location']} for c in sim['collisions']]

    anno = {'scene_index': index, 'video_filename': video_fn, 'seed': seed,
            'object_property': obj_prop, 'motion_trajectory': motion,
            'collisions': collisions}
    path = annotation_path(root, split, index)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(anno, f)
    return path


def write_proposals(root, index, sim, frame_masks):
    """Per-frame object masks/bboxes. `frame_masks[t]` is list of [H,W] arrays
    aligned with object ids (None where the object is absent from the frame)."""
    obj_prop = {o['id']: o for o in sim['object_property']}
    frames = []
    for t, masks in enumerate(frame_masks):
        objs = []
        for j, m in enumerate(masks):
            if m is None or m.sum() == 0:
                continue
            ys, xs = np.where(m)
            bbox = [int(xs.min()), int(ys.min()),
                    int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]
            o = obj_prop[j]
            objs.append({'object_id': j, 'shape': o['shape'], 'color': o['color'],
                         'material': o['material'], 'size': o['size'],
                         'mask': encode_mask(m), 'bbox': bbox})
        frames.append({'frame_index': t, 'objects': objs})

    prop = {'scene_index': index, 'frames': frames}
    path = proposal_path(root, index)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(prop, f)
    return path
