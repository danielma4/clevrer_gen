#!/usr/bin/env python
"""Write train/val index-list files for each scenario (80/20 by default).

Does not move or touch any video/annotation/flow files - all data stays
under videos/train, annotations/train, flow/train as generated. Instead
writes splits/train.txt and splits/val.txt per scenario, one video index
per line, for the dataloader to select from.

Usage:
  python scripts/split_dataset.py output/one_ball_nocollide --val_frac 0.2
  python scripts/split_dataset.py output/*_nocollide --val_frac 0.2 --seed 0
"""
import argparse
import glob
import os

import numpy as np


def _read_split(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return [int(line.strip()) for line in f if line.strip()]


def split_scenario(root, val_frac, seed, update):
    train_video_dir = os.path.join(root, 'videos', 'train')
    if not os.path.isdir(train_video_dir):
        print(f'  skip {root}: no videos/train')
        return

    indices = sorted(
        int(os.path.splitext(f)[0].split('_')[-1])
        for bucket in os.listdir(train_video_dir)
        for f in os.listdir(os.path.join(train_video_dir, bucket))
        if f.endswith('.mp4')
    )
    if not indices:
        print(f'  skip {root}: no videos found')
        return

    split_dir = os.path.join(root, 'splits')
    existing_val = _read_split(os.path.join(split_dir, 'val.txt'))

    if update and existing_val is not None:
        # keep val frozen; every index not already in val goes to train
        val_indices = set(existing_val)
        train_indices = [i for i in indices if i not in val_indices]
    else:
        rng = np.random.default_rng(seed)
        n_val = round(len(indices) * val_frac)
        val_indices = set(rng.choice(indices, size=n_val, replace=False).tolist())
        train_indices = [i for i in indices if i not in val_indices]

    os.makedirs(split_dir, exist_ok=True)
    with open(os.path.join(split_dir, 'train.txt'), 'w') as f:
        f.write('\n'.join(f'{i:05d}' for i in train_indices) + '\n')
    with open(os.path.join(split_dir, 'val.txt'), 'w') as f:
        f.write('\n'.join(f'{i:05d}' for i in sorted(val_indices)) + '\n')

    print(f'  {root}: {len(indices)} total -> {len(train_indices)} train / {len(val_indices)} val'
          + (' (val frozen, update mode)' if update and existing_val is not None else ''))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('scenario_dirs', nargs='+', help='e.g. output/one_ball_nocollide (globs ok)')
    ap.add_argument('--val_frac', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--update', action='store_true',
                     help='keep existing val.txt frozen; add all other indices to train')
    args = ap.parse_args()

    roots = sorted({r for pattern in args.scenario_dirs for r in glob.glob(pattern)})
    for root in roots:
        split_scenario(root, args.val_frac, args.seed, args.update)


if __name__ == '__main__':
    main()
