#!/usr/bin/env python
"""Entry point: render top-down (orthographic overhead) videos for the same
scenes the main pipeline produces. Mirrors generate_dataset.py's flags.

    python render_topdown.py --scenario configs/scenarios/two_collide.yaml \
        --num_videos 5 --output_root output
-> <root>/topdown/<split>/video_<a>-<b>/video_<i>.mp4
"""
import argparse

from clevrer_gen.config import load_config
from clevrer_gen.topdown import render_topdown_dataset
from generate_dataset import add_common_args, apply_common_args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', default=None)
    ap.add_argument('--set', nargs='*', default=[], dest='overrides')
    ap.add_argument('--num_shards', type=int, default=1)
    ap.add_argument('--shard_id', type=int, default=0)
    add_common_args(ap)
    args = ap.parse_args()

    cfg = load_config(scenario=args.scenario, overrides=args.overrides)
    apply_common_args(cfg, args)
    render_topdown_dataset(cfg, shard_id=args.shard_id, num_shards=args.num_shards)


if __name__ == '__main__':
    main()
