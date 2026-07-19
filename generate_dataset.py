#!/usr/bin/env python
"""Entry point: generate a CLEVRER-format dataset from a config.

Output goes to output/<name>/ (CLEVRER layout inside), where <name> defaults to
the scenario name. Override with --name, or --output_root for an exact path.

    python generate_dataset.py --scenario configs/scenarios/two_collide.yaml \
        --set output.num_videos=20 output.split=val
"""
import argparse
import os

from clevrer_gen.config import load_config
from clevrer_gen.generate import generate_dataset


def apply_common_args(cfg, args):
    """Apply the shared convenience flags onto a loaded config (in place)."""
    if args.random:
        cfg['seed'] = None
    elif args.seed is not None:
        cfg['seed'] = args.seed
    if args.num_videos is not None:
        cfg['output']['num_videos'] = args.num_videos
    # run name: --name > scenario stem > whatever the config already has
    if args.name:
        cfg['output']['name'] = args.name
    elif args.scenario and cfg['output'].get('name') in (None, 'default'):
        cfg['output']['name'] = os.path.splitext(os.path.basename(args.scenario))[0]
    if args.output_base is not None:
        cfg['output']['base'] = args.output_base
    if args.output_root is not None:
        cfg['output']['root'] = args.output_root   # exact path, overrides base/name
    if args.overwrite:
        cfg['output']['overwrite'] = True


def add_common_args(ap):
    ap.add_argument('--seed', type=int, default=None, help='fixed RNG seed')
    ap.add_argument('--random', action='store_true',
                    help='use a fresh random seed each run (recorded in annotation)')
    ap.add_argument('--num_videos', type=int, default=None)
    ap.add_argument('--name', default=None,
                    help='run subdir under output base (default: scenario name)')
    ap.add_argument('--output_base', default=None, help='parent dir (default: output)')
    ap.add_argument('--output_root', default=None,
                    help='exact data_root path, overriding base/name')
    ap.add_argument('--overwrite', action='store_true',
                    help='re-render videos even if their mp4 already exists')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', default=None,
                    help='scenario yaml merged over configs/default.yaml')
    ap.add_argument('--set', nargs='*', default=[], dest='overrides',
                    help='dotted overrides, e.g. objects.num=4 seed=1')
    ap.add_argument('--num_shards', type=int, default=1,
                    help='split the index range across this many processes/GPUs')
    ap.add_argument('--shard_id', type=int, default=0,
                    help='which shard this process renders (0..num_shards-1)')
    add_common_args(ap)
    args = ap.parse_args()

    cfg = load_config(scenario=args.scenario, overrides=args.overrides)
    apply_common_args(cfg, args)
    generate_dataset(cfg, shard_id=args.shard_id, num_shards=args.num_shards)


if __name__ == '__main__':
    main()
