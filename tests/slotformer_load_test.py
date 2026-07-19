"""Acceptance test: load a clevrer-gen dataset through SlotFormer's own
CLEVRERDataset, exercising video frames + RLE masks. Run with the SlotFormer
env (decomp_diff) and its libstdc++ on LD_LIBRARY_PATH:

    DD=.../envs/decomp_diff
    LD_LIBRARY_PATH=$DD/lib $DD/bin/python tests/slotformer_load_test.py <data_root>
"""
import sys

sys.path.insert(0, '/net/holy-isilon/ifs/rc_labs/ydu_lab/dma/SlotFormer')

from slotformer.base_slots.datasets.clevrer import CLEVRERDataset
from slotformer.base_slots.datasets.utils import BaseTransforms

data_root = sys.argv[1]
ds = CLEVRERDataset(
    data_root=data_root,
    clevrer_transforms=BaseTransforms((64, 64)),
    split='train',
    n_sample_frames=6,
    video_len=128,
    load_mask=True,
)
print('num_videos:', ds.num_videos, 'num samples:', len(ds.valid_idx))
sample = ds[0]
for k, v in sample.items():
    shape = tuple(v.shape) if hasattr(v, 'shape') else v
    print(f'  {k}: {shape} {getattr(v, "dtype", type(v).__name__)}')
print('OK: SlotFormer CLEVRERDataset loaded clevrer-gen output.')
