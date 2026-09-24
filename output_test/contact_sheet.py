import glob, json, os
import imageio.v2 as iio
import numpy as np
import cv2

root = os.path.dirname(os.path.abspath(__file__))
for run in ['balls_accel_test', 'balls_accel_ood_test']:
    rows = []
    for mp4 in sorted(glob.glob(f'{root}/{run}/videos/train/*/*.mp4')):
        i = int(mp4[-9:-4])
        anno = json.load(open(glob.glob(f'{root}/{run}/annotations/train/*/annotation_{i:05d}.json')[0]))
        frames = iio.mimread(mp4)
        tiles = [cv2.resize(frames[t], (240, 160)) for t in range(0, len(frames), 9)]
        row = np.concatenate(tiles, 1)
        acc = ' '.join('%.1f' % np.hypot(*o['accel'][:2]) for o in anno['object_property'])
        cv2.putText(row, f'v{i} n={len(anno["object_property"])} |a|={acc}', (5, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        rows.append(row)
    iio.imwrite(f'{root}/{run}_sheet.png', np.concatenate(rows, 0))
