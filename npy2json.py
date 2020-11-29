import json
from collections import namedtuple
import argparse
import os
import numpy as np
import torch
from utility.common import iou_pairing
import multiprocessing as mp

item = namedtuple('item', ('frame_idx', 'track_id', 'left', 'top', 'right', 'bottom'))

parser = argparse.ArgumentParser()
parser.add_argument('--srcdir', type=str)
parser.add_argument('--dstdir', type=str)
cfg = parser.parse_args()

def tensor2item(tensor: torch.Tensor, frameid, tracks):
    ret = []
    if len(tensor) == 0:
        return []
    boxlist = tensor[:, :4].cpu().numpy()
    for xyxy, t in zip(boxlist, tracks):
        intxyxy = [int(element) for element in xyxy]
        ret.append(item(frameid, int(t), *intxyxy)._asdict())
    return ret

def single_job(args):
    npyidx, file = args
    todump = []
    data = np.load(os.path.join(cfg.srcdir, file), allow_pickle=True).item()['boxlists']
    start_index = 0
    last = []
    last_track_ids = None
    for frame_idx, cur in enumerate(data):
        pairings = iou_pairing(last, cur, use_index=True)  # MayBe Empty;;;

        tracks = []
        if len(pairings) != 0:  # There're pairs!
            track_pairs = pairings[:, -1]
            tracks = np.ones(len(cur)) * -1
            for p in track_pairs:
                if p[1] == -1:  # Car Disappeared!
                    continue
                if p[0] == -1:  # New Track!
                    tracks[p[1]] = start_index
                    start_index += 1
                else:  # There's a track!
                    tracks[p[1]] = last_track_ids[p[0]]

        todump.append(tensor2item(cur, frame_idx, tracks))
        last_track_ids = tracks
        last = cur

    with open(os.path.join(cfg.dstdir, f'{npyidx}-baseline.json'), 'w') as f:
        json.dump(todump, f)

    with open(os.path.join(cfg.dstdir, f'{npyidx}-detection.json'), 'w') as f:
        for group in todump:
            for item in group:
                item['track_id'] = -1
        json.dump(todump, f)

    print(f'DONE --> {file}')

if __name__ == '__main__':
    if not os.path.exists(cfg.dstdir):
        os.mkdir(cfg.dstdir)
    flist = sorted(os.listdir(cfg.srcdir))
    pool = mp.Pool(min(mp.cpu_count() - 1, len(flist)))
    results = pool.map(single_job, zip(range(len(flist)), flist))
    pool.close()