import orjson
import argparse
import numpy as np
from tqdm import tqdm
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str)
cfg = parser.parse_args()

if __name__ == '__main__':
    gtruth = None
    with open(cfg.src, "rb") as f:
        gtruth = orjson.loads(f.read())

    n_frames = len(gtruth)
    clip_data = {
        'car_count': np.zeros(n_frames),
        'max_skip': np.zeros(n_frames),
        'boxlists': [],
        'resolution': None,
        'src_path': None
    }

    max_skip = 0
    last_sample_start = 0
    last_result = -1

    for frame_id in tqdm(range(len(gtruth))):
        det = gtruth[frame_id]
        if det is None:
            det = []
        car_count = len(det)
        if car_count != last_result and frame_id != 0 or frame_id == n_frames - 1:
            assign_index_range = np.arange(last_sample_start, frame_id)
            clip_data['max_skip'][assign_index_range] = frame_id - assign_index_range
            skip = frame_id - last_sample_start
            if skip > max_skip:
                max_skip = skip
            if frame_id == n_frames - 1:
                clip_data['max_skip'][np.arange(last_sample_start, frame_id + 1)] += 1
            last_sample_start = frame_id

        last_result = car_count
        clip_data['car_count'][frame_id] = car_count
        tensor = torch.zeros((len(det), 5))
        for i, dslice in enumerate(det):
            tensor[i, :] = torch.Tensor([dslice['left'], dslice['top'], dslice['right'], dslice['bottom'], 1.])
        clip_data['boxlists'].append(tensor)

    with open(f'{cfg.src}.npy', 'wb') as f:
        np.save(f, clip_data)
