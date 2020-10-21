"""
Execution Command:
var=1 && \
for i in $(seq 4);
    do declare -i var=var*2 && \
        python best_evaluator.py --action_space $var && \
        python random_evaluator.py --action_space $var ;
done
"""

import os
import sys
import random
from tqdm import tqdm
import numpy as np

project_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_dir)

from utility.videoloader import VideoSamplerDataset
from utility.improcessing import _boxembedding, _boxlist2tensor_channelstack
from models.backbone import BoxNN, SimpleBoxMaskCNN
import torch
import argparse

RATE_OPTIONS = [0, 2, 4, 8, 12, 16]
parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='embedding')
parser.add_argument('--model', type=str)
parser.add_argument('--n_prev', type=int, default=8)
parser.add_argument('--fixed_skipping', type=int, default=None)
parser.add_argument('--dir', type=str, default='val_data_non_general')
cfg = parser.parse_args()

print('Configuration Parameters: ')
print(cfg)

VIDEO_FOLDER = os.path.join(project_dir, cfg.dir)


if __name__ == '__main__':
    dirlist = [os.path.join(VIDEO_FOLDER, x) for x in os.listdir(VIDEO_FOLDER) if os.path.isdir(os.path.join(VIDEO_FOLDER, x))]
    print('Evaluation random skipping algorithm ...')

    model = SimpleBoxMaskCNN(n_option=len(RATE_OPTIONS), n_stack=cfg.n_prev) if cfg.method == 'mask' else BoxNN(n_prev=cfg.n_prev, n_option=len(RATE_OPTIONS), top_n=16)
    model.load_state_dict(torch.load(cfg.model))
    model = model.cuda()
    model.eval()

    skip_ratio = []
    mae_list = []

    for d in tqdm(dirlist):
        print(d)
        test_data = VideoSamplerDataset(dirlist=[d], discount=1)
        test_data.reset()
        current_miss_counts = 0
        skip_accum = 0

        boxlists = []
        for (image, boxlist), (car_cnt, max_skip) in test_data:
            if len(boxlists) < cfg.n_prev:
                boxlists.append(boxlist)
                continue

            assert len(boxlists) == cfg.n_prev
            with torch.no_grad():
                x = _boxlist2tensor_channelstack(boxlists) if cfg.method == 'mask' else _boxembedding(boxlists)
                out = model(x.unsqueeze(0).cuda())
                _, predicted = torch.max(out.data, 1)
                predicted = predicted.cpu().numpy()[0]
                miss_counts, _ = test_data.skip_and_evaluate(RATE_OPTIONS[predicted])
                
                skip_accum += predicted
                current_miss_counts += miss_counts.sum()

            boxlists = []

        ratio = skip_accum / len(test_data)
        MAE = current_miss_counts / test_data.total_samples_number
        mae_list.append(MAE)
        skip_ratio.append(ratio)
        print(
            f'Random skipping: skipped_frames: {skip_accum} / {len(test_data)} = {ratio * 100:.3f}%, MAE: {MAE:.3f}')
    
    tag = f'ILp2f@modelnameis_{cfg.model.split("/")[-1]}'
    np.save(os.path.join(project_dir, 'result', f'{tag}.skip_ratio.npy'), skip_ratio)
    np.save(os.path.join(project_dir, 'result', f'{tag}.mae_list.npy'), mae_list)

    print(skip_ratio)
    print(mae_list)
    print(f'AVG Skipping Ratio: {np.array(skip_ratio).mean()}')
    print(f'AVG MAE : {np.array(mae_list).mean()}')