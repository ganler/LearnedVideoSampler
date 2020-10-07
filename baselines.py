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
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--action_space', type=int, default=64)
parser.add_argument('--fixed_skipping', type=int, default=None)
parser.add_argument('--mae', type=float, default=0.5)
parser.add_argument('--dir', type=str, default='val_data_non_general')
cfg = parser.parse_args()

print('Configuration Parameters: ')
print(cfg)

RATE_OPTIONS = np.arange(cfg.action_space)  # Skipping size starts from 0.
VIDEO_FOLDER = os.path.join(project_dir, cfg.dir)


if __name__ == '__main__':
    dirlist = [os.path.join(VIDEO_FOLDER, x) for x in os.listdir(VIDEO_FOLDER) if os.path.isdir(os.path.join(VIDEO_FOLDER, x))]
    print('Evaluation random skipping algorithm ...')

    skip_ratio = []
    mae_list = []

    for d in tqdm(dirlist):
        print(d)
        test_data = VideoSamplerDataset(dirlist=[d], discount=1)
        test_data.reset()

        # MAE = \sum Abs Error / test_data.total_samples_number
        max_tolerant_miss_counts = int(round(test_data.total_samples_number * cfg.mae))
        current_miss_counts = 0
        skip_accum = 0

        for (image, boxlists), (car_cnt, max_skip) in test_data:
            predicted = None
            miss_counts = None
            if current_miss_counts < max_tolerant_miss_counts:
                predicted = RATE_OPTIONS[random.randint(0, len(RATE_OPTIONS)-1)] if cfg.fixed_skipping is None else cfg.fixed_skipping
                miss_counts, _ = test_data.skip_and_evaluate(predicted)
            else:
                predicted = 0
                miss_counts, _ = test_data.skip_and_evaluate(predicted)
                assert miss_counts.sum() == 0
            
            skip_accum += predicted
            current_miss_counts += miss_counts.sum()

        ratio = skip_accum / len(test_data)
        MAE = current_miss_counts / test_data.total_samples_number
        mae_list.append(MAE)
        skip_ratio.append(ratio)
        print(
            f'Random skipping: skipped_frames: {skip_accum} / {len(test_data)} = {ratio * 100:.3f}%, MAE: {MAE:.3f}')
    
    tag = f'fixed_skipping@{cfg.fixed_skipping}' if cfg.fixed_skipping is not None else f'random_skip@{cfg.action_space}'
    np.save(os.path.join(project_dir, 'result', f'{tag}.skip_ratio.npy'), skip_ratio)
    np.save(os.path.join(project_dir, 'result', f'{tag}.mae_list.npy'), mae_list)

    print(skip_ratio, mae_list)
    print(f'AVG Skipping Ratio: {np.array(skip_ratio).mean()}')
    print(f'AVG MAE : {np.array(mae_list).mean()}')