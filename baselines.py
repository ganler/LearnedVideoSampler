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
from utility.common import *
import multiprocessing as mp

parser = argparse.ArgumentParser()
parser.add_argument('--period', type=float, default=1.2)
parser.add_argument('--fixed_profile', type=float, default=0.05)
parser.add_argument('--required_mae', nargs='+', type=float, default=[None])
parser.add_argument('--action_space', type=int, default=32)
parser.add_argument('--fixed_skipping', type=int, default=None)
parser.add_argument('--src', type=str)
cfg = parser.parse_args()

RATE_OPTIONS = np.arange(cfg.action_space)  # Skipping size starts from 0.

def compute_mae(l, r):
    return np.abs(l - r).mean()

def get_best_skip_size(car_counts, required_mae):
    min_ = 0
    max_ = len(car_counts) - 1 # 0 ~ MAX
    # We assume that higher accuracy less skip size.
    while min_ < max_:
        s = (min_ + max_) // 2
        cp = car_counts.copy()
        index = 0
        while True:
            next_index = index + s + 1
            cp[index:min(next_index, cp.shape[0])] = car_counts[index]
            if next_index >= cp.shape[0]:
                break
            index = next_index
        mae = compute_mae(cp, car_counts)
        if mae < required_mae:
            min_ = s + 1
        else:
            max_ = s
    print(min_ - 1) 
    return min_ - 1

def eval(required_mae):
    skip_ratio = []
    mae_list = []
    test_data = VideoSamplerDataset(dirlist=[cfg.src], discount=1)
    test_data.reset()

    # MAE = \sum Abs Error / test_data.total_samples_number
    current_miss_counts = 0
    skip_accum = 0
    prof_n = int(cfg.fixed_profile * len(test_data))
    period_n = int(cfg.period * len(test_data))

    print(period_n, prof_n)

    predicted = None
    for (image, boxlists), (car_cnt, max_skip) in test_data:
        if cfg.fixed_skipping is not None:
            predicted = cfg.fixed_skipping
        elif cfg.fixed_profile is not None: # We need `cfg.fixed_profile` data to reset the profiling.
            if predicted is None:
                predicted = 0
                period_cnt = 0
                car_cnt_data = []
                first = True
            period_cnt += 1
            if first or period_cnt > period_n * prof_n:
                car_cnt_data.append(car_cnt)
                if len(car_cnt_data) == prof_n:
                    predicted = get_best_skip_size(np.array(car_cnt_data), required_mae)
                    period_cnt = 0
                    first = False
                    car_cnt_data = [] # Repeat...
        else:
            predicted = RATE_OPTIONS[random.randint(0, len(RATE_OPTIONS)-1)]

        miss_counts, _ = test_data.skip_and_evaluate(predicted)
        
        skip_accum += predicted
        current_miss_counts += miss_counts.sum()

    ratio = skip_accum / len(test_data)
    MAE = current_miss_counts / test_data.total_samples_number
    mae_list.append(MAE)
    skip_ratio.append(ratio)
    # print(f'baseline skipping: skipped_frames: {skip_accum} / {len(test_data)} = {ratio * 100:.3f}%, MAE: {MAE:.3f}')
    
    tag = cfg
    np.save(os.path.join(f'{cfg.src}.baseline.skip_ratio.npy'), skip_ratio)
    np.save(os.path.join(f'{cfg.src}.baseline.mae_list.npy'), mae_list)

    # print(skip_ratio)
    # print(mae_list)
    # print(cfg)
    # print(f'AVG Skipping Ratio: {np.array(skip_ratio).mean()}')
    # print(f'AVG MAE : {np.array(mae_list).mean()}')
    return np.array(mae_list).mean(), np.array(skip_ratio).mean()

if __name__ == "__main__":
    print(cfg)
    if cfg.required_mae[0] is not None:
        cfg.required_mae = sorted(cfg.required_mae)
    pool = mp.Pool(min(mp.cpu_count() - 1, len(cfg.required_mae)))
    results = pool.map(eval, cfg.required_mae)
    pool.close()

    mae = [m for m, _ in results]
    ratio = [r for _, r in results]
    print(f'KEY{cfg.fixed_profile}k = [')
    for t in cfg.required_mae:
        print(f"'mae={t:.2f}',")
    print(f']')
    print(f'VAL{cfg.fixed_profile}v = {{')
    print(f"'mae': {mae},")
    print(f"'ratio': {ratio},")
    print(f'}}')