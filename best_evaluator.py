import os
import sys
from tqdm import tqdm
import numpy as np

project_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_dir)

from utility.videoloader import VideoSamplerDataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--action_space', type=int, default=16)
cfg = parser.parse_args()

print('Configuration Parameters: ')
print(cfg)

RATE_OPTIONS = np.arange(cfg.action_space)
VIDEO_FOLDER = os.path.join(project_dir, 'val_data_non_general')

if __name__ == '__main__':
    dirlist = [os.path.join(VIDEO_FOLDER, x) for x in os.listdir(VIDEO_FOLDER) if os.path.isdir(os.path.join(VIDEO_FOLDER, x))]
    print('Evaluating best skipping algorithm ...')

    skip_ratio = []
    acc_list = []
    for d in dirlist:
        test_data = VideoSamplerDataset(dirlist=[d], discount=1)
        test_data.reset()
        skip_accum = 0
        random_numerator = 0
        random_denominator = 1e-7
        for (image, boxlists), (car_cnt, max_skip) in tqdm(test_data):
            predicted = min(int(max_skip), RATE_OPTIONS[-1])
            res, _ = test_data.skip_and_evaluate(predicted)
            skip_accum += predicted
            random_numerator += sum(res)
            random_denominator += len(res)
        avg_accuracy = random_numerator / random_denominator
        acc_list.append(avg_accuracy)
        skip_ratio.append(skip_accum / len(test_data))
        print(
            f'Best skipping: skipped_frames: {skip_accum} / {len(test_data)} = {skip_accum / len(test_data) * 100:.3f}%, avg_accuracy: {avg_accuracy * 100:.3f} %')

    print(skip_ratio, acc_list)
    print(np.array(skip_ratio).mean(), np.array(acc_list).mean())