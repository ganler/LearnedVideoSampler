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

from utility.videoloader import create_train_test_datasets
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--action_space', type=int, default=16)
cfg = parser.parse_args()

print('Configuration Parameters: ')
print(cfg)

RATE_OPTIONS = np.arange(cfg.action_space)
VIDEO_FOLDER = os.path.join(project_dir, 'data')
VIDEO_SUFFIX = '.avi'


if __name__ == '__main__':
    train_data, test_data = create_train_test_datasets(
        folder=VIDEO_FOLDER, suffix=VIDEO_SUFFIX, train_proportion=0.8)
    print(f'Test Samples Number: {len(test_data)}')
    print('Evaluation random skipping algorithm ...')
    test_data.reset()
    skip_accum = 0
    random_numerator = 0
    random_denominator = 1e-7
    for (image, boxlists), (car_cnt, max_skip) in tqdm(test_data):
        predicted = RATE_OPTIONS[random.randint(0, len(RATE_OPTIONS)-1)]
        res, _ = test_data.skip_and_evaluate(predicted)
        skip_accum += predicted
        random_numerator += sum(res)
        random_denominator += len(res)
    avg_accuracy = random_numerator / random_denominator
    print(
        f'Best skipping: skipped_frames: {skip_accum} / {len(test_data)} = {skip_accum / len(test_data) * 100:.3f}%, avg_accuracy: {avg_accuracy * 100:.3f} %')
