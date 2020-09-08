import os
import sys
from tqdm import tqdm
import numpy as np

project_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_dir)

from utility.videoloader import create_train_test_datasets

EPOCH = 3
BATCH_SIZE = 128
FACTOR = 4
RATE_OPTIONS = np.arange(16)
VIDEO_FOLDER = os.path.join(project_dir, 'data')
LOSS_RECORD_DUR = BATCH_SIZE * 32
VIDEO_SUFFIX = '.avi'
PRETRAINED_PATH = None


if __name__ == '__main__':
    train_data, test_data = create_train_test_datasets(
        folder=VIDEO_FOLDER, suffix=VIDEO_SUFFIX, train_proportion=0.8)
    print(f'Test Samples Number: {len(test_data)}')
    print('Evaluation best skipping algorithm ...')
    test_data.reset()
    skip_accum = 0
    random_numerator = 0
    random_denominator = 1e-7
    for (image, boxlists), (car_cnt, max_skip) in tqdm(test_data):
        predicted = int(max_skip)
        res, _ = test_data.skip_and_evaluate(predicted)
        skip_accum += predicted
        random_numerator += sum(res)
        random_denominator += len(res)
    avg_accuracy = random_numerator / random_denominator
    print(
        f'Best skipping: skipped_frames: {skip_accum} / {len(test_data)}, avg_accuracy: {avg_accuracy * 100:.3f} %')