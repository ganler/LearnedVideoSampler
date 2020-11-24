# Copyright (c) 2020 Ganler
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# FIXME
# >> v0: 0~622
# >> v1: 0~669
# >> v2: 0~328
# >> To be further processed.

import os
import sys

project_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_dir)

from utility.common import str2bool
from application.carcounter import CarCounter
from tqdm import tqdm
import argparse

import cv2
import numpy as np

config = CarCounter.YOLOConfig()

CLIP_FOLDER_SUFFIX = '__clip'

parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str)
parser.add_argument('--dst', type=str)
parser.add_argument('--imshow', type=str2bool, default=False)
parser.add_argument('--video_format', type=str, default='.mp4')
parser.add_argument('--contains', type=str, default='')
parser.add_argument('--quiet', type=str2bool, default=True)
cfg = parser.parse_args()

if __name__ == '__main__':
    counter = CarCounter.CarCounter(config)  # DNN.

    '''
    input name: 'xx.mp4'
    output clip folder: 'xx_clip__${id}'
    '''
    video = cfg.src
    cap = cv2.VideoCapture(video)

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    clip_data = {
        'car_count': np.zeros(n_frames),
        'max_skip': np.zeros(n_frames),
        'boxlists': [],
        'resolution': config.resolution,
        'src_path': video
    }

    max_skip = 0
    last_sample_start = 0
    last_result = -1

    for frame_id in tqdm(range(n_frames)):
        ret, frame = cap.read()
        if not ret:
            print('ERROR: Got empty image in advance when decoding videos...')
            exit(-1)

        frame = cv2.resize(frame, dsize=config.resolution)

        inp = counter.process_image(frame)
        pred = counter.predict(inp)
        if cfg.imshow:
            counter.viz(pred, frame)

        car_count = len(pred[0])
        if car_count != last_result and frame_id != 0 or frame_id == n_frames - 1:
            assign_index_range = np.arange(last_sample_start, frame_id)
            clip_data['max_skip'][assign_index_range] = frame_id - assign_index_range
            skip = frame_id - last_sample_start
            if skip > max_skip:
                max_skip = skip
                if not cfg.quiet:
                    print(f'MAX SKIP in {video} updated to => {max_skip}')
            if frame_id == n_frames - 1:
                clip_data['max_skip'][np.arange(last_sample_start, frame_id + 1)] += 1
            last_sample_start = frame_id

        last_result = car_count
        clip_data['car_count'][frame_id] = car_count
        clip_data['boxlists'].append(pred[0].cpu())

    with open(cfg.dst, 'wb') as f:
        np.save(f, clip_data)
        assert len(clip_data['boxlists']) == n_frames
        if not cfg.quiet:
            print(f'VIDEO NAME => {video} :: Result written to => {f}')

    cap.release()
    if cfg.imshow:
        cv2.destroyAllWindows()
