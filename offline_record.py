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

'''
Usage:  Convert videos into datasets.
Input:  Video List(in `data/*.mp4`)
Output: 
    - Folders with images[name=$(ID).jpg] where each of them represents a video clip.
    - In each clip folder, we have a 'result.npy' which is a dict containing results:
        > dict['car_count']: car count results by YOLO.
        > dict['max_skip']: maximum skip size of current image.
        > dict['boxlists']: bounding box data.
'''

config = CarCounter.YOLOConfig()

CLIP_FOLDER_SUFFIX = '__clip'

parser = argparse.ArgumentParser()
parser.add_argument('--imshow', type=str2bool, default=False)
parser.add_argument('--video_format', type=str, default='.mp4')
parser.add_argument('--clip_size', type=int, default=1800)
parser.add_argument('--dir', type=str, default='data')
parser.add_argument('--contains', type=str, default='')
parser.add_argument('--quiet', type=str2bool, default=True)
cfg = parser.parse_args()

data_path = os.path.join(project_dir, cfg.dir)

if __name__ == '__main__':
    counter = CarCounter.CarCounter(config)  # DNN.

    '''
    input name: 'xx.mp4'
    output clip folder: 'xx_clip__${id}'
    '''
    processed_videos = [x.replace(CLIP_FOLDER_SUFFIX, '') for x in os.listdir(data_path) if CLIP_FOLDER_SUFFIX in x]
    print(f'Found processed videos names: {processed_videos}')
    videos_to_process = [os.path.join(data_path, x) for x in os.listdir(data_path) if
                         x.endswith(cfg.video_format) and x.split('.')[0] not in processed_videos and cfg.contains in x]

    for i, video in enumerate(videos_to_process):
        print(f'Video to be processed: #{i} ==> {video}')

    for video in tqdm(videos_to_process):
        cap = cv2.VideoCapture(video)

        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        clip_image_num = cfg.clip_size
        clip_num = frames // clip_image_num

        for clip_id in tqdm(range(clip_num)):
            output_folder = f'{video.replace(cfg.video_format, CLIP_FOLDER_SUFFIX)}{clip_id}'
            output_folder = os.path.join(data_path, output_folder)
            os.mkdir(output_folder)

            try:
                clip_data = {
                    'car_count': np.zeros(clip_image_num),
                    'max_skip': np.zeros(clip_image_num),
                    'frame_ids': np.arange(clip_id * clip_image_num, (1 + clip_id) * clip_image_num),
                    'boxlists': [],
                    'resolution': config.resolution,
                    'src_path': video
                }

                max_skip = 0
                last_sample_start = 0
                last_result = -1

                for frame_id in tqdm(range(clip_image_num)):
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
                    if car_count != last_result and frame_id != 0 or frame_id == clip_image_num - 1:
                        assign_index_range = np.arange(last_sample_start, frame_id)
                        clip_data['max_skip'][assign_index_range] = frame_id - assign_index_range
                        skip = frame_id - last_sample_start
                        if skip > max_skip:
                            max_skip = skip
                            if not cfg.quiet:
                                print(f'MAX SKIP in {video}:clip:{clip_id} updated to => {max_skip}')
                        if frame_id == clip_image_num - 1:
                            clip_data['max_skip'][np.arange(last_sample_start, frame_id + 1)] += 1
                        last_sample_start = frame_id

                    last_result = car_count
                    clip_data['car_count'][frame_id] = car_count
                    clip_data['boxlists'].append(pred[0].cpu())

                with open(f'{os.path.join(output_folder, "result")}.npy', 'wb') as f:
                    np.save(f, clip_data)
                    assert len(clip_data['boxlists']) == clip_image_num
                    if not cfg.quiet:
                        print(f'VIDEO NAME => {video} :: Result written to => {f}')

            except Exception as e:
                print(f'Failed to process all data aimed to be in {output_folder}')
                os.rmdir(output_folder)
                raise e

        cap.release()
    if cfg.imshow:
        cv2.destroyAllWindows()
