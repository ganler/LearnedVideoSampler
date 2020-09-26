# Copyright (c) 2020 Ganler
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import os
import sys

project_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_dir)

from application.carcounter import CarCounter
from tqdm import tqdm

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
data_path = os.path.join(project_dir, 'data')

IMSHOW = False
VIDEO_FORMAT = '.mp4'
AIM_CLIP_FRAMES = 900
CLIP_FOLDER_SUFFIX = '__clip'

if __name__ == '__main__':
    counter = CarCounter.CarCounter(config)  # DNN.

    '''
    input name: 'xx.mp4'
    output clip folder: 'xx_clip__${id}'
    '''
    processed_videos = [x.replace(CLIP_FOLDER_SUFFIX, '') for x in os.listdir(data_path) if CLIP_FOLDER_SUFFIX in x]
    videos_to_process = [os.path.join(data_path, x) for x in os.listdir(data_path) if
                         x.endswith(VIDEO_FORMAT) and x.split('.')[0] not in processed_videos]

    for i, video in enumerate(videos_to_process):
        print(f'Video to be processed: #{i} ==> {video}')

    for video in tqdm(videos_to_process):
        cap = cv2.VideoCapture(video)

        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        clip_image_num = AIM_CLIP_FRAMES
        clip_num = frames // clip_image_num

        for clip_id in tqdm(range(clip_num)):
            output_folder = f'{video.replace(VIDEO_FORMAT, CLIP_FOLDER_SUFFIX)}{clip_id}'
            output_folder = os.path.join(data_path, output_folder)
            os.mkdir(output_folder)

            try:
                clip_data = {
                    'car_count': np.zeros(clip_image_num),
                    'max_skip': np.zeros(clip_image_num),
                    'boxlists': []
                }

                max_skip = 0
                last_sample_start = 0
                last_result = -1

                for frame_id in range(clip_image_num):
                    ret, frame = cap.read()
                    if not ret:
                        print('ERROR: Got empty image in advance when decoding videos...')
                        exit(-1)

                    frame = cv2.resize(frame, dsize=config.resolution)
                    cv2.imwrite(os.path.join(output_folder, f'{frame_id}.jpg'), frame)

                    inp = counter.process_image(frame)
                    pred = counter.predict(inp)
                    if IMSHOW:
                        counter.viz(pred, frame)

                    car_count = len(pred[0])
                    if car_count != last_result and frame_id != 0:
                        assign_index_range = np.arange(last_sample_start, frame_id)
                        clip_data['max_skip'][assign_index_range] = frame_id - assign_index_range
                        skip = frame_id - last_sample_start
                        if skip > max_skip:
                            max_skip = skip
                            print(f'MAX SKIP in {video}:clip:{clip_id} updated to => {max_skip}')
                        last_sample_start = frame_id

                    last_result = car_count
                    clip_data['car_count'][frame_id] = car_count
                    clip_data['boxlists'].append(pred[0].cpu())

                with open(f'{os.path.join(output_folder, "result")}.npy', 'wb') as f:
                    np.save(f, clip_data)
                    assert len(clip_data['boxlists']) == clip_image_num
                    print(f'VIDEO NAME => {video} :: Result written to => {f}')

            except Exception as e:
                print(f'Failed to process all data aimed to be in {output_folder}')
                os.rmdir(output_folder)
                raise e

        cap.release()
    if IMSHOW:
        cv2.destroyAllWindows()
