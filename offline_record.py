import os
import sys

project_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_dir)

from application.carcounter import CarCounter
from tqdm import tqdm

import cv2
import numpy as np

config = CarCounter.YOLOConfig()
# config.resolution = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // fw, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // fh)
data_path = os.path.join(project_dir, 'data')

IMSHOW = False
VIDEO_FORMAT = '.mp4'
CLIP_FORMAT = '.avi'
SPLITTER = '___'
AIM_CLIP_FPS = 30
AIM_CLIP_FRAMES = 30

assert CLIP_FORMAT != VIDEO_FORMAT

if __name__ == '__main__':
    counter = CarCounter.CarCounter(config)

    processed_videos = [x.split(SPLITTER)[0] for x in os.listdir(data_path) if x.endswith(CLIP_FORMAT)]
    videos_to_process = [os.path.join(data_path, x) for x in os.listdir(data_path) if
                         x.endswith(VIDEO_FORMAT) and x not in processed_videos]

    for i, video in enumerate(videos_to_process):
        print(f'Video to be processed: #{i} ==> {video}')

    for video in tqdm(videos_to_process):
        cap = cv2.VideoCapture(video)

        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        clip_image_num = AIM_CLIP_FPS * AIM_CLIP_FRAMES
        clip_num = frames // clip_image_num

        last_sample_start = 0
        last_result = 0

        for clip_id in tqdm(range(clip_num)):
            output_prefix = f'{video.replace(VIDEO_FORMAT, SPLITTER)}{clip_id}'
            output_video = output_prefix + CLIP_FORMAT

            print(f'Writing :=> {output_video}')
            image_writer = cv2.VideoWriter(
                filename=output_video,
                fourcc=cv2.VideoWriter_fourcc(*'DIVX'),
                fps=AIM_CLIP_FPS,
                frameSize=(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

            clip_data = {
                'car_count': np.zeros(clip_image_num),
                'max_skip': np.zeros(clip_image_num),
                'boxlists': []
            }

            max_skip = 0
            for frame_id in range(clip_image_num):
                ret, frame = cap.read()
                if not ret:
                    print('ERROR: Got empty image in advance when decoding videos...')
                    exit(-1)
                image_writer.write(frame)

                inp = counter.process_image(frame)
                pred = counter.predict(inp)
                if IMSHOW:
                    counter.viz(pred, frame)

                car_count = len(pred[0])
                # print(f'CAR COUNT: {car_count}')
                if car_count != last_result and frame_id != 0:
                    assign_index_range = np.arange(last_sample_start, frame_id)
                    clip_data['max_skip'][assign_index_range] = frame_id - assign_index_range
                    # print(f'INDEX: {assign_index_range}')
                    # print(f"VALUE: {clip_data['max_skip'][assign_index_range]}")
                    skip = frame_id - last_sample_start
                    if skip > max_skip:
                        max_skip = skip
                        print(f'MAX SKIP in {video}:clip:{clip_id} updated to => {max_skip}')
                    last_sample_start = frame_id

                last_result = car_count
                clip_data['car_count'][frame_id] = car_count
                clip_data['boxlists'].append(pred[0].cpu())

            with open(f'{output_prefix}.npy', 'wb') as f:
                np.save(f, clip_data)
                assert len(clip_data['boxlists']) == clip_image_num
                print(f'VIDEO NAME => {video} :: Result written to => {f}')

            image_writer.release()

        cap.release()
    if IMSHOW:
        cv2.destroyAllWindows()
