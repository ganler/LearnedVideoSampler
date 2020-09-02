import os
import sys

project_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_dir)

from models.backbone import SamplerBackbone, encode_box
from application.carcounter import CarCounter
from tqdm import tqdm

import cv2
import numpy as np

imshow = False

config = CarCounter.YOLOConfig()
counter = CarCounter.CarCounter(config)

# config.resolution = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // fw, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // fh)
data_path = os.path.join(project_dir, 'data')

processed_videos = [x.replace('.npy', '') for x in os.listdir(data_path) if x.endswith('.npy')]
videos_to_process = [os.path.join(data_path, x) for x in os.listdir(data_path) if
                     not x.endswith('.npy') and x not in processed_videos]

print(f'Video to be processed: {videos_to_process}')

for video in tqdm(videos_to_process):
    cap = cv2.VideoCapture(video)

    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    data_dict = {'count': np.zeros(frames), 'max_skip': np.zeros(frames), 'bbox': np.zeros((frames, 3, 64 * 5))}

    print(f'VIDEO NAME => {video} | FRAME COUNT => {data_dict["count"].shape[0]}')
    print(config)

    stack = [[], [], []]
    last_sample_start = 0
    last_result = -1
    for c in tqdm(range(frames)):
        ret, frame = cap.read()
        if not ret:
            break

        inp = counter.process_image(frame)
        pred = counter.predict(inp)
        if imshow:
            counter.viz(pred, frame)

        car_count = len(pred[0])
        if car_count != last_result:
            assign_index_range = np.arange(last_result, c+1)
            data_dict['max_skip'][assign_index_range] = c - assign_index_range
            last_result = car_count
        data_dict["count"][c] = car_count
        stack.pop(0)
        stack.append(pred[0])
        data_dict['bbox'][c] = encode_box(stack).numpy()

    with open(f'{video}.npy', 'wb') as f:
        np.save(f, data_dict)
        print(f'VIDEO NAME => {video} :: Result written to => {f}')

    cap.release()
if imshow:
    cv2.destroyAllWindows()
