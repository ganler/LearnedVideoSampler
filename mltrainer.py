import os
import sys

project_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_dir)

from models.backbone import SamplerBackbone, encode_box
import cv2

rate_option = [1, 2, 4, 8, 12, 16, 32, 64, 128]

model = SamplerBackbone(len(rate_option)).cuda()

from application.carcounter import CarCounter
import cv2
import numpy as np

imshow = False

config = CarCounter.YOLOConfig()
config.resolution = (608, 352)
counter = CarCounter.CarCounter(config)

data_path = os.path.join(project_dir, 'data')

videos_to_process = [os.path.join(data_path, x) for x in os.listdir(data_path) if not x.endswith('.npy')]

print(f'Video to be processed: {videos_to_process}')

for video in videos_to_process:
    cap = cv2.VideoCapture(video)

    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    data = np.zeros(frames)

    print(f'VIDEO NAME => {video} | FRAME COUNT => {data.shape[0]}')
    print(config)

    c = 0
    stack = []
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        inp = counter.process_image(frame)
        pred = counter.predict(inp)
        if imshow:
            counter.viz(pred, frame)
        data[c] = len(pred[0])
        stack.append(pred[0])
        c += 1

        if c % 3 == 0:
            box_embedding = encode_box(stack).unsqueeze(0).cuda()
            model(inp, box_embedding)
            stack = []

    cap.release()

if imshow:
    cv2.destroyAllWindows()