import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from application.carcounter import CarCounter
import cv2
import numpy as np

video = 'data/video-clip-5min.mp4'
cap = cv2.VideoCapture(video)
config = CarCounter.YOLOConfig()

# config.resolution = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // fw, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // fh)
config.resolution = (608, 352)
counter = CarCounter.CarCounter(config)

frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
data = np.zeros(frames)

print(f'FRAME COUNT => {data.shape[0]}')
print(config)

c = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    pred = counter.predict(frame)
    counter.viz(pred, frame)
    data[c] = len(pred[0])
    c += 1

with open(f'{video}.npy', 'wb') as f:
    np.save(f, data)

cap.release()
cv2.destroyAllWindows()
