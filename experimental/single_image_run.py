import os
import sys

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)

from models.backbone import SamplerBackbone, boxlist2tensor
from application.carcounter import CarCounter
import torch

import torch.autograd.profiler as profiler
from application.carcounter.yolov3.utils import torch_utils
import cv2
import numpy as np

imshow = False


def timed(caller):
    t1 = torch_utils.time_synchronized()
    r = caller()
    print(f'{caller} => time: {(torch_utils.time_synchronized() - t1) * 1000} ms.')
    return r


config = CarCounter.YOLOConfig()
counter = CarCounter.CarCounter(config)

raw_image = cv2.imread(os.path.join(project_dir, 'data', 'test.png'))

# cv2.imshow(f'RawImage - {raw_image.shape}', raw_image)

inp = counter.process_image(raw_image)
# inp *= 0
pred = counter.predict(inp)  # Bounding box list.
# print(pred)
# counter.viz(pred, raw_image)

after_box2list = timed(lambda: boxlist2tensor(pred, config.resolution).cuda())
model = SamplerBackbone(16).cuda()

# with torchprof.Profile(model, use_cuda=True) as prof:
with torch.no_grad():
    model.eval()
    # with profiler.profile(record_shapes=True) as prof:
    #     with profiler.record_function("model_inference"):
    timed(lambda: model(inp, after_box2list))
# print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
# prof.export_chrome_trace("trace.json")
# cv2.waitKey()
#
# cv2.destroyAllWindows()
