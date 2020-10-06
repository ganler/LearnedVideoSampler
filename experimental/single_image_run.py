import os
import sys

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)

from models.backbone import SamplerBackbone, boxlist2tensor
from models.experimental import ImagePolicyNet
from application.carcounter import CarCounter
import torch

import argparse
import torch.autograd.profiler as profiler
from application.carcounter.yolov3.utils import torch_utils
import cv2
import numpy as np
from utility.common import str2bool


def timed(caller, name=None):
    t1 = torch_utils.time_synchronized()
    r = caller()
    print(f'{caller if name is None else name} => time: {(torch_utils.time_synchronized() - t1) * 1000:.3f} ms.')
    return r

model_options = [
    'image-resnet-box-mask',
    'image-resnet',
]  # TODO: Support more models.

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imshow', type=str2bool, default=False)
    parser.add_argument('--model', type=int, default=0)
    parser.add_argument('--n_opt', type=int, default=16)
    return parser.parse_args()

if __name__ == '__main__':
    cfg = parse_args()
    imshow = cfg.imshow

    config = CarCounter.YOLOConfig()
    counter = CarCounter.CarCounter(config)

    raw_image = cv2.imread(os.path.join(project_dir, 'data', 'test.png'))
    inp = counter.process_image(raw_image)
    counter.predict(inp)  # Cold start
    pred = timed(lambda: counter.predict(inp), f'YOLOv3 Inference @ {inp.shape[2:]}\t')  # Bounding box list.
    after_box2list = timed(lambda: boxlist2tensor(pred, config.resolution).cuda(), 'BoxList Encoding\t\t\t\t')
    if imshow:
        vized = counter.viz(pred, raw_image, imshow=False)
        cv2.imwrite('example.png', vized)

    model = SamplerBackbone(cfg.n_opt) if cfg.model == 0 else ImagePolicyNet(cfg.n_opt)
    model.eval()
    model = model.cuda()

    # with torchprof.Profile(model, use_cuda=True) as prof:
    with torch.no_grad():
        model.eval()
        # with profiler.profile(record_shapes=True) as prof:
        #     with profiler.record_function("model_inference"):
        print(f'Your model: {model_options[cfg.model]}')
        if cfg.model == 0:
            model(inp, after_box2list)  # Cold start.
            timed(lambda: model(inp, after_box2list), 'Sampler Network Inference\t\t\t')
        else:
            model(inp)  # Cold start.
            timed(lambda: model(inp), 'Sampler Network Inference\t\t\t')

    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    # prof.export_chrome_trace("trace.json")
    # cv2.waitKey()

    
