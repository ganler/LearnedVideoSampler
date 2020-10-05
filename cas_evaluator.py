# Copyright (c) 2020 Ganler
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from utility.imcliploader import CASEvaluator
from models.experimental import ImagePolicyNet
import os
import torch

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--tag', type=str, default=None)
parser.add_argument('--pretrained_backbone', type=bool, default=False)
cfg = parser.parse_args()

if __name__ == "__main__":
    torch.manual_seed(0)
    model = ImagePolicyNet(n_opt=2, pretrained=cfg.pretrained_backbone).cuda()
    if cfg.tag is not None:
        resdir = os.path.join(project_dir, 'trained')
        res = os.path.join(project_dir, 'trained', cfg.tag)
        print(f'Loading model from {res}')
        model.load_state_dict(torch.load(os.path.join(res, 'mlmodel.pth')))
    else:
        print(f'Using untrained model.')
    model.eval()
    
    evaluator = CASEvaluator(folder=os.path.join(project_dir, 'val_data_non_general'), fetch_size=16)
    pred, skip = evaluator.evaluate(model)
    print(pred, skip)
    print(pred.mean(), skip.mean())
