# Copyright (c) 2020 Ganler
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from utility.imcliploader import CASEvaluator
from models.experimental import ImagePolicyNet
import os
import torch
from utility.common import *
from utility.improcessing import *

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--tag', type=str, default=None)
parser.add_argument('--fetch_size', type=int, default=64)
parser.add_argument('--pretrained_backbone', type=str2bool, default=False)
parser.add_argument('--iou_pairing', type=float, default=None)
parser.add_argument('--mae', type=float, default=0.5)
parser.add_argument('--combinator', type=str, default='opticalflow', help='[opticalflow] [boxtensor] otherwise [concated image]')
cfg = parser.parse_args()
print(cfg)
if cfg.iou_pairing is not None:
    cfg.combinator = iou_pairing_skipper(conf_thresh=cfg.iou_pairing)
elif cfg.combinator == 'boxtensor':
    cfg.combinator = boxlist2tensor
else:
    cfg.combinator = opticalflow2tensor if cfg.combinator == 'opticalflow' else concat3channel2tensor

if __name__ == "__main__":
    evaluator = CASEvaluator(
        mae=cfg.mae,
        folder=os.path.join(project_dir, 'val_data_non_general'), 
        fetch_size=cfg.fetch_size, 
        combinator=cfg.combinator)

    model = None
    if cfg.iou_pairing is None:
        torch.manual_seed(1999)
        model = ImagePolicyNet(n_opt=2, pretrained=cfg.pretrained_backbone).cuda()
        if cfg.tag is not None:
            resdir = os.path.join(project_dir, 'trained')
            res = os.path.join(project_dir, 'trained', cfg.tag)
            print(f'Loading model from {res}')
            model.load_state_dict(torch.load(os.path.join(res, 'mlmodel.pth')))
        else:
            print(f'Using untrained model.')
        model.eval()
    
    mae_list, skip_ratio = evaluator.evaluate(model)

    mltag = f'ML-{cfg.tag}'
    nonmltag = f'iou_pairing-tresh-{cfg.iou_pairing}'
    tag = f'cas@{mltag if cfg.iou_pairing is None else nonmltag}-mae-{cfg.mae}'
    np.save(os.path.join(project_dir, 'result', f'{tag}.skip_ratio.npy'), skip_ratio)
    np.save(os.path.join(project_dir, 'result', f'{tag}.mae_list.npy'), mae_list)

    print(f'MAEs: {mae_list}')
    print(f'Skip Ratios: {skip_ratio}')
    print(f'AVG MAE: {mae_list.mean()}')
    print(f'AVG Skipping Ratio: {skip_ratio.mean()}')
