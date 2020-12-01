# Copyright (c) 2020 Ganler
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from utility.imcliploader import CASEvaluator
from models.backbone import ImagePolicyNet
import os
import torch
from utility.common import *
from utility.improcessing import *
import multiprocessing as mp

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--npy', type=str)
parser.add_argument('--tag', type=str, default=None)
parser.add_argument('--fetch_size', type=int, default=16)
parser.add_argument('--best', type=str2bool, default=False)
parser.add_argument('--iou_pairing', nargs='+', type=float, default=[None])
parser.add_argument('--max_diff', type=int, default=2)
parser.add_argument('--combinator', type=str, default='opticalflow', help='[opticalflow] [boxtensor] otherwise [concated image]')
cfg = parser.parse_args()

def eval(thresh):
    evaluator = CASEvaluator(
        path=os.path.join(project_dir, cfg.npy), 
        fetch_size=cfg.fetch_size,
        max_diff=cfg.max_diff, 
        combinator=iou_pairing_skipper(conf_thresh=thresh),
        best=cfg.best)

    model = None
    # if cfg.iou_pairing is None:
    #     torch.manual_seed(1999)
    #     model = ImagePolicyNet(n_opt=2, pretrained=cfg.pretrained_backbone).cuda()
    #     model = model.cuda()
    #     if cfg.tag is not None:
    #         resdir = os.path.join(project_dir, 'trained')
    #         res = os.path.join(project_dir, 'trained', cfg.tag)
    #         print(f'Loading model from {res}')
    #         model.load_state_dict(torch.load(os.path.join(res, 'mlmodel.pth')))
    #     else:
    #         print(f'Using untrained model.')
    #     model.eval()
    
    mae_list, skip_ratio = evaluator.evaluate(model)
    np.save(os.path.join(project_dir, f'{cfg.npy}.skip_ratio.npy'), skip_ratio)
    np.save(os.path.join(project_dir, f'{cfg.npy}.mae_list.npy'), mae_list)

    # print(f'MAEs: {mae_list}')
    # print(f'Skip Ratios: {skip_ratio}')
    # print(f'AVG MAE: {mae_list.mean()}')
    # print(f'AVG Skipping Ratio: {skip_ratio.mean()}')
    return mae_list.mean(), skip_ratio.mean()

if __name__ == "__main__":
    print(cfg)
    if cfg.iou_pairing[0] is not None:
        cfg.iou_pairing = sorted(cfg.iou_pairing)
    pool = mp.Pool(min(mp.cpu_count() - 1, len(cfg.iou_pairing)))
    results = pool.map(eval, cfg.iou_pairing)
    pool.close()

    mae = [m for m, _ in results]
    ratio = [r for _, r in results]
    print(f'betterpair{cfg.fetch_size}k = [')
    for t in cfg.iou_pairing:
        print(f"'thresh={t:.2f}',")
    print(f']')
    print(f'betterpair{cfg.fetch_size}v = {{')
    print(f"'mae': {mae},")
    print(f"'ratio': {ratio},")
    print(f'}}')
