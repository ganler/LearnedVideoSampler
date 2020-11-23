import os
import sys

import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
import argparse

project_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_dir)

from application.carcounter.CarCounter import YOLOConfig
from models.backbone import SimpleBoxMaskCNN, ResEmbedding
from utility.imcliploader import P2FDataset
from utility.common import *

config = YOLOConfig()
RATE_OPTIONS = range(32)
RESULT_FOLDER = os.path.join(project_dir, 'result')
MODEL_FOLDER = os.path.join(project_dir, 'trained')

parser = argparse.ArgumentParser()
# This means that training & evaluation data all come from `data` folder.
parser.add_argument('--use_fixed_valdata', type=str2bool, default=True)
parser.add_argument('--epoch', type=int, default=2)
parser.add_argument('--n_prev', type=int, default=8)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--fraction', type=float, default=1.0)
parser.add_argument('--frozen', type=str2bool, default=True)
parser.add_argument('--method', type=str, default='image')
cfg = parser.parse_args()
print(cfg)

if __name__ == '__main__':
    model, best_model = None, None
    if cfg.method == 'mask':
        model = SimpleBoxMaskCNN(n_option=len(RATE_OPTIONS), n_stack=cfg.n_prev)
        best_model = SimpleBoxMaskCNN(n_option=len(RATE_OPTIONS), n_stack=cfg.n_prev)
    elif cfg.method == 'image':
        model = ResEmbedding(n_prev=cfg.n_prev, n_opt=len(RATE_OPTIONS), frozen=cfg.frozen)
        best_model = ResEmbedding(n_prev=cfg.n_prev, n_opt=len(RATE_OPTIONS), frozen=cfg.frozen)
    else:
        raise Exception('please select a valid method')

    model = model.cuda()
    best_model = best_model.cuda()

    loss_func = SmoothCrossEntropyLoss(smoothing=0.1)
    optimizer = torch.optim.RMSprop(model.parameters())

    train, test = None, None
    if not cfg.use_fixed_valdata:
        dataset = P2FDataset(os.path.join(project_dir, 'data'), prev_n=cfg.n_prev, options=RATE_OPTIONS, method=cfg.method, fraction=cfg.fraction)
        to_train = int(round(len(dataset) * 0.9))
        train, test = torch.utils.data.random_split(dataset, [to_train, len(dataset) - to_train])
    else:
        train = P2FDataset(os.path.join(project_dir, 'data'), prev_n=cfg.n_prev, options=RATE_OPTIONS, method=cfg.method, fraction=cfg.fraction)
        test = P2FDataset(os.path.join(project_dir, 'val_data_non_general'), prev_n=cfg.n_prev, options=RATE_OPTIONS, method=cfg.method)

    train_loader = DataLoader(dataset=train, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(dataset=test, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    records = {
        'loss': [],
        'val_distance': []
    }

    print(f'Training Samples: {len(train_loader) * cfg.batch_size}, Test Samples: {len(test_loader) * cfg.batch_size}')

    check_size = len(train_loader) // 4
    best_val_accuracy = 0

    for epoch in range(cfg.epoch):
        running_loss = 0.0
        with trange(len(train_loader)) as t:
            for i, data in zip(t, train_loader):
                x, y = data
                out = model(x.cuda())
                loss = loss_func(out, torch.LongTensor(y).cuda())
                running_loss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if (i + 1) % (check_size) == 0:
                    # Postfix will be displayed on the right,
                    # formatted automatically based on argument's datatype
                    records['loss'].append(running_loss / check_size)
                    val_distance = 0
                    opts = np.array(RATE_OPTIONS)
                    for x, y in test_loader:
                        with torch.no_grad():
                            out = model(x.cuda())
                            _, predicted = torch.max(out.data, 1)
                            val_distance += np.abs(opts[predicted.cpu().numpy()] - opts[y]).sum()
                    val_distance /= len(test)
                    records['val_distance'].append(val_distance)
                    t.set_postfix(
                        epoch=epoch+1, 
                        loss=running_loss / check_size,
                        label_distance=val_distance
                        )
                    if val_distance < best_val_accuracy:
                        best_model.load_state_dict(model.state_dict())
                    running_loss = 0.0

    record_sign = f'IL-{datetime.now().isoformat().split(".")[0]}-method{cfg.method}-epoch-{cfg.epoch}-clip-{len(train_loader)}'
    np.save(f'{RESULT_FOLDER}/{record_sign}.npy', records)
    torch.save(
        model.state_dict(),
        f'{MODEL_FOLDER}/{record_sign}.pth')