import os
import sys

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from datetime import datetime
import random
from torch.utils.data import DataLoader, random_split
import argparse

project_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_dir)

from application.carcounter.CarCounter import YOLOConfig, preprocess_image
from models.backbone import SimpleBoxMaskCNN
from utility.imcliploader import P2FDataset
from utility.common import *

config = YOLOConfig()
RATE_OPTIONS = [0, 2, 4, 8, 12, 16]
RESULT_FOLDER = os.path.join(project_dir, 'result')
MODEL_FOLDER = os.path.join(project_dir, 'trained')

parser = argparse.ArgumentParser()
# This means that training & evaluation data all come from `data` folder.
parser.add_argument('--use_fixed_valdata', type=str2bool, default=True)
parser.add_argument('--epoch', type=int, default=2)
parser.add_argument('--n_prev', type=int, default=8)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--fraction', type=float, default=1.0)
cfg = parser.parse_args()
print(cfg)

if __name__ == '__main__':
    model = SimpleBoxMaskCNN(n_option=len(RATE_OPTIONS), n_stack=cfg.n_prev).cuda()
    best_model = SimpleBoxMaskCNN(n_option=len(RATE_OPTIONS), n_stack=cfg.n_prev).cuda()
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3, momentum=0.2)

    train, test = None, None
    if not cfg.use_fixed_valdata:
        dataset = P2FDataset(os.path.join(project_dir, 'data'), options=RATE_OPTIONS, fraction=cfg.fraction)
        to_train = int(round(len(dataset) * 0.9))
        train, test = torch.utils.data.random_split(dataset, [to_train, len(dataset) - to_train])
    else:
        train = P2FDataset(os.path.join(project_dir, 'data'), options=RATE_OPTIONS, fraction=cfg.fraction)
        test = P2FDataset(os.path.join(project_dir, 'val_data_non_general'), options=RATE_OPTIONS)

    train_loader = DataLoader(dataset=train, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.batch_size)
    test_loader = DataLoader(dataset=test, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.batch_size)

    records = {
        'loss': [],
        'skipped_frames': [],
        'mae': []
    }

    print(f'Training Samples: {len(train_loader)}, Test Samples: {len(test_loader)}')

    check_size = len(train_loader) / 4
    loss_record = []
    precision_record = []
    best_val_accuracy = 0

    for epoch in range(cfg.epoch):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            x, y = data
            out = model(x.cuda())
            loss = loss_func(out, torch.LongTensor(y).cuda())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if (i + 1) % (check_size) == 0:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / check_size))
                loss_record.append(running_loss / check_size)
                running_loss = 0.0
                n_right = 0
                for x, y in tqdm(test_loader, 0):
                    with torch.no_grad():
                        out = model(x.cuda())
                        _, predicted = torch.max(out.data, 1)
                        n_right += (predicted.cpu().numpy() == np.array(y)).sum()
                acc = n_right / len(test)
                print(f'Validation accuracy: {acc * 100:.2f} %')
                precision_record.append(acc)
                if acc > best_val_accuracy:
                    best_model.load_state_dict(model.state_dict())

    record_sign = f'IL-{datetime.now().isoformat().split(".")[0]}-epoch-{cfg.epoch}-clip-{len(train_loader)}'
    np.save(f'{RESULT_FOLDER}/{record_sign}.npy', records)
    torch.save(
        model.state_dict(),
        f'{MODEL_FOLDER}/{record_sign}.pth')