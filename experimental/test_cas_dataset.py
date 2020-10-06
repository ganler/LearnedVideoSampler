# Copyright (c) 2020 Ganler
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# video 623

from itertools import combinations
import sys
import torch
import os
from torch.utils.data import DataLoader, random_split
from torch import nn
from tqdm import tqdm
from datetime import datetime
import numpy as np

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)

from utility.improcessing import concat3channel2tensor, opticalflow2tensor
from models.experimental import ImagePolicyNet
from utility.imcliploader import CAPDataset
from utility.common import str2bool
import argparse

parser = argparse.ArgumentParser()
# This means that training & evaluation data all come from `data` folder.
parser.add_argument('--use_fixed_valdata', type=str2bool, default=True)
parser.add_argument('--epoch', type=int, default=3)
parser.add_argument('--fraction', type=float, default=1.0)
parser.add_argument('--combinator', type=str, default='opticalflow', help='[opticalflow] otherwise [concated image]')
parser.add_argument('--pretrained_backbone', type=str2bool, default=False)
cfg = parser.parse_args()
print(cfg)
cfg.combinator = opticalflow2tensor if cfg.combinator == 'opticalflow' else concat3channel2tensor

if __name__ == "__main__":
    model = ImagePolicyNet(n_opt=2, pretrained=cfg.pretrained_backbone).cuda()
    best_model = ImagePolicyNet(n_opt=2, pretrained=cfg.pretrained_backbone)

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)

    best_val_accuracy = 0
    train = None
    test = None

    if not cfg.use_fixed_valdata:
        dataset = CAPDataset(os.path.join(project_dir, 'data'), fraction=cfg.fraction, combinator=cfg.combinator)
        to_train = int(round(len(dataset) * 0.9))
        train, test = torch.utils.data.random_split(dataset, [to_train, len(dataset) - to_train])
    else:
        train = CAPDataset(os.path.join(project_dir, 'data'), fraction=cfg.fraction, combinator=cfg.combinator)
        test = CAPDataset(os.path.join(project_dir, 'val_data_non_general'), sample_rate=0.8, combinator=cfg.combinator)

    print(f'Lengths: TRAIN = {len(train)}, TEST = {len(test)}')

    train_loader = DataLoader(dataset=train, batch_size=64, shuffle=True, num_workers=16)
    test_loader = DataLoader(dataset=test, batch_size=64, shuffle=False, num_workers=16)

    loss_record = []
    precision_record = []
    for epoch in range(cfg.epoch):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            x, y = data
            out = model(x.cuda())
            loss = loss_func(out, torch.LongTensor(y).cuda())
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            check_size = int(len(train_loader) * 0.25)
            if (i + 1) % (check_size) == 0:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / check_size))
                loss_record.append(running_loss / check_size)
                running_loss = 0.0
                n_right = 0
                for x, y in tqdm(test_loader):
                    with torch.no_grad():
                        out = model(x.cuda())
                        _, predicted = torch.max(out.data, 1)
                        n_right += (predicted.cpu().numpy() == np.array(y)).sum()
                acc = n_right / len(test)
                print(f'Validation accuracy: {acc * 100:.2f} %')
                precision_record.append(acc)
                if acc > best_val_accuracy:
                    best_model.load_state_dict(model.state_dict())


    record_sign = f'{datetime.now().isoformat().split(".")[0]}-epoch-{cfg.epoch}-samples-{len(train)}'
    aim_dir = os.path.join(project_dir, 'trained', record_sign)
    os.mkdir(aim_dir)
    torch.save(
        best_model.state_dict(),
        os.path.join(aim_dir, 'mlmodel.pth'))
    np.save(os.path.join(aim_dir, 'loss.npy'), np.array(loss_record))
    np.save(os.path.join(aim_dir, 'train_acc.npy'), np.array(precision_record), )
