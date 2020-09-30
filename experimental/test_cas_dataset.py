# Copyright (c) 2020 Ganler
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

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

from models.experimental import ImagePolicyNet

from utility.imcliploader import CAPDataset

EPOCH_N = 4
EVAL_MODE = 1
# 0: use val_data_non_general

if __name__ == "__main__":
    torch.manual_seed(0)
    model = ImagePolicyNet(n_opt=2).cuda()
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters())

    train = None
    test = None

    if EVAL_MODE == 1:
        dataset = CAPDataset(os.path.join(project_dir, 'data'))
        to_train = int(len(dataset) * 0.9)
        train, test = torch.utils.data.random_split(dataset, [to_train, len(dataset) - to_train])
    
    if EVAL_MODE == 0:
        train = CAPDataset(os.path.join(project_dir, 'data'))
        test = CAPDataset(os.path.join(project_dir, 'val_data_non_general'), sample_rate=0.8)

    print(f'Lengths: TRAIN = {len(train)}, TEST = {len(test)}')

    train_loader = DataLoader(dataset=train, batch_size=16, shuffle=True, num_workers=8)
    test_loader = DataLoader(dataset=test, batch_size=16, shuffle=False, num_workers=8)

    loss_record = []
    precision_record = []
    for epoch in range(EPOCH_N):
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
                print(f'Validation accuracy: {n_right / len(test) * 100:.2f} %')
                precision_record.append(n_right / len(test))


    record_sign = f'{datetime.now().isoformat().split(".")[0]}-epoch-{EPOCH_N}-samples-{len(train)}'
    aim_dir = os.path.join(project_dir, 'trained', record_sign)
    os.mkdir(aim_dir)
    torch.save(
        model.state_dict(),
        os.path.join(aim_dir, 'mlmodel.pth'))
    np.save(os.path.join(aim_dir, 'loss.npy'), np.array(loss_record))
    np.save(os.path.join(aim_dir, 'train_acc.npy'), np.array(precision_record), )
