# Copyright (c) 2020 Ganler
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import sys
import os
from torch.utils.data import DataLoader, random_split

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)

from utility.imcliploader import CAPDataset

if __name__ == "__main__":
    dataset = CAPDataset(os.path.join(project_dir, 'data'))

    train_ratio = 0.9
    n_train = int(len(dataset) * train_ratio)
    train, test = random_split(dataset=dataset, lengths=[n_train, len(dataset) - n_train])

    print(f'Lengths: TRAIN = {len(train)}, TEST = {len(test)}')

    train_loader = DataLoader(dataset=train, batch_size=8, shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset=test, batch_size=8, shuffle=False, num_workers=4)

    for x, y in test_loader:
        print(x.shape, y)
