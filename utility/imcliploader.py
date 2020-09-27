# Copyright (c) 2020 Ganler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

'''
Dataloader in PyTorch Style.
* Image folder => video clip.
* This is used for imitation learning/simple machine learning.
'''

from random import sample
import torch
import numpy as np
from torch.utils.data import Dataset
import os
import random
from typing import List, NamedTuple
from collections import namedtuple
from itertools import combinations
from .improcessing import opticalflow, totensor
import cv2
from tqdm import tqdm

Entry = namedtuple('Entry', ('left', 'right', 'label'))


'''
Doing NOW:
> Hard label.
> Optical flow.
> Imitation learning.
==========================================
TODO enhancements:
0: Soft label.
'''


class CAPDataset(Dataset):
    def __init__(self, clip_home, outlier_size=10, sample_rate=0.2):
        self.entries: List[Entry] = []

        # Create TemporalSets.
        folderlist = [x for x in os.listdir(clip_home) if os.path.isdir(os.path.join(clip_home, x))]
        print('Loading data...')
        for folder in tqdm(folderlist):
            folder = os.path.join(clip_home, folder)
            raw_labels = np.load(os.path.join(folder, 'result.npy'), allow_pickle=True).item()
            max_skip = np.array(raw_labels['max_skip'], dtype=np.int32)
            car_count = raw_labels['car_count']
            index = 0
            while index < len(max_skip):
                end = index + max_skip[index]
                if max_skip[index] == 0:
                    raise Exception('fuck')
                if end + outlier_size > len(max_skip):
                    break
                if max_skip[index] < 3:  # No skip? No set!
                    index = end
                    continue
                interior_range = (index, end)
                border_outlier = [int(car_count[ind])
                                  for ind in range(end, end + outlier_size) if car_count[ind] == car_count[0]]
                
                index = end
                if len(border_outlier) == 0:
                    continue

                pos = [x for x in combinations(interior_range, 2)]
                neg = [(x, y) for x in interior_range for y in border_outlier]

                sample_num = int(max(1, min(len(pos), len(neg)) * sample_rate))
                pos = random.sample(pos, sample_num)
                neg = random.sample(neg, sample_num)

                for x in pos:
                    l = os.path.join(folder, f'{x[0]}.jpg')
                    r = os.path.join(folder, f'{x[1]}.jpg')
                    self.entries.append(Entry(l, r, True))

                for x in neg:
                    l = os.path.join(folder, f'{x[0]}.jpg')
                    r = os.path.join(folder, f'{x[1]}.jpg')
                    self.entries.append(Entry(l, r, False))

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int):
        l, r, label = self.entries[index]
        l = cv2.imread(l)
        r = cv2.imread(r)  # NOTE: OpenCV mat's shape means: Height, Width, Channel.
        print(l.shape)
        flow = opticalflow(l, r)
        im = np.zeros((*l.shape[:2], 3))
        im[:, :, :2] += flow
        return totensor(im, wh=l.shape[1::-1]), label
        