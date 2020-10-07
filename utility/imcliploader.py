# Copyright (c) 2020 Ganler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

'''
Dataloader in PyTorch Style.
* Image folder => video clip.
* This is used for imitation learning/simple machine learning.
'''

from dataclasses import dataclass
from random import sample
import torch
import numpy as np
from torch.utils.data import Dataset
import os
import random
from typing import List, NamedTuple
from collections import namedtuple
from itertools import combinations
from .improcessing import opticalflow2tensor, concat3channel2tensor, boxlist2tensor
import cv2
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator

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
    def __init__(self, clip_home, outlier_size=30, fraction=1.0, sample_rate=0.25, combinator=opticalflow2tensor):
        self.entries: List[Entry] = []
        self.combinator = combinator

        # Create TemporalSets.
        folderlist = [x for x in os.listdir(clip_home) if os.path.isdir(os.path.join(clip_home, x))]
        folderlist = folderlist[:int(round(len(folderlist) * fraction))]
        print('Loading data...')
        for folder in tqdm(folderlist):
            folder = os.path.join(clip_home, folder)
            raw_labels = np.load(os.path.join(folder, 'result.npy'), allow_pickle=True).item()
            max_skip = np.array(raw_labels['max_skip'], dtype=np.int32)
            car_count = raw_labels['car_count']
            boxlists = raw_labels['boxlists']
            index = 0
            while index < len(max_skip):
                end = index + max_skip[index]
                if end + outlier_size > len(max_skip):
                    break
                if max_skip[index] < 3:  # No skip? No set!
                    index = end
                    continue

                # Positive: a * (a - 1) / 2
                # Negative: ab
                # a = 2 b + 1
                interior_range = np.arange(index, end)
                border_outlier = np.arange(end, end + outlier_size)
                border_outlier = border_outlier[np.where(car_count[end:end + outlier_size] == car_count[index])[0]]

                a = max(1, int(round(len(interior_range) * sample_rate)))
                b = max(1, int(round(len(border_outlier) * sample_rate)))

                if a > 2 * b + 1:
                    a = 2 * b + 1
                
                if b > (a - 1) // 2:
                    b = (a - 1) // 2                
                
                index = end
                if a == 0 or b == 0 or len(border_outlier) == 0:
                    continue

                interior_range = interior_range[np.sort(np.random.choice(len(interior_range), a))]
                border_outlier = border_outlier[np.sort(np.random.choice(len(border_outlier), b))]

                assert border_outlier.min() > interior_range.max()

                pos = [x for x in combinations(interior_range, 2)]
                neg = np.array(np.meshgrid(interior_range, border_outlier)).T.reshape(-1, 2)

                for x in pos:
                    if combinator is boxlist2tensor:
                        l = boxlists[x[0]]
                        r = boxlists[x[1]]
                    else:
                        l = os.path.join(folder, f'{x[0]}.jpg')
                        r = os.path.join(folder, f'{x[1]}.jpg')
                    self.entries.append(Entry(l, r, 1))

                for x in neg:
                    if combinator is boxlist2tensor:
                        l = boxlists[x[0]]
                        r = boxlists[x[1]]
                    else:
                        l = os.path.join(folder, f'{x[0]}.jpg')
                        r = os.path.join(folder, f'{x[1]}.jpg')
                    self.entries.append(Entry(l, r, 0)) # 0 => False

    def __len__(self) -> int:
        return len(self.entries)
    
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

    def __getitem__(self, index: int):
        l, r, label = self.entries[index]
        if self.combinator is not boxlist2tensor:
            l = cv2.imread(l)
            r = cv2.imread(r)  # NOTE: OpenCV mat's shape means: Height, Width, Channel.
        x = self.combinator(l, r)
        # if x.sum() > 0:
        #     from .improcessing import _boxlist2tensor
        #     print(torch.nonzero(_boxlist2tensor(l), as_tuple=True))
        #     raise Exception('...')

        return x, label

ClipElement = namedtuple('ClipElement', ('path', 'max_size', 'labels'))


class CASEvaluator:
    def __init__(self, folder, fetch_size=32, combinator=opticalflow2tensor):
        self.clips: List[ClipElement] = []
        self.fetch_size = fetch_size
        self.combinator = combinator
        for f in os.listdir(folder):
            if os.path.isdir(os.path.join(folder, f)):
                path = os.path.join(folder, f)
                labels = np.load(os.path.join(path, 'result.npy'), allow_pickle=True).item()['car_count']
                max_size = len(labels)
                self.clips.append(ClipElement(path=path, max_size=max_size, labels=labels))

    def evaluate(self, model, mae_bound=None):
        ret_mae = []
        ret_skip = []
        for cc in tqdm(self.clips):
            c = cc
            predicted = np.ones(c.max_size) * -1 # -1 is a flag.
            
            def fetch_one(index):
                return cv2.imread(os.path.join(c.path, f'{index}.jpg')) if self.combinator is not boxlist2tensor else 1 # TODO.

            def CAS(begin, end, skipped_size):
                # [begin] [end]
                if end == begin:
                    return
                # [begin] [?] [end]
                # [begin] [?] [?] [end]
                if end - begin <= 2:
                    predicted[begin] = c.labels[begin]
                    predicted[end - 1] = c.labels[end - 1]
                    return
                
                if end - begin > 2:
                    lc = c.labels[begin]
                    rc = c.labels[end - 1]
                    predicted[begin] = lc
                    predicted[end - 1] = rc
                    if lc == rc and torch.max(
                        model.forward(
                            self.combinator(
                                fetch_one(begin), fetch_one(end - 1), batch_dim=True
                                ).cuda()).data, 1)[1].cpu().numpy()[0] == True:
                        predicted[begin+1 : end-1] = lc
                        skipped_size += (end - begin - 1)
                    else:
                        partition = (end + begin) // 2
                        CAS(begin + 1, partition, skipped_size)
                        CAS(partition, end - 1, skipped_size)

            begin_ = 0
            end_ = 0
            skipped_size = np.array([0])

            while end_ != c.max_size:
                end_ = min(begin_ + self.fetch_size, c.max_size)
                CAS(begin_, end_, skipped_size)
                begin_ = end_

            bugs = (predicted < 0).nonzero()[0]
            if len(bugs) != 0:
                print(predicted)
                print(bugs)
                raise Exception('Bugs occurred: There are unchecked frames...')

            predicted = np.array(predicted)
            ret_mae.append(np.abs(predicted - c.labels).mean())
            ret_skip.append(skipped_size[0] / len(c.labels))
        return np.array(ret_mae), np.array(ret_skip)
