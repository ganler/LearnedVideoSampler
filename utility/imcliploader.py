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
import random
from numpy.core.numerictypes import maximum_sctype
import torch
import numpy as np
from torch.utils.data import Dataset
import os
import random
from typing import List
from collections import namedtuple
from itertools import combinations
from threading import Lock
from .improcessing import *
from .improcessing import _boxlist2tensor_channelstack, _boxembedding
from .common import *
import cv2
import multiprocessing
from tqdm import tqdm

from prefetch_generator import BackgroundGenerator

Entry = namedtuple('Entry', ('left', 'right', 'label'))
FrameDescription = namedtuple('FrameDescription', ('cap_lock', 'index', 'resolution'))


'''
Doing NOW:
> Hard label.
> Optical flow.
> Imitation learning.
==========================================
TODO enhancements:
0: Soft label.
'''


class P2FDataset(Dataset):
    def __init__(self, clip_home, options, method='image', fraction=1.0, prev_n=8, factor=4):
        self.options = np.array(options, dtype=np.int32)
        self.max_skip_collection = []
        self.bbox_collection = []
        self.prev_n = prev_n
        self.sampling = len(options)
        self.factor = factor
        self.frames = []
        self.method = method
        self.video_kv = {}
        self.aligned_num = None

        # Create TemporalSets. # FIXME: Try one video 1st.
        folderlist = [x for x in os.listdir(clip_home) if os.path.isdir(os.path.join(clip_home, x)) and 'video0' in x]
        folderlist = folderlist[:int(round(len(folderlist) * fraction))]
        print('Loading data...')

        for folder in tqdm(folderlist):
            folder = os.path.join(clip_home, folder)
            raw_data = np.load(os.path.join(folder, 'result.npy'), allow_pickle=True).item()
            max_skip = np.array(raw_data['max_skip'], dtype=np.int32) - 1

            self.max_skip_collection.append(max_skip)
            if self.method != 'image':
                boxlists = raw_data['boxlists']
                self.bbox_collection.append(boxlists)
            else:
                self.video_kv[raw_data['src_path']] = {}
                self.frames.append({'reso': raw_data['resolution'], 'videokey': raw_data['src_path'], 'ids': raw_data['frame_ids']})

            if self.aligned_num is None:
                self.aligned_num = len(max_skip) - (prev_n - 1)
            assert len(max_skip) - (prev_n - 1) == self.aligned_num
        
        for k in self.video_kv:
            self.video_kv[k] = {'cap': cv2.VideoCapture(k), 'lk': Lock()}

    def __len__(self) -> int:
        return (self.aligned_num // self.sampling) * len(self.max_skip_collection)
    
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

    def __getitem__(self, index: int):
        n_sample_per_clip = self.aligned_num // self.sampling
        clip_index = index // n_sample_per_clip
        clip_size = len(self.max_skip_collection[clip_index])
        seg_index = index - clip_index * n_sample_per_clip
        begin = clip_size // n_sample_per_clip * seg_index
        end = begin + self.prev_n
        best_skip = int(self.max_skip_collection[clip_index][end-1])
        ans = np.nonzero((best_skip - self.options) >= 0)[0][-1]
        
        x = None
        if self.method == 'image':
            frame_meta = self.frames[clip_index]
            kv = self.video_kv[frame_meta['videokey']]
            cap = kv['cap']
            lk = kv['lk']
            wh = frame_meta['reso']
            begin_frame = frame_meta['ids'][begin]
            lk.acquire()
            frames = []
            assert cap.set(cv2.CAP_PROP_POS_FRAMES, begin_frame)
            for _ in range(self.prev_n):
                success, frame = cap.read()
                assert success
                frames.append(frame)
            lk.release()
            tensors = []
            for f in frames:
                tensors.append(totensor(im=f, wh=wh))
            x = torch.stack(tensors)
        elif self.method == 'mask':
            boxlists = self.bbox_collection[clip_index][begin:end]
            x = _boxlist2tensor_channelstack(boxlists, factor=self.factor)

        assert x is not None
        return x, ans


class CASDataset(Dataset):
    def __init__(self, clip_home, outlier_size=30, fraction=1.0, combinator=opticalflow2tensor, max_diff=2, window_size=16):
        self.entries: List[Entry] = []
        self.max_diff = max_diff
        self.combinator = combinator
        self.video_cap_pool = dict()

        # Create TemporalSets. # FIXME: ...
        folderlist = [x for x in os.listdir(clip_home) if os.path.isdir(os.path.join(clip_home, x)) and 'video0' in x]
        folderlist = folderlist[:int(round(len(folderlist) * fraction))]
        print('Loading data...')
        for folder in tqdm(folderlist):
            folder = os.path.join(clip_home, folder)
            raw_data = np.load(os.path.join(folder, 'result.npy'), allow_pickle=True).item()
            max_skip = np.array(raw_data['max_skip'], dtype=np.int32)
            car_count = raw_data['car_count']
            boxlists = raw_data['boxlists']

            frame_idx = raw_data['frame_ids']
            reso = raw_data['resolution']
            src_path = raw_data['src_path']

            if src_path not in self.video_cap_pool.keys():
                self.video_cap_pool[src_path] = (cv2.VideoCapture(src_path), Lock())

            begins = np.arange(0, len(car_count) // window_size)
            ends = begins + np.random.randint(2, window_size, size=(len(begins)))

            pos_labels = []
            neg_labels = []
            for k, (li, ri) in enumerate(zip(begins, ends)):
                label = int(car_count[li:ri].std() < 0.1)
                if label:
                    pos_labels.append(k)
                else:
                    neg_labels.append(k)
                    
            if len(pos_labels) > len(neg_labels):
                pos_labels = random.sample(pos_labels, len(neg_labels))
            else:
                neg_labels = random.sample(neg_labels, len(pos_labels))

            for idx in pos_labels:
                li = begins[idx]
                ri = ends[idx]
                assert li < ri
                if combinator in [boxlist2tensor, boxembeddingpair, diff_encoder]:
                    l = boxlists[li]
                    r = boxlists[ri]
                else:
                    # (Video, Index)
                    l = FrameDescription(self.video_cap_pool[src_path], frame_idx[li], reso)
                    r = FrameDescription(self.video_cap_pool[src_path], frame_idx[ri], reso)
                self.entries.append(Entry(l, r, 1))
            
            for idx in neg_labels:
                li = begins[idx]
                ri = ends[idx]
                assert li < ri
                if combinator in [boxlist2tensor, boxembeddingpair, diff_encoder]:
                    l = boxlists[li]
                    r = boxlists[ri]
                else:
                    # (Video, Index)
                    l = FrameDescription(self.video_cap_pool[src_path], frame_idx[li], reso)
                    r = FrameDescription(self.video_cap_pool[src_path], frame_idx[ri], reso)
                self.entries.append(Entry(l, r, 0))

    def __len__(self) -> int:
        return len(self.entries)
    
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

    def __getitem__(self, index: int):
        l, r, label = self.entries[index]
        if self.combinator not in [boxlist2tensor, boxembeddingpair, diff_encoder]:
            lock: Lock = l.cap_lock[1]
            cap: cv2.VideoCapture = l.cap_lock[0]
            lock.acquire()
            
            assert cap.set(cv2.CAP_PROP_POS_FRAMES, l.index)
            success, frame = cap.read()
            assert success
            l = cv2.resize(frame, l.resolution)
            
            assert cap.set(cv2.CAP_PROP_POS_FRAMES, r.index)
            success, frame = cap.read()
            assert success
            r = cv2.resize(frame, r.resolution)

            lock.release()
        x = self.combinator(l, r)
        # if x.sum() > 0:
        #     from .improcessing import _boxlist2tensor
        #     print(torch.nonzero(_boxlist2tensor(l), as_tuple=True))
        #     raise Exception('...')

        return x, label

ClipElement = namedtuple('ClipElement', ('data', 'max_size', 'labels'))


class CASEvaluator:
    def __init__(self, folder, fetch_size=32, combinator=opticalflow2tensor, max_diff=2, best=False):
        self.clips: List[ClipElement] = []
        self.fetch_size = fetch_size
        self.video_cap_pool = dict()
        self.best = best
        self.combinator = combinator
        self.tolerant_diff = max_diff
        item_list = [x for x in os.listdir(folder) if 'video0' in x]  # FIXME: ...
        assert len(item_list) > 0
        for f in tqdm(item_list):
            if os.path.isdir(os.path.join(folder, f)):
                path = os.path.join(folder, f)
                raw_data = np.load(os.path.join(path, 'result.npy'), allow_pickle=True).item()
                labels = raw_data['car_count']
                max_size = len(labels)
                if raw_data['src_path'] not in self.video_cap_pool.keys():
                    self.video_cap_pool[raw_data['src_path']] = cv2.VideoCapture(raw_data['src_path'])
                if self.combinator in [boxlist2tensor, boxembeddingpair, diff_encoder] or type(self.combinator) is iou_pairing_skipper:
                    self.clips.append(ClipElement(data=raw_data['boxlists'], max_size=max_size, labels=labels))
                else:
                    self.clips.append(
                        ClipElement(data=(raw_data['src_path'], raw_data['frame_ids'], raw_data['resolution']), max_size=max_size, labels=labels))

    def evaluate(self, model, train_hook=None):
        ret_mae = []
        ret_skip = []
        for cc in tqdm(self.clips):
            c = cc
            predicted = np.ones(c.max_size) * -1 # -1 is a flag.

            def fetch_one(index):
                if self.combinator in [boxlist2tensor, boxembeddingpair, diff_encoder] or type(self.combinator) is iou_pairing_skipper:
                    return c.data[index]
                cap: cv2.VideoCapture = self.video_cap_pool[c.data[0]]
                assert cap.set(cv2.CAP_PROP_POS_FRAMES, c.data[1][index])
                success, frame = cap.read()
                assert success
                return cv2.resize(frame, c.data[2])

            def CAS(begin, end, skipped_size, cur_errors):
                # [begin] [end]
                if end == begin:
                    return
                # [begin] [?] [end]
                # [begin] [?] [?] [end]
                if end - begin <= 2:
                    predicted[begin] = c.labels[begin]
                    predicted[end - 1] = c.labels[end - 1]
                    return
                
                skipped = False
                if end - begin > 2:
                    lc = c.labels[begin]
                    rc = c.labels[end - 1]
                    predicted[begin] = lc
                    predicted[end - 1] = rc
                    
                    if abs(lc - rc) <= self.tolerant_diff:
                        skip_or_not = None
                        
                        if self.best:
                            skip_or_not = len(np.unique(c.labels[begin+1 : end-1])) < 2 and c.labels[begin+1] == lc
                        elif type(self.combinator) is iou_pairing_skipper:
                            skip_or_not = self.combinator.judge(fetch_one(begin), fetch_one(end - 1))
                        else:
                            inp = self.combinator(fetch_one(begin), fetch_one(end - 1)).cuda()
                            skip_or_not = torch.max(
                                model.forward(inp).data, 1)[1].cpu().numpy()[0] == True

                        if skip_or_not:
                            predicted[begin+1 : end-1] = lc
                            cur_errors[0] += np.abs(lc - c.labels[begin+1 : end-1]).sum()
                            skipped_size += (end - begin - 1)
                            skipped = True
                
                if not skipped:
                    partition = (end + begin) // 2
                    CAS(begin + 1, partition, skipped_size, cur_errors)
                    CAS(partition, end - 1, skipped_size, cur_errors)

            begin_ = 0
            end_ = 0
            skipped_size = np.array([0])
            cur_errors = np.array([0]) 

            if train_hook is not None:
                train_hook.total = len(self.clips)

            last_skip = 0
            while end_ != c.max_size:
                if train_hook is not None: # update ... 
                    # Training: Input Prev Segments...
                    if train_hook.is_train:
                        train_hook.train(c.labels, predicted, begin_, skipped_size[0] - last_skip)
                    else:
                        train_hook.adaptation_inference(c.labels, predicted, begin_)
                end_ = min(begin_ + self.fetch_size, c.max_size)
                CAS(begin_, end_, skipped_size, cur_errors)
                begin_ = end_
                last_skip = skipped_size[0]

            if train_hook is not None:
                train_hook.processed += 1
                train_hook.reset_for_next_clip()

            bugs = (predicted < 0).nonzero()[0]
            if len(bugs) != 0:
                print(predicted)
                print(bugs)
                raise Exception('Bugs occurred: There are unchecked frames...')

            predicted = np.array(predicted)
            ret_mae.append(np.abs(predicted - c.labels).mean())
            ret_skip.append(skipped_size[0] / len(c.labels))
        return np.array(ret_mae), np.array(ret_skip)
