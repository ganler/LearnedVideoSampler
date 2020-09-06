import numpy as np
import cv2
import os
import random
from typing import List, Tuple
import torch


class VideoSamplerDataset:
    cap_list: List[cv2.VideoCapture] = []
    cur_ptr = None
    ptr = None
    last_ptr = None
    last_frame_ptr = None

    def __init__(self, data_pairs: List[Tuple[str, str]], cross_video=False, n_box=5):
        self.data_pairs = data_pairs
        self.n_box = n_box
        self.cross_video = cross_video
        self.data_list = []
        for _, npy_path in self.data_pairs:
            self.data_list.append(np.load(npy_path, allow_pickle=True).item())
        self.total_samples_number = sum([len(x['car_count']) for x in self.data_list])

    def reset(self):
        for x in self.cap_list:
            x.release()
        self.cap_list: List[cv2.VideoCapture] = []
        for clip_path, _ in self.data_pairs:
            self.cap_list.append(cv2.VideoCapture(clip_path))
        self.cur_ptr = 0
        self.ptr = np.zeros(len(self.cap_list), dtype=np.int32)

    def __len__(self):
        return self.total_samples_number

    def __next__(self):
        this_index = None
        if self.cross_video:  # e.g., video 0 -> video 1 -> ... round robin.
            try_n = 0
            this_index = self.cur_ptr
            while try_n < len(self.data_list) and self.ptr[this_index] >= len(self.data_list[this_index]['car_count']):
                try_n += 1
                this_index = (this_index + 1) % len(self.data_list)  # Next round.
            if try_n == len(self.data_list):
                raise StopIteration('Data exhausted...')
            # this_index is OK now.
            self.cur_ptr = (this_index + 1) % len(self.data_list)
        else:  # e.g., read all of video 0, then video 1, ... till video[final]
            if self.ptr[self.cur_ptr] >= len(self.data_list[self.cur_ptr]['car_count']):  # Current clip exhausted.
                self.cur_ptr += 1  # Next One.
            if self.cur_ptr >= len(self.data_list):
                raise StopIteration('Data exhausted...')
            this_index = self.cur_ptr

        self.last_ptr = this_index
        self.last_frame_ptr = self.ptr[this_index]

        this_frame_index = self.ptr[this_index]
        self.ptr[this_index] += 1

        ret, frame = self.cap_list[this_index].read()
        car_count = self.data_list[this_index]['car_count'][this_frame_index]
        max_skip = self.data_list[this_index]['max_skip'][this_frame_index]
        real_boxes = self.data_list[this_index]['boxlists'][max(0, this_frame_index - self.n_box):this_frame_index]
        box_lists = [torch.Tensor([])] * (self.n_box - len(real_boxes)) + real_boxes
        return (frame, box_lists), (car_count, max_skip)

    def __iter__(self):
        return self

    def skip_and_evaluate(self, n):
        self.ptr[self.last_ptr] += n

        if n != 0:
            is_success = self.cap_list[self.last_ptr].set(cv2.CAP_PROP_POS_FRAMES, self.ptr[self.last_ptr])
            if not is_success:
                assert self.ptr[self.last_ptr] >= self.cap_list[self.last_ptr].get(cv2.CAP_PROP_FRAME_COUNT)

        car_counts = self.data_list[self.last_ptr]['car_count']
        predicted_val = car_counts[self.last_frame_ptr]
        max_item_size = len(car_counts)

        ret = [1]
        for label in car_counts[min(self.last_frame_ptr+1, max_item_size):min(self.last_frame_ptr+1+n, max_item_size)]:
            min_val = min(predicted_val, label)
            max_val = max(predicted_val, label)
            ret.append(min_val / max_val if max_val != 0 else 1.)

        return ret



    def next(self):
        return self.__next__()


def create_train_test_datasets(folder, suffix, train_proportion=0.6, n_box=5):
    assert train_proportion <= 1
    assert len(suffix) > 1
    if suffix[0] != '.':
        suffix = '.' + suffix
    name_tokens = [x.replace(suffix, '') for x in os.listdir(folder) if x.endswith(suffix)]
    data_pairs = []
    random.seed(0)
    random.shuffle(name_tokens)  # Shuffle.
    for t in name_tokens:
        data_pairs.append((os.path.join(folder, t + suffix), os.path.join(folder, t + '.npy')))
    split_index = int(train_proportion * len(data_pairs))
    train_pairs, test_pairs = data_pairs[:split_index], data_pairs[split_index:]
    assert len(train_pairs) > 0
    assert len(test_pairs) > 0

    print(f'===> Got {len(train_pairs)} training clips, and {len(test_pairs)} test clips.')
    return \
        VideoSamplerDataset(train_pairs, cross_video=True, n_box=n_box), \
        VideoSamplerDataset(test_pairs, n_box=n_box)