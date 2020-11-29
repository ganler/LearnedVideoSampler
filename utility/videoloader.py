import numpy as np
import cv2
import os
import random
from typing import List, Tuple, Dict
import torch


class VideoSamplerDataset:
    current_frame_ptr: List[int] = []
    cur_video_ptr = None
    ptr = None
    last_ptr = None

    def __init__(self, dirlist, cross_video=False, use_image=True, discount=0.95):
        self.dirlist = dirlist
        self.cross_video = cross_video
        self.data_list = []
        self.use_image = use_image
        for f in dirlist:
            full_data: Dict = np.load(f, allow_pickle=True).item()
            discount_index = int(len(full_data['car_count']) * discount)
            full_data['max_skip'] -= 1 # Make it compatible
            if full_data['boxlists'][0] is None:
                full_data['boxlists'][0] = []
            for k in ['car_count', 'boxlists', 'max_skip']:
                full_data[k] = full_data[k][:discount_index]
            self.data_list.append(full_data)
            self.current_frame_ptr.append(0)
        self.total_samples_number = sum([len(x['car_count']) for x in self.data_list])

    def reset(self):
        for x in self.current_frame_ptr:
            x = 0
        self.cur_video_ptr = 0
        self.ptr = np.zeros(len(self.current_frame_ptr), dtype=np.int32)

    def __len__(self):
        return self.total_samples_number

    def __next__(self):
        if self.cross_video:  # e.g., video 0 -> video 1 -> ... round robin.
            candidates = []
            for i in range(len(self.ptr)):
                if self.ptr[i] < len(self.data_list[i]['car_count']):
                    candidates.append(i)
            if len(candidates) == 0:
                raise StopIteration('Data exhausted...')
            this_index = candidates[random.randint(0, len(candidates) - 1)]
        else:  # e.g., read all of video 0, then video 1, ... till video[final]
            if self.ptr[self.cur_video_ptr] >= len(self.data_list[self.cur_video_ptr]['car_count']):  # Current clip exhausted.
                self.cur_video_ptr += 1  # Next One.
            if self.cur_video_ptr >= len(self.data_list):
                raise StopIteration('Data exhausted...')
            this_index = self.cur_video_ptr

        self.last_ptr = this_index

        this_frame_index = self.ptr[this_index]
        self.ptr[this_index] += 1

        frame = None
        if self.use_image:
            frame = cv2.imread(os.path.join(self.dirlist[this_index], f'{self.current_frame_ptr[this_index]}.jpg'))
            self.current_frame_ptr[this_index] += 1

        car_count = self.data_list[this_index]['car_count'][this_frame_index]
        max_skip = self.data_list[this_index]['max_skip'][this_frame_index]
        box_lists = self.data_list[this_index]['boxlists'][this_frame_index]
        if self.use_image:
            return (frame, box_lists), (car_count, max_skip)
        else:
            return box_lists, (car_count, max_skip)

    def __iter__(self):
        return self

    def skip_and_evaluate(self, n):
        last_frame_ptr = self.ptr[self.last_ptr] - 1
        self.ptr[self.last_ptr] += n
        is_over = self.ptr[self.last_ptr] >= len(self.data_list[self.last_ptr]['car_count'])

        if not is_over and n != 0:
            self.current_frame_ptr[self.last_ptr] = self.ptr[self.last_ptr]

        car_counts = self.data_list[self.last_ptr]['car_count']
        predicted_val = car_counts[last_frame_ptr]
        max_item_size = len(car_counts)

        # MAE.
        ret = np.abs(
            car_counts[min(last_frame_ptr, max_item_size):min(last_frame_ptr + 1 + n, max_item_size)] - predicted_val)

        return ret, is_over

    def next(self):
        return self.__next__()


def create_train_test_datasets(folder, episode_mode=False, use_image=True, train_proportion=0.6):
    assert train_proportion <= 1
    dirlist = [os.path.join(folder, x) for x in os.listdir(folder) if os.path.isdir(os.path.join(folder, x))]
    random.seed(0)
    random.shuffle(dirlist)  # Shuffle.
    split_index = int(train_proportion * len(dirlist))
    train_pairs, test_pairs = dirlist[:split_index], dirlist[split_index:]

    print(f'===> Got {len(train_pairs)} training clips, and {len(test_pairs)} test clips.')
    return \
        VideoSamplerDataset(train_pairs, cross_video=(not episode_mode), use_image=use_image), \
        VideoSamplerDataset(test_pairs, use_image=use_image)
