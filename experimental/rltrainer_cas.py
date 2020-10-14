# Copyright (c) 2020 Ganler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import os
import sys
import argparse

import numpy as np
import cv2
import torch.nn.functional as F
import torch
import math
import random
from datetime import datetime

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)


from utility.imcliploader import CASEvaluator
from models.backbone import CASNet
from utility.common import str2bool
from utility.rlhelpers import *
from utility.common import iou_pairing_skipper

parser = argparse.ArgumentParser()
# This means that training & evaluation data all come from `data` folder.
parser.add_argument('--epoch', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--history_length', type=int, default=128)
parser.add_argument('--fetch_size', type=int, default=64)
parser.add_argument('--mae', type=float, default=0.1)
parser.add_argument('--mae_penalty', type=float, default=1.)
parser.add_argument('--skip_award', type=float, default=0.5)
parser.add_argument('--check_point_size', type=int, default=50)
parser.add_argument('--replay_memory_fac', type=int, default=1000)
parser.add_argument('--eps_start', type=float, default=0.6)
parser.add_argument('--eps_end', type=float, default=0.05)
parser.add_argument('--alpha', type=float, default=0.05)

cfg = parser.parse_args()
print(cfg)

# Checks...
assert cfg.eps_start > cfg.eps_end

# Helpers...
REPLAY_MEMORY = ReplayMemory(cfg.batch_size * cfg.replay_memory_fac)

# RW func...
def offline_reward_function(mae_list, frame_skipped):
    mean_mae = np.mean(mae_list)
    return  - (mean_mae - cfg.mae) * len(mae_list) * cfg.mae_penalty if mean_mae > cfg.mae \
        else frame_skipped * cfg.skip_award



# class ClipRealTimeDataLoader:
#     def __init__(self, folder):
#         raw_data = np.load(os.path.join(folder, 'result.npy'), allow_pickle=True).item()
#         self.labels = raw_data['car_count']
#         self.ptr = 0

#     def skip_and_evaluate(self, skip_len):
#         assert self.ptr < len(self.labels)
#         '''
#         skip_len: 0   | => next one.
#         skip_len: > 0 | => skip N using previous label.
#         '''
#         aes = None
#         if skip_len == 0:
#             aes = np.zeros(1)
#         else:
#             assert self.ptr != 0
#             prev_label = self.labels[self.ptr - 1]
#             aes = prev_label - self.labels[]
#         self.ptr += (1 + skip_len)
#         return aes

GAMMA = 0.999

# Main codes...
if __name__ == "__main__":
    policy_net = CASNet(n_inp=cfg.history_length).cuda()
    target_net = CASNet(n_inp=cfg.history_length).cuda()
    target_net.load_state_dict(policy_net.state_dict())
    target_net.requires_grad_(False)
    target_net.eval()

    optimizer = torch.optim.RMSprop(policy_net.parameters(), lr=1e-3)

    record = {
        'acc_reward': [],
        'mae_list': [],
        'ratio_list': []
    }

    class TrainingHook:
        def __init__(self):
            self.total = None
            self.processed = 0
            self.current_state = None
            self.next_state = None
            self.is_train = True
            self.acc_reward = 0
            self.iou_thresher = iou_pairing_skipper()

        def adaptation_inference(self, labels, predicted, cur_ptr):
            history = np.array(labels[(cur_ptr + 1 - cfg.history_length) : (cur_ptr + 1)] \
                    - predicted[(cur_ptr + 1 - cfg.history_length) : (cur_ptr + 1)], dtype=np.float32)

            if len(history) < cfg.history_length:
                return
            
            with torch.no_grad():
                out = target_net.forward(torch.from_numpy(history.astype(np.float32)).unsqueeze(0).cuda())
                at, arg = torch.max(out.data, 1)
                action = arg.cpu().long().numpy()[0]
                at = at.cpu().numpy()[0]
                self.adapt_thresh(action, at)

        def adapt_thresh(self, action, s=None):
            if s is None:
                s = 1
            if action == 0:  # down
                self.iou_thresher.conf_thresh *= (1 + cfg.alpha * s)
            else: # up
                self.iou_thresher.conf_thresh /= (1 + cfg.alpha * s)
            
            self.iou_thresher.conf_thresh = min(self.iou_thresher.conf_thresh, 1.)
            self.iou_thresher.conf_thresh = max(self.iou_thresher.conf_thresh, 0.)

        def reset_for_next_clip(self): # New clip.
            if (self.processed + 1) % cfg.check_point_size == 0:
                record['acc_reward'].append(self.acc_reward)
                self.is_train = False
                target_net.load_state_dict(policy_net.state_dict())
                evaluator = CASEvaluator(
                    folder=os.path.join(project_dir, 'val_data_non_general'), 
                    fetch_size=cfg.fetch_size, 
                    combinator=hooker.iou_thresher)
                mae_list, skip_ratio = evaluator.evaluate(model=None, train_hook=self)
                print('[cur] Acc. Rewards: ', self.acc_reward)
                print('[cur] AVG MAE: ', mae_list.mean())
                print('[cur] Skip Ratio: ', skip_ratio.mean())
                record['mae_list'] = np.append(record['mae_list'], mae_list)
                record['skip_ratio'] = np.append(record['skip_ratio'], skip_ratio)
                print('[global] AVG MAE: ', record['mae_list'].mean())
                print('[global] Skip Ratio: ', record['skip_ratio'].mean())

            self.current_state = None
            self.next_state = None
            self.acc_reward = 0
            self.is_train = True

        def train(self, labels, predicted, cur_ptr, skipping):
            if cur_ptr + 1 < cfg.history_length:
                return
            
            diff = np.array(labels[(cur_ptr + 1 - cfg.history_length) : (cur_ptr + 1)] \
                    - predicted[(cur_ptr + 1 - cfg.history_length) : (cur_ptr + 1)], dtype=np.float32)
            mae = np.abs(diff)
            if self.current_state is None:
                self.current_state = diff
                return
            else:
                self.next_state = diff
                
            eps_threshold = cfg.eps_end + (cfg.eps_start - cfg.eps_end) * math.exp(-1. * self.processed / self.total)
            
            at = None
            if random.random() > eps_threshold:
                with torch.no_grad():
                    out = policy_net(torch.from_numpy(self.current_state).unsqueeze(0).cuda())
                    at, arg = torch.max(out.data, 1)
                    action = arg.cpu().long().numpy()[0]
                    at = at.cpu().numpy()[0]
            else:
                action = random.randint(0, 1)
            
            self.adapt_thresh(action, at)

            reward = offline_reward_function(mae, skipping)
            self.acc_reward *= GAMMA
            self.acc_reward += reward
            '''
            `current_state: current approx. accuracy.
            `next_state: next approx. accuracy.
            `reward: see reward function.
            `action: what you did?
            '''
            REPLAY_MEMORY.push(self.current_state.copy(), action, self.next_state.copy(), reward)

                        # Optimize Model
            if len(REPLAY_MEMORY) >= cfg.batch_size:
                transitions = REPLAY_MEMORY.sample(cfg.batch_size)
                batch = Transition(*zip(*transitions))

                # state_batch_image =
                state_batch = torch.from_numpy(np.array(batch.state)).cuda()
                next_state_batch = torch.from_numpy(np.array(batch.next_state)).cuda()

                action_batch = torch.from_numpy(np.array(batch.action, dtype=np.long)).unsqueeze(1).long().cuda()
                reward_batch = torch.from_numpy(np.array(batch.reward, dtype=np.long)).cuda()

                state_action_values = policy_net(state_batch).gather(1, action_batch)
                next_state_values = target_net(next_state_batch).max(1)[0].detach()

                expected_state_action_values = (next_state_values * GAMMA) + reward_batch

                loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
                '''
                Loss = LossFunc{r + Gamma * max Q(s', a') - Q(s, a)}
                '''

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    hooker = None
    for _ in range(cfg.epoch):
        hooker = TrainingHook()
        evaluator = CASEvaluator(
            folder=os.path.join(project_dir, 'data'), 
            fetch_size=cfg.fetch_size, 
            combinator=hooker.iou_thresher)
        mae_list, skip_ratio = evaluator.evaluate(model=None, train_hook=hooker)
    
    record_sign = f'RL-{datetime.now().isoformat().split(".")[0]}-epoch-{cfg.epoch}-clip-{hooker.total}'
    np.save(f'{project_dir}/result/{record_sign}.npy', record)
    torch.save(
        target_net.state_dict(),
        f'{project_dir}/result/{record_sign}.pth')
