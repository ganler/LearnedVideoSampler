import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime
import math
import time
import cv2

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)

from utility.rlhelpers import *
from application.carcounter.CarCounter import YOLOConfig, preprocess_image
from models.experimental import ImagePolicyNet
from utility.videoloader import create_train_test_datasets

config = YOLOConfig()
EPOCH = 3
BATCH_SIZE = 32
FACTOR = 4
RATE_OPTIONS = np.arange(16)
VIDEO_FOLDER = os.path.join(project_dir, 'data')
LOSS_RECORD_DUR = 10
GAMMA = 0.9
VIDEO_SUFFIX = '.avi'
PRETRAINED_PATH = None
TARGET_UPDATE = 10
ACCURACY_SLA = 0.95
REPLAY_BUFFER = ReplayMemory(BATCH_SIZE * 50)
SKIP_COST = 6
INFER_COST = 3
SLA_PENALTY_LONG = -1000
SLA_PENALTY_SHORT = SLA_PENALTY_LONG * 10
SKIP_REWARD_FACTOR = 1
EPS_START = 0.6
EPS_END = 0.05


def reward_function(avg_accumulative_accuracy, this_acc, frames_skipped, best_skip):
    return max(ACCURACY_SLA - avg_accumulative_accuracy, 0) * SLA_PENALTY_LONG * (1 - this_acc) \
           + max(ACCURACY_SLA - this_acc, 0) * SLA_PENALTY_SHORT \
           + (frames_skipped - 1) * INFER_COST \
           + abs(best_skip - frames_skipped) * SKIP_REWARD_FACTOR \
           - SKIP_COST


if __name__ == '__main__':
    policy_net = ImagePolicyNet(len(RATE_OPTIONS), pretrained=False, frozen=False).cuda()
    target_net = ImagePolicyNet(len(RATE_OPTIONS), pretrained=False, frozen=False).cuda()
    target_net.load_state_dict(policy_net.state_dict())
    target_net.requires_grad_(False)
    target_net.eval()

    optimizer = torch.optim.RMSprop(policy_net.parameters(), weight_decay=0.1)

    train_data, test_data = create_train_test_datasets(
        folder=VIDEO_FOLDER, suffix=VIDEO_SUFFIX, episode_mode=True, train_proportion=0.8)
    records = {
        'reward': [],
        'skipped_frames': [],
        'accuracy': []
    }

    print(f'Training Samples: {len(train_data)}, Test Samples: {len(test_data)}')

    for epoch in range(EPOCH):
        train_data.reset()

        current_state = None
        next_state = None
        acc_numerator = 0
        acc_denominator = 1e-7
        acc_skipped = 0
        acc_frames = 0
        acc_reward = 0

        episode_index = 0


        def reset_states():
            global current_state, next_state, \
                acc_numerator, acc_denominator, \
                acc_skipped, acc_frames, \
                episode_index, acc_reward
            current_state = None
            next_state = None
            acc_numerator = 0
            acc_denominator = 1e-7
            acc_skipped = 0
            acc_frames = 0
            episode_index += 1
            acc_reward = 0

        hsv = np.zeros((config.resolution[1], config.resolution[0], 3), dtype=np.uint8)
        hsv[..., 1] = 255

        prev = None
        for i, ((image, boxlists), (car_cnt, max_skip)) in tqdm(enumerate(train_data),
                                                                desc=f'#{epoch + 1} Training Epoch'):
            next = cv2.resize(image, config.resolution)
            next = cv2.cvtColor(next, cv2.COLOR_BGR2GRAY)

            if prev is not None:
                flow = cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                hsv[..., 0] = ang * 180 / np.pi / 2
                hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

                # cv2.imshow('f', rgb)
                # cv2.waitKey(1)

                input = preprocess_image(rgb, config.resolution).cuda()

                if current_state is None:
                    current_state = input
                    continue
                else:
                    next_state = input

                # State = (image, encoded_boxlists)
                # def select_action():
                # TODO: \epsilon-greedy exploration
                eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * episode_index / len(train_data.ptr))

                if random.random() > eps_threshold:
                    with torch.no_grad():
                        _, next_skip = torch.max(policy_net(current_state).data, 1)
                        action = next_skip.cpu().long().numpy()[0]
                else:
                    action = random.sample(RATE_OPTIONS.tolist(), 1)[0]
            else:
                action = 0  # No skip.

            accuracy_list, done = train_data.skip_and_evaluate(action)

            # * Update records
            acc_numerator += sum(accuracy_list)
            acc_denominator += len(accuracy_list)
            acc_skipped += action
            acc_frames += (action + 1)
            # * REWARD
            reward = reward_function(
                acc_numerator / acc_denominator,
                sum(accuracy_list) / len(accuracy_list),
                action,
                min(max_skip, len(RATE_OPTIONS)))
            acc_reward *= GAMMA
            acc_reward += reward

            if prev is not None:
                REPLAY_BUFFER.push(current_state, action, next_state, reward)
                current_state = next_state

            prev = next
            if action != 0:
                prev = None

            if done:
                avg_accu_this_episode = (acc_numerator / acc_denominator) * 100
                info = f'\n>> In #{episode_index + 1} episode, accum. reward: {acc_reward:.3f}, '
                info += f'skipped / all: {acc_skipped} / {acc_frames}, avg. accuracy: {avg_accu_this_episode:.3f} %'
                print(info)
                print(f'>> ReplayBuffer size := {len(REPLAY_BUFFER)} / {REPLAY_BUFFER.capacity}')
                records['reward'].append(acc_reward)
                prev = None
                if (episode_index + 1) % LOSS_RECORD_DUR == 0 or (episode_index + 1) == len(train_data.ptr):
                    target_net.load_state_dict(policy_net.state_dict())
                    print('>> .... Start periodic evaluation.')
                    test_data.reset()
                    with torch.no_grad():
                        target_net.eval()
                        skip_accum = 0
                        numerator = 0
                        denominator = 1e-7
                        test_prev = None
                        for (image, boxlists), (car_cnt, max_skip) in test_data:
                            next = cv2.resize(image, config.resolution)
                            next = cv2.cvtColor(next, cv2.COLOR_BGR2GRAY)

                            if test_prev is not None:
                                flow = cv2.calcOpticalFlowFarneback(test_prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                                hsv[..., 0] = ang * 180 / np.pi / 2
                                hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                                rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                                imtensor = preprocess_image(rgb, config.resolution).cuda()

                                out = target_net(imtensor)
                                _, predicted = torch.max(out.data, 1)
                                predicted = RATE_OPTIONS[predicted.cpu().numpy()[0]]
                            else:
                                predicted = 0

                            test_prev = None if predicted != 0 else next
                            res, test_done = test_data.skip_and_evaluate(predicted)

                            if test_done:
                                test_prev = None

                            skip_accum += predicted
                            numerator += sum(res)
                            denominator += len(res)
                    avg_accuracy = numerator / denominator
                    records['accuracy'].append(avg_accuracy)
                    records['skipped_frames'].append(skip_accum)
                    print(
                        f'>> EVAL => acc.: {avg_accuracy * 100:.3f}% | skipped_frames: {skip_accum} / {len(test_data)}')
                reset_states()

            # Optimize Model
            if len(REPLAY_BUFFER) >= BATCH_SIZE:
                transitions = REPLAY_BUFFER.sample(BATCH_SIZE)
                batch = Transition(*zip(*transitions))

                # state_batch_image =
                state_image_batch = torch.cat(batch.state)
                next_state_image_batch = torch.cat(batch.next_state)

                action_batch = torch.from_numpy(np.array(batch.action, dtype=np.long)).unsqueeze(1).long().cuda()
                reward_batch = torch.from_numpy(np.array(batch.reward, dtype=np.long)).cuda()

                state_action_values = policy_net(state_image_batch).gather(1, action_batch)

                next_state_values = target_net(next_state_image_batch).max(1)[0].detach()

                # print(next_state_values.shape, reward_batch.shape)

                expected_state_action_values = (next_state_values * GAMMA) + reward_batch

                # print(state_action_values.shape, expected_state_action_values.shape)

                loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
                '''
                Loss = LossFunc{r + Gamma * max Q(s', a') - Q(s, a)}
                '''

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    record_sign = f'RL-image-{datetime.now().isoformat().split(".")[0]}-epoch-{EPOCH}-clip-{len(train_data)}'
    np.save(os.path.join(VIDEO_FOLDER, f'{record_sign}.npy'), records)
    torch.save(
        target_net.state_dict(),
        os.path.join(VIDEO_FOLDER, f'{record_sign}.pth'))