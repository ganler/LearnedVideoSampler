import argparse
import math
import os
import sys
from datetime import datetime

import torch
import torch.nn.functional as F

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)

from utility.rlhelpers import *
from application.carcounter.CarCounter import YOLOConfig
from models.experimental import SeriesUnlinearPolicyNet
from utility.videoloader import create_train_test_datasets

config = YOLOConfig()

parser = argparse.ArgumentParser()
parser.add_argument('--action_space', type=int, default=16)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--accuracy_sla', type=float, default=0.95)
cfg = parser.parse_args()

print('Configuration Parameters: ')
print(cfg)

LOG_DISTRIBUTION = True
EPOCH = 3
BATCH_SIZE = cfg.batch_size
FACTOR = 4
N_PREV = 12
RATE_OPTIONS = np.arange(cfg.action_space)
VIDEO_FOLDER = os.path.join(project_dir, 'data')
LOSS_RECORD_DUR = 10
GAMMA = 0.9
VIDEO_SUFFIX = '.avi'
PRETRAINED_PATH = None
TARGET_UPDATE = 15
ACCURACY_SLA = cfg.accuracy_sla
# REPLAY_BUFFER = ReplayMemory(BATCH_SIZE * 40)
REPLAY_BUFFER = BalancedReplayMemory(BATCH_SIZE * 20, n_option=len(RATE_OPTIONS))
SKIP_COST = 0
INFER_COST = 3
SLA_PENALTY_LONG = -500
SLA_PENALTY_SHORT = SLA_PENALTY_LONG * 10
SKIP_REWARD_FACTOR = 1
EPS_START = 0.9
EPS_END = 0.05


def reward_function(avg_accumulative_accuracy, this_acc, frames_skipped, best_skip):
    return max(ACCURACY_SLA - avg_accumulative_accuracy, 0) * SLA_PENALTY_LONG * (1 - this_acc) \
           + max(ACCURACY_SLA - this_acc, 0) * SLA_PENALTY_SHORT \
           + (frames_skipped - 1) * INFER_COST \
           + abs(best_skip - frames_skipped) * SKIP_REWARD_FACTOR \
           - SKIP_COST


# Simplified reward function.
# def reward_function(avg_accumulative_accuracy, this_acc, frames_skipped, best_skip):
#     return max(ACCURACY_SLA - avg_accumulative_accuracy, 0) * SLA_PENALTY_SHORT * max(frames_skipped - best_skip, 0) \
#            + (frames_skipped - 1) * INFER_COST \
#            + abs(best_skip - frames_skipped) * SKIP_REWARD_FACTOR \
#            - SKIP_COST


if __name__ == '__main__':
    policy_net = SeriesUnlinearPolicyNet(N_PREV, len(RATE_OPTIONS)).cuda()
    target_net = SeriesUnlinearPolicyNet(N_PREV, len(RATE_OPTIONS)).cuda()
    target_net.load_state_dict(policy_net.state_dict())
    target_net.requires_grad_(False)
    target_net.eval()

    optimizer = torch.optim.RMSprop(policy_net.parameters(), lr=1e-4, weight_decay=0.1)

    train_data, test_data = create_train_test_datasets(
        folder=VIDEO_FOLDER,
        suffix=VIDEO_SUFFIX,
        episode_mode=True,
        train_proportion=0.8,
        use_image=False,
        n_box=N_PREV)

    records = {
        'reward': [],
        'skipped_frames': [],
        'accuracy': [],
        'loss': []
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


        print(f'Starting Epoch #{epoch}')
        for i, (boxlists, (car_cnt, max_skip)) in enumerate(train_data):
            detection_seq = torch.Tensor([b.shape[0] for b in boxlists]).unsqueeze_(0).cuda()
            if current_state is None:
                current_state = detection_seq
                continue
            else:
                next_state = detection_seq

            # State = (image, encoded_boxlists)
            # def select_action():
            # TODO: \epsilon-greedy exploration
            eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * episode_index / len(train_data.ptr))

            if random.random() > eps_threshold:
                with torch.no_grad():
                    _, next_skip = torch.max(policy_net(current_state).data, 1)
                    action = next_skip.cpu().long().numpy()[0]
            else:
                action = RATE_OPTIONS[np.random.randint(0, len(RATE_OPTIONS))]

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
                min(max_skip, RATE_OPTIONS[-1]))
            # FIXME: Moving to seperate mode.(origin = use acc_numerator / acc_denominator)
            acc_reward *= GAMMA
            acc_reward += reward
            REPLAY_BUFFER.push(current_state, action, next_state, reward)

            if done:
                avg_accu_this_episode = (acc_numerator / acc_denominator) * 100
                info = f'\n>> In #{episode_index + 1} episode, accum. reward: {acc_reward:.3f}, '
                info += f'skipped / all: {acc_skipped} / {acc_frames}, avg. accuracy: {avg_accu_this_episode:.3f} %'
                print(info)
                print(f'>> ReplayBuffer size := {len(REPLAY_BUFFER)} / {REPLAY_BUFFER.capacity}')
                records['reward'].append(acc_reward)
                if (episode_index + 1) % LOSS_RECORD_DUR == 0 or (episode_index + 1) == len(train_data.ptr):
                    target_net.load_state_dict(policy_net.state_dict())
                    print('>> .... Start periodic evaluation.')
                    test_data.reset()
                    with torch.no_grad():
                        target_net.eval()
                        skip_accum = 0
                        numerator = 0
                        denominator = 1e-7
                        distribution = np.zeros(len(RATE_OPTIONS))
                        for boxlists, (car_cnt, max_skip) in test_data:
                            detection_seq = torch.Tensor([b.shape[0] for b in boxlists]).unsqueeze(0).cuda()
                            out = target_net(detection_seq)
                            _, predicted = torch.max(out.data, 1)
                            predicted = RATE_OPTIONS[predicted.cpu().numpy()[0]]
                            distribution[predicted] += 1
                            res, _ = test_data.skip_and_evaluate(predicted)
                            skip_accum += predicted
                            numerator += sum(res)
                            denominator += len(res)
                    distribution /= distribution.sum()
                    for idx, d in enumerate(distribution):
                        print(f'For skip size = {idx},\t the proportion in skipping is {d:.3f}')
                    avg_accuracy = numerator / denominator
                    records['accuracy'].append(avg_accuracy)
                    records['skipped_frames'].append(skip_accum)
                    print(
                        f'>> EVAL => acc.: {avg_accuracy * 100:.3f}% | skipped_frames: {skip_accum} / {len(test_data)}')
                reset_states()
            else:
                current_state = next_state

            # Optimize Model
            if len(REPLAY_BUFFER) >= BATCH_SIZE:
                transitions = REPLAY_BUFFER.sample(BATCH_SIZE)
                batch = Transition(*zip(*transitions))

                # state_batch_image =
                state_image_batch = torch.cat(batch.state)
                next_state_image_batch = torch.cat(batch.next_state)

                action_batch = torch.from_numpy(np.array(batch.action, dtype=np.long)).unsqueeze(1).long().cuda()
                reward_batch = torch.from_numpy(np.array(batch.reward, dtype=np.long)).cuda()

                prob = policy_net(state_image_batch)
                state_action_values = prob.gather(1, action_batch)
                next_state_values = target_net(next_state_image_batch).max(1)[0].detach()

                if done and LOG_DISTRIBUTION:
                    with torch.no_grad():
                        count = np.zeros(len(RATE_OPTIONS))
                        pred = np.array(batch.action)
                        val, c_ = np.unique(pred, return_counts=True)
                        count[val] += c_
                        print('The proportion of actions in training samples')
                        print(count / count.sum())
                        _, pred = torch.max(prob.data, 1)
                        count = np.zeros(len(RATE_OPTIONS))
                        pred = pred.cpu().numpy()
                        val, c_ = np.unique(pred, return_counts=True)
                        count[val] += c_
                        print('The proportion of predicted actions in the same training samples')
                        print(count / count.sum())

                # print(next_state_values.shape, reward_batch.shape)

                expected_state_action_values = ((next_state_values * GAMMA) + reward_batch).unsqueeze_(1)

                # print(state_action_values.shape, expected_state_action_values.shape)

                loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

                records['loss'].append(loss.item())
                '''
                Loss = LossFunc{r + Gamma * max Q(s', a') - Q(s, a)}
                '''

                optimizer.zero_grad()
                loss.backward()
                for param in policy_net.parameters():
                    param.grad.data.clamp_(-1, 1)
                optimizer.step()

    record_sign = f'RL-image-{datetime.now().isoformat().split(".")[0]}-epoch-{EPOCH}-clip-{len(train_data)}'
    np.save(f'{VIDEO_FOLDER}/{record_sign}.npy', records)
    torch.save(
        target_net.state_dict(),
        f'{VIDEO_FOLDER}/{record_sign}.pth')
