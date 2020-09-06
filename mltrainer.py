import os
import sys

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from datetime import datetime
from typing import List

project_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_dir)

from application.carcounter.CarCounter import YOLOConfig, preprocess_image
from models.backbone import SamplerBackbone, boxlist2tensor
from utility.videoloader import create_train_test_datasets

config = YOLOConfig()
EPOCH = 5
BATCH_SIZE = 5
RATE_OPTIONS = np.arange(32)
VIDEO_FOLDER = os.path.join(project_dir, 'data')
LOSS_RECORD_DUR = 900 * 4
VIDEO_SUFFIX = '.avi'
PRETRAINED_PATH = None

# def sampler_loss_function(pred: torch.Tensor, label: List[int], ):
#     # Batched impl.
#     assert len(label) == len(pred)
#     numerator = torch.exp(pred)
#     denominator = torch.sum(numerator, dim=1)
#     loss = denominator
#
#     # Discounted loss.
#     for single_label in label:
#         mask = []

if __name__ == '__main__':
    model = SamplerBackbone(len(RATE_OPTIONS)).cuda()
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adamax(model.parameters(), 1e-3, weight_decay=0.05)

    train_data, test_data = create_train_test_datasets(folder=VIDEO_FOLDER, suffix=VIDEO_SUFFIX, train_proportion=0.8)
    records = {
        'loss': [],
        'skipped_frames': [],
        'accuracy': []
    }
    print(f'Training Samples: {len(train_data)}, Test Samples: {len(test_data)}')
    for ep in range(EPOCH):
        train_data.reset()
        running_loss = 0.0
        image_batch = torch.zeros((BATCH_SIZE, 3, config.resolution[1], config.resolution[0])).cuda()
        bbox_batch = torch.zeros((BATCH_SIZE, train_data.n_box, 1, config.resolution[1], config.resolution[0])).cuda()
        label_batch = torch.zeros(BATCH_SIZE, dtype=torch.long).cuda()
        for i, ((image, boxlists), (car_cnt, max_skip)) in tqdm(enumerate(train_data)):
            bbox_batch[i % BATCH_SIZE] = boxlist2tensor(boxlists, tensor_resolution=config.resolution).cuda()
            image_batch[i % BATCH_SIZE] = preprocess_image(image, config.resolution).cuda()
            image_batch[i % BATCH_SIZE] = max(max_skip, len(RATE_OPTIONS) - 1)

            if i % BATCH_SIZE == BATCH_SIZE - 1:
                out = model(image_batch, bbox_batch)
                loss = loss_func(out, label_batch)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            if i % 10000 == LOSS_RECORD_DUR - 1:  # print every 2000 mini-batches
                print('.... Start periodic evaluation.')
                avg_loss = running_loss / LOSS_RECORD_DUR
                records['loss'].append(avg_loss)
                test_data.reset()
                with torch.no_grad():
                    model.eval()
                    skip_accum = 0
                    numerator = 0
                    denominator = 1e-5
                    for (image, boxlists), (car_cnt, max_skip) in test_data:
                        boxtensor = boxlist2tensor(boxlists, tensor_resolution=config.resolution).cuda()
                        imtensor = preprocess_image(image, config.resolution).cuda()
                        out = model(imtensor, boxtensor)
                        _, predicted = torch.max(out.data, 1)
                        predicted = predicted.cpu().numpy()[0]
                        res = test_data.skip_and_evaluate(predicted)
                        skip_accum += predicted
                        numerator += sum(res)
                        denominator += len(res)
                avg_accuracy = numerator / denominator
                records['accuracy'].append(avg_accuracy)
                records['skipped_frames'].append(skip_accum)
                print(f'CURRENT EVALUATION RESULT: accuracy @ {avg_accuracy} | skipped_frames @ {skip_accum}')
                model.train()

                print(
                    f'[epoch={ep + 1}, ingested={i + 1}] AVG loss over {LOSS_RECORD_DUR} samples: {avg_loss:.5f}')
                running_loss = 0.0

    record_sign = f'{datetime.now().isoformat().split(".")[0]}-epoch-{EPOCH}-clip-{len(train_data)}'
    np.save(f'{VIDEO_FOLDER}/{record_sign}.npy', records)
    torch.save(
        model.state_dict(),
        f'{VIDEO_FOLDER}/{record_sign}.pth')