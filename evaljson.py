import orjson
import argparse
import torch
from utility.f1compute import compute_f1
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--pred', type=str)
parser.add_argument('--gt', type=str)
cfg = parser.parse_args()

def json2tensor(det):
    # det = gtruth[frame_id]
    if det is None:
        det = []
    tensor = torch.zeros((len(det), 5))
    for i, dslice in enumerate(det):
        tensor[i, :] = torch.Tensor([dslice['left'], dslice['top'], dslice['right'], dslice['bottom'], 1.])
    return tensor

if __name__ == '__main__':
    skip_size = 16
    gtruth = None
    with open(cfg.gt, "rb") as f:
        gtruth = orjson.loads(f.read())
    pred = None
    with open(cfg.pred, "rb") as f:
        pred = orjson.loads(f.read())

    error = 0
    skipped = 0
    n = min(len(gtruth), len(pred))
    last = None
    for i in tqdm(range(n)):
        if gtruth[i] is None or pred[i] is None:
            continue
        gt = json2tensor(gtruth[i])
        if len(gt) != 0 and len(pred[i]) == 0 and last != None:  # Assume Skipping;;;
            skipped += 1
        else:
            last = json2tensor(pred[i])
        error += compute_f1(last, gt)
    print(f'EVAL: pred: {cfg.pred} v.s. gt: {cfg.gt}')
    print(f'F1 error: {error / n}')
    print(f'Filtered: {skipped / n}')
