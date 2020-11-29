import orjson
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--pred', type=str)
parser.add_argument('--gt', type=str)
cfg = parser.parse_args()

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
    for i in range(n):
        if gtruth[i] is None or pred[i] is None:
            continue
        if len(gtruth[i]) != 0 and len(pred[i]) == 0 and last != None:  # Assume Skipping;;;
            skipped += 1
            error += abs(len(last) - len(gtruth[i]))
        else:
            error += abs(len(gtruth[i]) - len(pred[i]))
            last = pred[i]
    print(f'EVAL: pred: {cfg.pred} v.s. gt: {cfg.gt}')
    print(f'MAE error: {error / n}')
    print(f'Filtered: {skipped / n}')
