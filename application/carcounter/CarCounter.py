import torch
import cv2
import pathlib
from dataclasses import dataclass
from typing import Tuple

from .yolov3.models import *
from .yolov3.utils.datasets import *
from .yolov3.utils.utils import *


def get_device():
    return torch.device('cuda')


cur_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), 'yolov3')


@dataclass
class YOLOConfig:
    cfg: str = 'yolov3-spp.cfg'
    weights: str = 'yolov3-spp-ultralytics.pt'
    conf_thres: float = 0.3
    iou_thres: float = 0.3
    resolution: Tuple[int, int] = (608, 352)


@torch.no_grad()
def preprocess_image(im, resolution):
    im = cv2.resize(im, resolution)
    inp = im[:, :, ::-1].transpose(2, 0, 1)
    inp = np.ascontiguousarray(inp, dtype=np.float32)
    inp = torch.from_numpy(inp).to(get_device()).float() / 255.

    if inp.ndimension() == 3:
        inp = inp.unsqueeze(0)

    return inp


class CarCounter:
    def __init__(self, config):
        self.model = None
        self.config = config
        self.names = load_classes(os.path.join(cur_dir, 'data', 'coco.names'))
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]

        if type(config) == YOLOConfig:
            config: YOLOConfig
            with torch.no_grad():
                self.model = Darknet(check_file(os.path.join(cur_dir, 'cfg', config.cfg)), config.resolution)

                weight_path = os.path.join(cur_dir, 'weights', config.weights)

                if config.weights.endswith('.pt'):  # pytorch format
                    self.model.load_state_dict(torch.load(weight_path, map_location=get_device())['model'])
                else:  # darknet format
                    load_darknet_weights(self.model, weight_path)

                self.model.to(get_device()).eval()
        else:
            raise Exception('Unknown config type! Maybe you can use `YOLO`')

    def process_image(self, im):
        return preprocess_image(im, self.config.resolution)

    @torch.no_grad()
    def predict(self, inp):
        # t1 = torch_utils.time_synchronized()

        pred = self.model(inp)[0]
        pred = non_max_suppression(pred, self.config.conf_thres, self.config.iou_thres, classes=[2, 5, 7])

        # for p in pred:
        #     print(f'Time elapsed: {1000 * (torch_utils.time_synchronized() - t1)} ms. Detected cars :=> {len(p)}')
        return pred

    def viz(self, pred: torch.Tensor, img):
        img = cv2.resize(img, self.config.resolution)
        for i, det in enumerate(pred):  # detections for image i
            if det is not None and len(det):
                # Rescale boxes from imgsz to im0 size
                # det[:, :4] = scale_coords(, det[:, :4], img.shape).round()

                for j, (*xyxy, conf, cls) in enumerate(reversed(det)):
                    label = '[%i] %s %.2f' % (j, self.names[int(cls)], conf)
                    plot_one_box(xyxy, img, label=label, color=self.colors[int(cls)])

        # Stream results
        cv2.imshow('viz', img)
        if cv2.waitKey(1) == ord('q'):  # q to quit
            raise StopIteration
