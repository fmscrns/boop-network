import torch
import cv2
import numpy as np
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.plots import Annotator, colors

@torch.no_grad()
def detclas(
    weights,
    source,
    imgsz=640,
    conf_thres=0.25,
    iou_thres=0.45,
    max_det=1000,
    device='',
):
    model = torch.load(weights, map_location=device)
    model = (model['model']).float()
    names = model.names
    stride = model.stride

    img0 = cv2.imread(str(source))
    img = letterbox(img0, imgsz)[0]
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255
    img = img[None]

    pred = model(img)

    pred = non_max_suppression(pred[0], conf_thres, iou_thres, max_det=max_det)

    det = pred[0]
    p, im0 = str(source), img0.copy()
    annotator = Annotator(im0, line_width=3, example=str(names))
    if len(det):
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

        for *xyxy, conf, cls in reversed(det):
            c = int(cls)
            label = f'{names[c]} {conf:.2f}'
            annotator.box_label(xyxy, label, color=colors(c, True))

    im0 = annotator.result()
    cv2.imshow(p, im0)
    cv2.waitKey(10000)

detect(
    weights="weights.pt",
    source="dog.jpeg",
    imgsz=640,
    conf_thres=0.25,
    iou_thres=0.45,
    max_det=1000,
    device="cpu"
)