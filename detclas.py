import torch
import cv2
import numpy as np
from utils.augmentations import letterbox, ResizeWithAspectRatio
from utils.general import non_max_suppression, scale_coords
from utils.plots import Annotator, colors

@torch.no_grad()
def detclas(
    weights,
    source,
    conf_thres=0.25,
    iou_thres=0.45,
    max_det=1000,
    device='',
    line_width=15
):
    model = torch.load(weights, map_location=device)
    model = (model['model']).float()

    img0 = cv2.imread(str(source))
    img = letterbox(img0)[0]
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255
    img = img[None]

    pred = model(img)
    det = non_max_suppression(pred[0], conf_thres, iou_thres, max_det=max_det)[0]

    im0 = img0.copy()
    names = model.names
    annotator = Annotator(im0, line_width=line_width, example=str(names))
    if len(det):
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
        for *xyxy, conf, cls in reversed(det):
            c = int(cls)
            label = f'{names[c]} {conf:.2f}'
            annotator.box_label(xyxy, label, color=colors(c, True))
    
    anot_img = annotator.result()
    anot_img = ResizeWithAspectRatio(im0, 640)
    cv2.imshow(str(source), anot_img)
    cv2.waitKey(10000)

detclas(
    weights="weights.pt",
    source="dog.jpg",
    conf_thres=0.25,
    iou_thres=0.45,
    max_det=1000,
    device="cpu",
    line_width=2
)