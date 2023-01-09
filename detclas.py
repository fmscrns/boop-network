import torch
import cv2
import numpy as np
from pathlib import Path
from utils.augmentations import letterbox, ResizeWithAspectRatio
from utils.general import non_max_suppression as nms, scale_coords
from utils.plots import Annotator, colors

@torch.no_grad()
def detclas(weights, src, conf_thres, iou_thres, max_det, device, line_width):
    yolo = torch.load(weights, map_location=device)
    yolo_32bit = (yolo['model']).float()

    img_to_numpyArr = cv2.imread(str(src))
    numpyArr_padded = letterbox(img_to_numpyArr)[0]
    numpyArr_channelFix = numpyArr_padded.transpose((2, 0, 1))[::-1]
    numpyArr_contig = np.ascontiguousarray(numpyArr_channelFix)

    numpyArr_to_tensor = torch.from_numpy(numpyArr_contig).to(device)
    tensor_32bit = numpyArr_to_tensor.float()
    tensor_normalized = tensor_32bit/255
    tensor_addedBatchDim = tensor_normalized[None]

    pred = yolo_32bit(tensor_addedBatchDim)
    det = nms(pred[0], conf_thres, iou_thres, max_det=max_det)[0]

    im0 = img_to_numpyArr.copy()
    classes = yolo_32bit.names
    annotator = Annotator(im0, line_width=line_width, example=str(classes))
    if len(det):
        det[:, :4] = scale_coords(tensor_addedBatchDim.shape[2:], det[:, :4], im0.shape).round()
        for *xyxy, conf, cls in reversed(det):
            c = int(cls)
            label = f'{classes[c]} {conf:.2f}'
            annotator.box_label(xyxy, label, color=colors(c, True))
    
    annot = annotator.result()
    annot = ResizeWithAspectRatio(im0, 640)
    cv2.imshow(str(src), annot)
    cv2.waitKey(0)

if __name__ == "__main__":
    dog_pname = input("Enter the full pathname of the image with dog:")
    
    dog_file = Path(dog_pname)
    if dog_file.is_file():
        detclas("weights.pt", r"{}".format(dog_pname), 0.25, 0.45, 5, "cpu", 2)

    else:
        print("Error on getting file. No such file exists.")