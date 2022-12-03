
import torch
from torch import nn
import torchvision
from models.retinaface import RetinaFace
from models.net import *
import cv2
from torchvision import transforms
from PIL import Image
from layers.functions.prior_box import PriorBox
from utils.box_utils import decode, decode_landm
from utils.nms.py_cpu_nms import py_cpu_nms
import numpy as np
import os
import argparse
from Pytorch_Retinaface.test_widerface import load_model

cfg_mnet = {
    'name': 'mobilenet0.25',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 190,
    'decay2': 220,
    'image_size': 640,
    'pretrain': True,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64
}

cfg_re50 = {
    'name': 'Resnet50',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 24,
    'ngpu': 4,
    'epoch': 100,
    'decay1': 70,
    'decay2': 90,
    'image_size': 840,
    'pretrain': True,
    'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
    'in_channel': 256,
    'out_channel': 256
}

test_transforms = transforms.Compose(
    [
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    ]
)

def get_face_detector(args) :

    if args is None or args.face_detector == 'Resnet':
        cfg = cfg_re50
    else:
        cfg = cfg_mnet

    detector = RetinaFace(cfg=cfg, phase='test')
    detector = load_model(detector, args.trained_model, args.cpu).eval()

    return detector , cfg

## 간단한 wrapper class for convenience


def detect_faces(args, detector, image_raw, cfg , device  )  :


    img = np.float32(image_raw)
    # testing scale
    target_size = 1600
    max_size = 2150
    im_shape = img.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    resize = float(target_size) / float(im_size_min)

    # prevent bigger axis from being more than max_size:
    if np.round(resize * im_size_max) > max_size:
        resize = float(max_size) / float(im_size_max)
    if args.origin_size:
        resize = 1
    if resize != 1:
        img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)

    loc, conf, landms = detector(img)

    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)
    landms = landms * scale1 / resize
    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > args.confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1]
    # order = scores.argsort()[::-1][:args.top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)

    keep = py_cpu_nms(dets, args.nms_threshold)

    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]

    landms = landms[keep]
    # keep top-K faster NMS
    # dets = dets[:args.keep_top_k, :]
    # landms = landms[:args.keep_top_k, :]
    dets = np.concatenate((dets, landms), axis=1)

    return dets , landms



def get_cropped_img_raw(args, image_raw , dets  , filename) :

    faces = []
    for det in dets:
        b2 = det

        if b2[4] < args.vis_threshold :
            continue

        b = list(map(int, b2))
        cropped_img = image_raw[max(0, b[1]): b[3], b[0]:b[2]]
        faces.append(cropped_img)

        if args.save_image :
            image_template = image_raw.copy()
            text = "{:.4f}".format(b2[4])
            cv2.rectangle(image_template, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)

            cx = b[0]
            cy = b[1] + 12
            cv2.putText(image_template, text, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
    if args.save_image:
        if not os.path.exists("./facedetection_result/"):
                os.makedirs("./facedetection_result/")
        name = "./facedetection_result/" + filename + ".jpg"
        image_template = cv2.cvtColor(image_template, cv2.COLOR_RGB2BGR)
        cv2.imwrite(name, image_template)


    return faces





if __name__ == '__main__'  :

    fpath = './sample_data' # for sample image face detection
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--trained_model', default='./weights/Resnet50_Final.pth',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
    parser.add_argument('--origin_size', default=True, type=str, help='Whether use origin image size to evaluate')
    parser.add_argument('--save_folder', default='./widerface_evaluate/widerface_txt/', type=str,
                        help='Dir to save txt results')
    parser.add_argument('--cpu', action="store_true", default=True, help='Use cpu inference')
    parser.add_argument('--dataset_folder', default='./data/widerface/val/images/', type=str, help='dataset path')
    parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
    parser.add_argument('--top_k', default=5000, type=int, help='top_k')
    parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
    parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
    parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
    parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
    parser.add_argument('--face_detector' , default = "Resnet" )
    parser.add_argument('--device' , default = 'mps')
    parser.add_argument('--image_path', default = './sample_data', help = 'directory of input images ')
    args  = parser.parse_args()

    detect_faces(args = args)