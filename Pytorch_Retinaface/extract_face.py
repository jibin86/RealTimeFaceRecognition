import torch
import numpy as np
import cv2
from detect import Args
import detect
from alignment import alignment_procedure
import matplotlib.pyplot as plt

from data import cfg_mnet, cfg_re50
from models.retinaface import RetinaFace
import torch.backends.cudnn as cudnn


def extract(img_path, trained_model, align=True, network="resnet50", cpu=True):
    torch.set_grad_enabled(False)
    cfg = None

    args = Args(network=network, cpu=cpu, trained_model=trained_model)

    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50

    # net and model
    net = RetinaFace(cfg=cfg, phase="test")
    net = args.load_model(net, args.trained_model, args.cpu)
    net.eval()
    print("Finished loading model!")

    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    ##### 여기 전까지는 미리 실행해놓자 #####

    img_array = np.fromfile(img_path, np.uint8)
    img_raw = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    dets = args.detect_faces(img_raw, cfg, net)

    # alignment
    resp = []
    for b in dets:
        facial_img = img_raw[int(b[1]) : int(b[3]), int(b[0]) : int(b[2])]

        if align:
            left_eye = (b[5], b[6])
            right_eye = (b[7], b[8])
            nose = (b[9], b[10])
            facial_img = alignment_procedure(facial_img, left_eye, right_eye, nose)

        resp.append(facial_img[:, :, ::-1])
    return resp

if __name__ == "__main__":
    img_path = "../../celebrity_dataset/차은우12.jpg"
    faces = extract(img_path = img_path, trained_model='weights/Resnet50_Final.pth', cpu=True)

    for face in faces:
        plt.imshow(face)
        plt.axis('off')
        plt.show()