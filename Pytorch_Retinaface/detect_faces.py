import os
import matplotlib.pyplot as plt
import time
from PIL import Image
import numpy as np
from numpy import dot
from numpy.linalg import norm
import math
import warnings
import torch
import torch.backends.cudnn as cudnn
import cv2
import time

from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from alignment import alignment_procedure

### 이 코드는 이미지를 입력 받고, 이미지의 얼굴 부분을 검출하고, 눈과 코에 맞춰 얼굴 정렬까지 하는 알고리즘이다.
### 이 코드를 실행하면, if __name__ == "__main__": 부분의 코드가 실행되며, 얼굴 검출을 할 수 있다.


class Face_Dectector:
    def __init__(
        self,
        trained_model="./weights/Resnet50_Final.pth",
        network="resnet50",
        cpu=True
    ):
        self.trained_model = trained_model  # 학습된 모델의 경로
        self.network = network  # 사용할 네트워크의 이름 (resnet50, mobile0.25)
        self.cpu = cpu  # CPU를 사용할지 여부

        self.confidence_threshold = 0.02  # 얼굴 검출에 사용할 confidence 임계값
        self.nms_threshold = 0.4  # 비최대 억제(non-maximum suppression)를 위한 임계값
        self.vis_thres = 0.5  # 시각화를 위한 임계값
        self.device = torch.device("cpu" if cpu else "cuda")
        self.cfg = self.set_cfg()

    def set_cfg(self):
        if self.network == "mobile0.25":
            return cfg_mnet
        elif self.network == "resnet50":
            return cfg_re50
        
    ### Face detection을 위한 함수 ###

    def check_keys(self, model, pretrained_state_dict):
        ckpt_keys = set(pretrained_state_dict.keys())
        model_keys = set(model.state_dict().keys())
        used_pretrained_keys = model_keys & ckpt_keys
        unused_pretrained_keys = ckpt_keys - model_keys
        missing_keys = model_keys - ckpt_keys
        print("Missing keys:{}".format(len(missing_keys)))
        print("Unused checkpoint keys:{}".format(len(unused_pretrained_keys)))
        print("Used keys:{}".format(len(used_pretrained_keys)))
        assert len(used_pretrained_keys) > 0, "load NONE from pretrained checkpoint"
        return True

    def remove_prefix(self, state_dict, prefix):
        """Old style model is stored with all names of parameters sharing common prefix 'module.'"""
        print("remove prefix '{}'".format(prefix))
        f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
        return {f(key): value for key, value in state_dict.items()}

    def load_model(self, model, pretrained_path, load_to_cpu):
        print("Loading pretrained model from {}".format(pretrained_path))
        if load_to_cpu:
            pretrained_dict = torch.load(
                pretrained_path, map_location=lambda storage, loc: storage
            )
        else:
            device = torch.cuda.current_device()
            pretrained_dict = torch.load(
                pretrained_path, map_location=lambda storage, loc: storage.cuda(device)
            )
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = self.remove_prefix(
                pretrained_dict["state_dict"], "module."
            )
        else:
            pretrained_dict = self.remove_prefix(pretrained_dict, "module.")
        self.check_keys(model, pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)
        return model

    def detect(self, img_array, net):
        # testing scale
        resize = 1

        img = np.float32(img_array)
        if resize != 1:
            img = cv2.resize(
                img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR
            )
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        scale = scale.to(self.device)

        tic = time.time()
        loc, conf, landms = net(img)  # forward pass
        print("net forward time: {:.4f}".format(time.time() - tic))

        tic = time.time()
        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg["variance"])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg["variance"])
        scale1 = torch.Tensor(
            [
                img.shape[3],
                img.shape[2],
                img.shape[3],
                img.shape[2],
                img.shape[3],
                img.shape[2],
                img.shape[3],
                img.shape[2],
                img.shape[3],
                img.shape[2],
            ]
        )
        scale1 = scale1.to(self.device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > self.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        order = scores.argsort()[::-1]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_threshold)

        dets = dets[keep, :]
        landms = landms[keep]

        # dets: [바운딩 박스 좌표(xmin, ymin, xmax, ymax), 신뢰도 점수(score), 얼굴 랜드마크 좌표(x1, y1, x2, y2, x3, y3, x4, y4, x5, y5)]
        # dets.shape = (n, 15), n명, 15개의 정보
        dets = np.concatenate((dets, landms), axis=1)
        print("misc time: {:.4f}".format(time.time() - tic))

        return dets

    # 시각화 용도
    def draw_info(self, img, dets):
        for b in dets:
            if b[4] < self.vis_thres:  # score
                continue
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(
                img, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255)
            )

            # landms
            cv2.circle(img, (b[5], b[6]), 1, (0, 0, 255), 4)  # 왼쪽 눈
            cv2.circle(img, (b[7], b[8]), 1, (0, 255, 255), 4)  # 오른쪽 눈
            cv2.circle(img, (b[9], b[10]), 1, (255, 0, 255), 4)  # 코
            cv2.circle(img, (b[11], b[12]), 1, (0, 255, 0), 4)  # 오른쪽 입
            cv2.circle(img, (b[13], b[14]), 1, (255, 0, 0), 4)  # 왼쪽 입

        return img
    
def crop_alignment(img_array, b): # b: 얼굴 정보 데이터
    # crop
    facial_img = img_array[int(b[1]) : int(b[3]), int(b[0]) : int(b[2])]

    # alignment
    left_eye = (b[5], b[6])
    right_eye = (b[7], b[8])
    nose = (b[9], b[10])
    facial_img = alignment_procedure(facial_img, left_eye, right_eye, nose)

    return facial_img[:, :, ::-1] # RGB 색상 변환


def extract(img_path, trained_model, align=True, network="resnet50", cpu=True):
    """
    Face_Dectector(
        network="resnet50",
        trained_model="./weights/Resnet50_Final.pth",
        cpu=True
    )
    """
    
    img_path = img_path
    img_array = np.fromfile(img_path, np.uint8) # 한글 파일 이름 처리
    img_array = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    trained_model=trained_model
    network=network
    align=align
    cpu=cpu
    
    cfg = None
    if network == "mobile0.25":
        cfg = cfg_mnet
    elif network == "resnet50":
        cfg = cfg_re50

    torch.set_grad_enabled(False)

    detector = Face_Dectector(network=network, cpu=cpu, trained_model=trained_model)

    # net and model
    net = RetinaFace(cfg=cfg, phase="test")
    net = detector.load_model(net, detector.trained_model, detector.cpu)
    net.eval()
    print("Finished loading model!")

    cudnn.benchmark = True
    device = torch.device("cpu" if detector.cpu else "cuda")
    net = net.to(device)

    ##### 여기 전까지는 미리 실행해놓자 #####

    """
    detect(img_array, net)
    """
    dets = detector.detect(img_array, net)

    # crop & alignment
    resp = []
    for b in dets:
        resp.append(crop_alignment(img_array, b))
    return resp

def save_img(img, file_name, save_path):
    if not os.path.exists(save_path + "/results/"):
        os.makedirs(save_path + "/results/")
    name = save_path + "/results/" + file_name + ".jpg"
    cv2.imwrite(name, img)
    print('done')


"""
extract(img_path, trained_model, align=True, network="resnet50", cpu=True)
"""
if __name__ == "__main__":
    
    img_path = "../../celebrity_dataset/차은우12.jpg"

    trained_model = 'weights/Resnet50_Final.pth'
    network="resnet50"
    align=True
    cpu=True

    faces = extract(
        trained_model=trained_model,
        img_path=img_path,
        network=network,
        align=align,
        cpu=cpu,
    )

    for face in faces:
        plt.imshow(face)
        plt.axis("off")
        plt.show()


# detect 직접 실행
if __name__ == "__main__":
    
    ## 모델 로드 ##
    """
    Face_Dectector(
        network="resnet50",
        trained_model="./weights/Resnet50_Final.pth",
        cpu=True
    )
    """
    img_path = "../../celebrity_dataset/차은우12.jpg"
    img_array = np.fromfile(img_path, np.uint8) # 한글 파일 이름 처리
    img_array = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    trained_model = "./weights/Resnet50_Final.pth"
    network = "resnet50"
    cpu = True

    torch.set_grad_enabled(False)

    detector = Face_Dectector(
        trained_model=trained_model,
        network=network,
        cpu=cpu
    )

    # net and model
    net = RetinaFace(cfg=detector.cfg, phase="test")
    net = detector.load_model(net, detector.trained_model, detector.cpu)
    net.eval()
    print("Finished loading model!")

    cudnn.benchmark = True
    device = detector.device
    net = net.to(device)

    ##### 여기 전까지는 미리 실행해놓자 #####

    """
    detect(img_array, net)
    """
    dets = detector.detect(img_array, net)

    # 이미지 위에 bounding box와 랜드마크 표시하기
    detected_img = img_array.copy()
    detected_img = detector.draw_info(detected_img, dets=dets)
    

    # 수정된 이미지를 화면에 표시
    # 시각화 1
    # plt.imshow(detected_img[:, :, ::-1]) # RGB 색상 변환

    # 시각화 2
    cv2.imshow("image", detected_img)
    cv2.waitKey(0)

    # 시각화 3
    # from google.colab.patches import cv2_imshow
    # cv2_imshow(detected_img)

    # 저장 경로에 이미지 저장하기
    save_path = os.getcwd()
    save_img(detected_img, "detected", save_path)

    # crop & alignment
    faces = []
    for b in dets:
        faces.append(crop_alignment(img_array, b))