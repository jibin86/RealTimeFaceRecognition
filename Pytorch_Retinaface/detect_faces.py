import os
import cv2
import time
import torch
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import numpy as np

from models.retinaface import RetinaFace
from data import cfg_mnet, cfg_re50
from alignment import alignment_procedure
from layers.functions.prior_box import PriorBox
from utils.box_utils import decode, decode_landm
from utils.nms.py_cpu_nms import py_cpu_nms

# 이 코드는 RetinaFace를 사용하여 얼굴을 감지하고, 감지된 얼굴을 크롭하고 정렬한다.
class Face_Dectector:
    def __init__(
        self,
        trained_model="./weights/Resnet50_Final.pth",
        network="resnet50"
    ):
        self.trained_model = trained_model  # 학습된 모델의 경로
        self.network = network  # 사용할 네트워크의 이름 (resnet50, mobile0.25)

        self.confidence_threshold = 0.02  # 얼굴 검출에 사용할 confidence 임계값
        self.nms_threshold = 0.4  # 비최대 억제(non-maximum suppression)를 위한 임계값
        self.vis_thres = 0.5  # 시각화를 위한 임계값
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(self.device)
        self.cfg = self.set_cfg()
        self.net = self.set_net()

    def set_cfg(self):
        if self.network == "mobile0.25":
            return cfg_mnet
        elif self.network == "resnet50":
            return cfg_re50

    def set_net(self):
        net = RetinaFace(cfg=self.cfg, phase="test")
        net = self.load_model(net)
        net.eval()
        torch.set_grad_enabled(False)
        cudnn.benchmark = True
        net = net.to(self.device)
        print("Finished loading model!")
        return net
        
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


    def load_model(self, model):
        print("Loading pretrained model from {}".format(self.trained_model))
        pretrained_dict = torch.load(self.trained_model, map_location=self.device)
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = self.remove_prefix(
                pretrained_dict["state_dict"], "module."
            )
        else:
            pretrained_dict = self.remove_prefix(pretrained_dict, "module.")
        self.check_keys(model, pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)
        return model

    def detect(self, img_array):
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
        loc, conf, landms = self.net(img)  # forward pass
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
    def draw_info(self, detected_img, dets):
        for b in dets:
            if b[4] < self.vis_thres:  # score
                continue
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            cv2.rectangle(detected_img, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(
                detected_img, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255)
            )

            # landms
            cv2.circle(detected_img, (b[5], b[6]), 1, (0, 0, 255), 4)  # 왼쪽 눈
            cv2.circle(detected_img, (b[7], b[8]), 1, (0, 255, 255), 4)  # 오른쪽 눈
            cv2.circle(detected_img, (b[9], b[10]), 1, (255, 0, 255), 4)  # 코
            cv2.circle(detected_img, (b[11], b[12]), 1, (0, 255, 0), 4)  # 오른쪽 입
            cv2.circle(detected_img, (b[13], b[14]), 1, (255, 0, 0), 4)  # 왼쪽 입

        return detected_img
    
def crop_alignment(img_array, b, align=True): # b: 얼굴 정보 데이터
    # crop
    img_height, img_width = img_array.shape[:2]
    x1, y1, x2, y2 = max(int(b[0]), 0), max(int(b[1]), 0), min(int(b[2]), img_width), min(int(b[3]), img_height)
    facial_img = img_array[y1:y2, x1:x2]

    # alignment
    if align:
        left_eye = (b[5], b[6])
        right_eye = (b[7], b[8])
        nose = (b[9], b[10])
        facial_img = alignment_procedure(facial_img, left_eye, right_eye, nose)

    return facial_img[:, :, ::-1] # RGB 색상 변환


def extract(trained_model, img_path, network="resnet50", align=True):
    """
    Face_Dectector(
        trained_model="./weights/Resnet50_Final.pth",
        network="resnet50"
    )
    """
    trained_model=trained_model
    img_path = img_path
    network=network
    align=align

    img_array = handle_korean_file_name(img_path) # 한글 파일 이름 처리
    
    detector = Face_Dectector(
        trained_model=trained_model,
        network=network
    )

    ##### 여기 전까지는 미리 실행해놓자 #####

    """
    detect(img_array)
    """
    dets = detector.detect(img_array)

    # crop & alignment
    resp = []
    for b in dets:
        resp.append(crop_alignment(img_array, b, align=align))
    return resp

def save_img(img, file_name, save_path):
    if not os.path.exists(save_path + "/results/"):
        os.makedirs(save_path + "/results/")
    name = save_path + "/results/" + file_name + ".jpg"
    cv2.imwrite(name, img)
    print('save done')
    
def handle_korean_file_name(img_path):
    img_path = img_path
    img_array = np.fromfile(img_path, np.uint8) # 한글 파일 이름 처리
    img_array = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img_array


"""
extract(img_path, trained_model, align=True, network="resnet50")
"""
if __name__ == "__main__":
    
    img_path = "../celebrity_dataset/차은우12.jpg"

    trained_model = 'weights/Resnet50_Final.pth'
    network="resnet50"
    align=True

    faces = extract(
        trained_model=trained_model,
        img_path=img_path,
        network=network,
        align=align
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
        trained_model="./weights/Resnet50_Final.pth",
        network="resnet50"
    )
    """
    trained_model = "./weights/Resnet50_Final.pth"
    img_path = "../celebrity_dataset/차은우12.jpg"
    network = "resnet50"
    align=True
    
    img_array = handle_korean_file_name(img_path) # 한글 파일 이름 처리

    detector = Face_Dectector(
        trained_model=trained_model,
        network=network
    )

    ##### 여기 전까지는 미리 실행해놓자 #####

    """
    detect(img_array, net)
    """
    dets = detector.detect(img_array)

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
        faces.append(crop_alignment(img_array, b, align=align))