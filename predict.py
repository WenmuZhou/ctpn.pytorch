# -*- coding: utf-8 -*-
# @Time    : 1/4/19 11:14 AM
# @Author  : zhoujun
import torch
from torchvision import transforms
import os
import cv2
import time
import numpy as np
import torch.nn.functional as F
from utils.bbox_utils import gen_anchor, bbox_transfor_inv, clip_box, filter_bbox, nms
from utils.TextProposalConnector import TextProposalConnectorOriented


def resize1(im: np.ndarray, min_len: int = 600, max_len: int = 1200) -> np.ndarray:
    img_size = im.shape

    # 图片缩放
    im_size_min = np.min(img_size[0:2])
    im_size_max = np.max(img_size[0:2])
    # 短边缩放到600 并且保证长边不超过1200
    im_scale = float(min_len) / float(im_size_min)
    if np.round(im_scale * im_size_max) > max_len:
        im_scale = float(max_len) / float(im_size_max)
    new_h = int(img_size[0] * im_scale)
    new_w = int(img_size[1] * im_scale)
    # 保证边长能被16整除
    new_h = new_h if new_h // 16 == 0 else (new_h // 16 + 1) * 16
    new_w = new_w if new_w // 16 == 0 else (new_w // 16 + 1) * 16

    re_im = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return re_im


class Pytorch_model:
    def __init__(self, model_path, net, gpu_id=None):
        '''
        初始化pytorch模型
        :param model_path: 模型地址(可以是模型的参数或者参数和计算图一起保存的文件)
        :param net: 网络计算图，如果在model_path中指定的是参数的保存路径，则需要给出网络的计算图
        :param img_channel: 图像的通道数: 1,3
        :param gpu_id: 在哪一块gpu上运行
        '''
        if gpu_id is not None and isinstance(gpu_id, int) and torch.cuda.is_available():
            self.device = torch.device("cuda:{}".format(gpu_id))
        else:
            self.device = torch.device("cpu")

        self.net = torch.load(model_path, map_location=self.device)['state_dict']
        print('device:', self.device)

        if net is not None:
            # 如果网络计算图和参数是分开保存的，就执行参数加载
            net = net.to(self.device)
            try:
                sk = {}
                for k in self.net:
                    sk[k[7:]] = self.net[k]
                net.load_state_dict(sk)
            except:
                net.load_state_dict(self.net)
            self.net = net
            print('load model')
        self.net.eval()

    def predict(self, img: str):
        '''
        对传入的图像进行预测，支持图像地址,opecv 读取图片，偏慢
        :param img: 图像地址
        :param is_numpy:
        :return:
        '''
        prob_thresh = 0.5
        assert os.path.exists(img), 'file is not exists'
        image = cv2.imread(img_path)
        h, w = image.shape[:2]
        image = resize1(image, min_len=600, max_len=1200)
        new_h, new_w = image.shape[:2]
        # image = image.astype(np.float32) - config.IMAGE_MEAN
        tensor = transforms.ToTensor()(image)
        tensor = tensor.unsqueeze_(0)

        with torch.no_grad():
            start = time.time()
            tensor = tensor.to(self.device)
            cls, regr = self.net(tensor)
            cls_prob = F.softmax(cls, dim=-1).cpu().numpy()
            regr = regr.cpu().numpy()
            anchor = gen_anchor((int(new_h / 16), int(new_w / 16)), 16)
            bbox = bbox_transfor_inv(anchor, regr)
            bbox = clip_box(bbox, [new_h, new_w])

            fg = np.where(cls_prob[0, :, 1] > prob_thresh)[0]
            select_anchor = bbox[fg, :]
            select_score = cls_prob[0, fg, 1]
            select_anchor = select_anchor.astype(np.int32)

            keep_index = filter_bbox(select_anchor, 16)

            # nsm
            select_anchor = select_anchor[keep_index]
            select_score = select_score[keep_index]
            select_score = np.reshape(select_score, (select_score.shape[0], 1))
            nmsbox = np.hstack((select_anchor, select_score))
            keep = nms(nmsbox, 0.3)
            select_anchor = select_anchor[keep]
            select_score = select_score[keep]

            textConn = TextProposalConnectorOriented()
            scale = (new_w / w, new_h / h)
            text = textConn.get_text_lines(select_anchor, select_score, [new_h, new_w], scale)
            # print(scale)
            # preds, boxes_list = decode(preds,num_pred=-1)
            t = time.time() - start
        return text, t


def _get_annotation(label_path):
    boxes = []
    with open(label_path, encoding='utf-8', mode='r') as f:
        for line in f.readlines():
            params = line.strip().strip('\ufeff').strip('\xef\xbb\xbf').split(',')
            try:
                label = params[8]
                if label == '*' or label == '###':
                    continue
                x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, params[:8]))
                boxes.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
            except:
                print('load label failed on {}'.format(label_path))
    return np.array(boxes, dtype=np.float32)


if __name__ == '__main__':
    import config
    from model import CTPN_Model
    import matplotlib.pyplot as plt
    from utils.utils import show_img, draw_bbox, draw_anchor

    # os.environ['CUDA_VISIBLE_DEVICES'] = str('2')

    model_path = 'output/PSENet_298_loss0.001575.pth'

    img_id = 1
    img_path = 'D:/zj/dataset/ICD15/train/imgs/img_{}.jpg'.format(img_id)
    # img_path = '0.jpg'
    label_path = 'D:/zj/dataset/ICD15/train/gt/gt_img_{}.txt'.format(img_id)
    label = _get_annotation(label_path)

    # img_path = '/data1/gcz/拍照清单数据集_备份/87436979.jpg'
    # 初始化网络
    net = CTPN_Model(pretrained=False)
    model = Pytorch_model(model_path, net=net, gpu_id=None)
    # for i in range(100):
    #     model.predict(img_path)
    boxes_list, t = model.predict(img_path)
    print(boxes_list)
    img = draw_bbox(img_path, boxes_list, color=(0, 0, 255))
    cv2.imwrite('result.jpg', img)
    # img = draw_bbox(img, label,color=(0,0,255))
    show_img(img[:, :, ::-1], color=True)

    plt.show()
