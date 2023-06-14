# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 16:07:58 2021

@author: 2102066
"""

import os
import time
import numpy as np
import threading
import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression

conf_thres = 0.25
iou_thres = 0.45
print('conf_thres', conf_thres)
print('iou_thres', iou_thres)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
lock = threading.Lock()


class YOLO(object):
    _defaults = {
        "model_path": './model_data/yolov7.pt',
        "score": 0.5,
        "iou": 0.4,
        "model_image_size": (416, 416),
        "gpu_num": 1
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, kwargs):
        self.__dict__.update(kwargs)  # and update with user overrides
        self.model_image_size = tuple(self.model_image_size)
        self.model = self.load_model()

    def get_time_stamp(self):
        ct = time.time()
        local_time = time.localtime(ct)
        data_head = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
        data_secs = (ct - int(ct)) * 1000
        time_stamp = "%s.%03d" % (data_head, data_secs)
        stamp = ("".join(time_stamp.split()[0].split("-")) + "".join(time_stamp.split()[1].split(":"))).replace('.', '')
        # print(stamp)
        return str(stamp)

    def get_class(self):
        class_names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names
        return class_names

    def load_model(self):
        model_path = os.path.expanduser(self.model_path)
        model = attempt_load(model_path, map_location=device)
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        print('names', names)
        model = model.autoshape()  # for file/URI/PIL/cv2/np inputs and NMS
        model.to(device)
        return model

    def detect_image(self, image):
        image_rgb = image
        imgsize = list(image_rgb.shape)[:2]
        image_rgb = image_rgb[:int(imgsize[0] / 32) * 32, :int(imgsize[1] / 32) * 32]
        image_rgb = np.transpose(image_rgb, (2, 0, 1))
        # names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        with lock:
            image_rgb = torch.from_numpy(image_rgb).to(device)
            image_rgb = image_rgb.float()  # uint8 to fp32
            image_rgb /= 255.0  # 0 - 255 to 0.0 - 1.0
            if image_rgb.ndimension() == 3:
                image_rgb = image_rgb.unsqueeze(0)
            pred = self.model(image_rgb)[0]
            pred = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_thres)[0].cpu().numpy().tolist()
        return pred

    def close_session(self):
        self.sess.close()

