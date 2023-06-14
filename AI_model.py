#!/usr/bin/env python
# coding: utf-8

import os
import cv2
import json
from yolov7 import YOLO


class AI_model():
    def __init__(self):
        model_path = '/usr/src/node-red/.auo/yolov7-torch-svc/model_data/'
        os.makedirs(model_path, exist_ok=True)
        DEFAULT_WEIGHT_FILE_PATH = os.environ.get("DEFAULT_WEIGHT", os.path.join('weight', 'yolov7.pt'))
        self.model_name_list = []
        self.yolo_list = []
        self.model_dict = None
        model_fold = os.listdir(model_path)
        if len(model_fold) > 0:
            for i in model_fold:
                with open('config_temp.json') as f:
                    config = json.load(f)
                    config['model_path'] = model_path + i + '/yolov7.pt'
                    yolo = YOLO(config)
                    self.yolo_list.append(yolo)
                    self.model_name_list.append(i)
        else:
            with open('config_temp.json') as f:
                config = json.load(f)
                config['model_path'] = DEFAULT_WEIGHT_FILE_PATH
                yolo = YOLO(config)
                self.yolo_list.append(yolo)
                self.model_name_list.append("default")
        self.model_dict = dict(zip(self.model_name_list, list(range(len(self.model_name_list)))))

    def Predict(self, frame, model_name):
        # height, width = frame.shape[:2]
        result = self.yolo_list[self.model_dict[model_name]].detect_image(frame.copy())
        class_names = self.yolo_list[self.model_dict[model_name]].get_class()
        result_ = []  # modify the format
        if len(result) > 0:
            for obj in result:
                x1 = obj[0]
                y1 = obj[1]
                x2 = obj[2]
                y2 = obj[3]
                conf = round(obj[4], 3)
                label_index = int(obj[5])
                # tmp = [class_names[label_index], [[x1, y1], [x1, y2], [x2, y2], [x2, y1]], conf]
                tmp = [class_names[label_index], [[y1, x1], [y1, x2], [y2, x2], [y2, x1]], conf]
                result_.append(tmp)
        return result_

    def get_Model_name(self):
        return list(self.model_dict)

    def get_label_list(self, model_name):
        try:
            return self.yolo_list[self.model_dict[model_name]].get_class()
        except:
            return []


if __name__ == "__main__":
    temp = AI_model()
    print(temp.Predict(cv2.imread('test.jpg'), 'PM'))

