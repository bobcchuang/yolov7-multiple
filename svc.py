
import time
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import base64
import threading
import uvicorn
from AI_model import *

# fastapi package
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles
import gc
from pydantic import BaseModel
from typing import Optional
import torch


lock = threading.Lock()

model = None
device = None


def clean_model():
    global model, device
    if model is not None:
        try:
            del model
            torch.cuda.memory_cached()
            torch.cuda.empty_cache()
            gc.collect()
            model = None
        except:
            print("fail to delete model")
    pass


def initialize_model():
    global model, device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('device', device)
    model = AI_model()


initialize_model()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/docs2", include_in_schema=False)
async def custom_swagger_ui_html():
    """
    For local js, css swagger in AUO
    :return:
    """
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="/static/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui.css",
    )


@app.get("/")
def HelloWorld():
    return {"Hello": "World"}


class modelBase(BaseModel):
    model_name: str

    # 以下兩個 functions 請盡可能不要更動-----------------------
    @classmethod
    def __get_validators__(cls):
        yield cls.validate_to_json

    @classmethod
    def validate_to_json(cls, value):
        if isinstance(value, str):
            return cls(**json.loads(value))
        return value


@app.post("/predict/")
def predict(data: modelBase = Form(...), file: UploadFile = File(...)):
    t0 = time.time()
    # image_rgb = bytes_to_rgbimage(file)
    image_rgb = bytes_to_rgbimage(file.file.read())
    ave_hsv = get_hsv_from_img_rgb(image_rgb)

    return_json = {}
    global model
    # get info
    model_name = data.model_name
    if not model_name:
        model_name = "default"
    if model_name not in model.get_Model_name():
        return_json = alarm_code(return_json, '0001')
        label_info = []
    else:
        label_info = model.Predict(image_rgb, model_name)
        return_json = alarm_code(return_json, '0000')
    return_json['label_info'] = label_info

    names = model.get_label_list(model_name)
    t1 = time.time()
    f_fps = 1.0 / (t1 - t0)
    return_json["class"] = names
    return_json['fps'] = f_fps
    return_json['hsv'] = ave_hsv
    return return_json


class uploadBase(BaseModel):
    filename: str
    model_name: str

    # 以下兩個 functions 請盡可能不要更動-----------------------
    @classmethod
    def __get_validators__(cls):
        yield cls.validate_to_json

    @classmethod
    def validate_to_json(cls, value):
        if isinstance(value, str):
            return cls(**json.loads(value))
        return value


@app.post("/save_file/")
def save_file(data: uploadBase = Form(...), file: UploadFile = File(...)):
    t0 = time.time()

    # get info
    filename = data.filename
    model_name = data.model_name
    file_dir = os.path.join('/usr/src/node-red/.auo/yolov7-torch-svc/model_data/'+model_name)
    # mkdir
    os.makedirs(file_dir, exist_ok=True)

    tmp_file = file.file.read()

    file_path = os.path.join(file_dir, filename)
    with open(file_path, 'wb') as file:
        file.write(tmp_file)
    file.close()

    t1 = time.time()
    fps = 1.0 / (t1 - t0)

    return {"return_text": "success", 'fps': fps}


@app.post("/delete_model/")
def delete_model(data: modelBase = Form(...)):
    t0 = time.time()

    root_path = '/usr/src/node-red/.auo/yolov7-torch-svc/model_data/'
    # get info
    filename = data.model_name

    model_folder_list = os.listdir(root_path)
    if not (filename in model_folder_list):
        return {"return_text": "model does not exist"}
    else:
        os.system("rm -r " + os.path.join(root_path, filename))
    t1 = time.time()
    fps = 1.0 / (t1 - t0)

    return {"return_text": "success to delete", 'fps': fps}


@app.get("/reload_model/")
def reload_model():
    t0 = time.time()
    clean_model()
    initialize_model()

    t1 = time.time()
    fps = 1.0 / (t1 - t0)
    return {"return_text": "success to start", 'fps': fps}


@app.post("/get_class/")
def get_class(data: modelBase = Form(...)):
    t0 = time.time()
    model_name = data.model_name

    global model

    names = model.get_label_list(model_name)
    t1 = time.time()
    fps = 1.0 / (t1 - t0)
    return {"class": names, 'fps': fps}


color_list = [(29, 178, 255),
              (168, 153, 44),
              (49, 210, 207),
              (243, 126, 162),
              (89, 190, 22),
              (207, 190, 23),
              (99, 112, 171),
              (194, 119, 227),
              (180, 119, 31),
              (40, 39, 214)]

_cfg = {
    "line_width": 3,
    "label_font_size": 1.5,
    "text_font_size": 0.5,
}


class bboxBase(BaseModel):
    detections: list
    class_names: list

    # 以下兩個 functions 請盡可能不要更動-----------------------
    @classmethod
    def __get_validators__(cls):
        yield cls.validate_to_json

    @classmethod
    def validate_to_json(cls, value):
        if isinstance(value, str):
            return cls(**json.loads(value))
        return value


@app.post("/draw_bbox/")
def draw_bbox(parameter: bboxBase = Form(...), file: UploadFile = File(...)):
    t0 = time.time()

    # get image
    cv2_img = bytes_to_cv2image(file.file.read())

    # get info
    object_list = parameter.detections
    label_list = parameter.class_names

    if len(object_list) == 0:
        t1 = time.time()
        f_fps = 1.0 / (t1 - t0)
        output_dict = {"img_base64": cv2image_to_base64(cv2_img), 'fps': f_fps}
        return output_dict

    # draw
    for object in object_list:
        box_xmin = int(object[1][0][1])
        box_ymin = int(object[1][0][0])
        box_xmax = int(object[1][2][1])
        box_ymax = int(object[1][2][0])
        # print("-"*30)
        # print(box_xmin, box_ymin, box_xmax, box_ymax)
        # print(cv2_img.shape)
        confidence = object[2]
        label = object[0]
        label_index = label_list.index(label)
        color = color_list[int(label_index % len(color_list))]

        auoDrawBbox(cv2_img, (box_xmin, box_ymin), (box_xmax, box_ymax), color, line_width=_cfg['line_width'])
        od_str = label + '  ' + str(confidence)[:4]
        text2image(cv2_img, (box_xmin, box_ymin), od_str, font_scale=_cfg['label_font_size'],
                   font_color=(255, 255, 255), font_face=cv2.FONT_HERSHEY_DUPLEX,
                   background_color=color)

    t1 = time.time()
    f_fps = 1.0 / (t1 - t0)
    output_dict = {"img_base64": cv2image_to_base64(cv2_img), 'fps': f_fps}
    return output_dict


class polyBase(BaseModel):
    points: list = [[10, 90], [20, 90], [20, 100], [10, 100]]
    color: list = [0, 255, 0]
    thickness: int = 2
    is_closed: bool = True

    # 以下兩個 functions 請盡可能不要更動-----------------------
    @classmethod
    def __get_validators__(cls):
        yield cls.validate_to_json

    @classmethod
    def validate_to_json(cls, value):
        if isinstance(value, str):
            return cls(**json.loads(value))
        return value


@app.post("/draw_poly/")
def draw_poly(parameter: polyBase = Form(...), file: UploadFile = File(...)):
    t0 = time.time()

    # get image
    cv2_img = bytes_to_cv2image(file.file.read())

    # get info
    object_list = parameter.points

    if len(object_list) == 0:
        t1 = time.time()
        f_fps = 1.0 / (t1 - t0)
        output_dict = {"img_base64": cv2image_to_base64(cv2_img), 'fps': f_fps}
        return output_dict

    pts = np.array(object_list)
    pts = pts.reshape((-1, 1, 2))

    cv2_img = cv2.polylines(cv2_img, [pts], parameter.is_closed, parameter.color, parameter.thickness)

    t1 = time.time()
    f_fps = 1.0 / (t1 - t0)
    output_dict = {"img_base64": cv2image_to_base64(cv2_img), 'fps': f_fps}
    return output_dict


def get_hsv_from_img_rgb(image_rgb):
    image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    ave_hsv = {'h': round(image_hsv[:, :, 0].mean(), 3), 's': round(image_hsv[:, :, 1].mean(), 3),
               'v': round(image_hsv[:, :, 2].mean(), 3)}
    return ave_hsv


def bytes_to_cv2image(imgdata):
    cv2img = cv2.cvtColor(np.array(Image.open(BytesIO(imgdata))), cv2.COLOR_RGB2BGR)
    return cv2img


def bytes_to_rgbimage(imgdata):
    img = np.array(Image.open(BytesIO(imgdata)))
    return img


def cv2image_to_base64(cv2img):
    retval, buffer_img = cv2.imencode('.jpg', cv2img)
    base64_str = base64.b64encode(buffer_img)
    str_a = base64_str.decode('utf-8')
    return str_a


def auoDrawBbox(image_bgr, bbox_min, bbox_max, line_color, line_width=2):
    cv2.rectangle(image_bgr, bbox_min, bbox_max, line_color, line_width)


def text2image(image, xy, label, font_scale=0.5, thickness=1, font_color=(0, 0, 0),
               font_face=cv2.FONT_HERSHEY_COMPLEX, background_color=(0, 255, 0)):
    label_size = cv2.getTextSize(label, font_face, font_scale, thickness)
    _x1 = xy[0]  # bottomleft x of text
    _y1 = xy[1]  # bottomleft y of text
    _x2 = xy[0] + label_size[0][0]  # topright x of text
    _y2 = xy[1] - label_size[0][1]  # topright y of text
    cv2.rectangle(image, (_x1, _y1), (_x2, _y2), background_color, cv2.FILLED)  # text background
    cv2.putText(image, label, (_x1, _y1), font_face, font_scale, font_color,
                thickness, cv2.LINE_AA)


def alarm_code(json, code):
    return_str = ''

    if code == '0000':
        return_str = 'success!'
    elif code == '0001':
        return_str = 'Model name not exist!'
    elif code == '0002':
        return_str = 'Get model list fail!'
    elif code == '0003':
        return_str = 'Get label list fail!'

    json["return_code"] = code
    json['return_text'] = return_str
    return json


def base64tocv2(b64_string):
    img_bytes = base64.b64decode(b64_string.encode('utf-8'))
    nparr = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image


def get_hsv_from_img_bgr(image_bgr):
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    ave_hsv = {'h': round(image_hsv[:, :, 0].mean(), 3), 's': round(image_hsv[:, :, 1].mean(), 3),
               'v': round(image_hsv[:, :, 2].mean(), 3)}
    return ave_hsv


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5109))
    uvicorn.run(app, log_level='info', host='0.0.0.0', port=port)
