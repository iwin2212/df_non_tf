from flask import request, render_template, Blueprint, Response
from importlib import import_module
import os
from flask import render_template, Response, jsonify
from utils import get_new_brand, check_file_exist, get_list_unknown_img, rename
from view.utils.data import detect_face
from ast import literal_eval
if os.environ.get('CAMERA'):
    Camera = import_module('camera_' + os.environ['CAMERA']).Camera
else:
    from view.camera_flask.camera_opencv import Camera
import cv2
import numpy as np
import time
from view.utils.stream import draw_retangle
from utils import predict_snapshot
import logging
mod = Blueprint('take_shot', __name__)


@mod.route("/take_shots")
def take_shots():
    return render_template('./take_shot.html')


def generate(camera):
    """Video streaming generator function."""
    while True:
        frame = np.asarray(bytearray(camera.get_frame()), dtype="uint8")
        img = cv2.imdecode(frame, cv2.IMREAD_COLOR)

        try:
            image = draw_retangle(img=img)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpeg', image)[1].tobytes() + b'\r\n')
        except Exception as error:
            logging.warning('Error in generate(camera): {}'.format(error))
            return ""

@mod.route('/video_feed')
def video_feed():
    return Response(generate(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@mod.route('/snap_shot',  methods=['POST'])
def snap_shot():
    img = Camera().get_frame()
    image = np.asarray(bytearray(img), dtype="uint8")
    image = detect_face(cv2.imdecode(image, cv2.IMREAD_COLOR))
    # print(image.shape)
    new_shot = get_new_brand()
    cv2.imwrite(new_shot, image)
    return jsonify(result=check_file_exist(new_shot))


@mod.route('/snapshot',  methods=['POST'])
def snapshot():
    link_list = []
    while(len(link_list) <= 10):
        img = Camera().get_frame()
        image = np.asarray(bytearray(img), dtype="uint8")
        prev = time.time()
        try:
            image = detect_face(cv2.imdecode(image, cv2.IMREAD_COLOR))
            new_shot = get_new_brand()
            # cv2.imshow(new_shot, image)
            cv2.imwrite(new_shot, image)
            link_list.append(new_shot)
            time.sleep(0.1)
        except Exception as error:
            logging.warning("Error: {}".format(error))
            now = time.time()

            if ((now-prev) > 10):
                logging.warning("Exceeded the time limit : {} > 10 (s)".format((now-prev)))
                return {"result": "Connection denied", "reason": "Exceeded the time limit", "file_list": link_list}
            continue
    cv2.destroyAllWindows()
    identity, duration = predict_snapshot(link_list)
    return {"result": "Success", "identity": identity, "duration": duration}


@mod.route('/brandname')
def brandname():
    list_unknown_img = get_list_unknown_img()
    return render_template('./brandname.html', list_unknown_img=list_unknown_img)


@mod.route("/readdress",  methods=['POST'])
def readdress():
    rename_list = request.args.get("rename_list")
    data = literal_eval(rename_list)
    for key, value in data.items():
        rename(key, value)
    return jsonify()
