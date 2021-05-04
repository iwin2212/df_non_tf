from flask import request, render_template, Blueprint, Response
from importlib import import_module
import os
from flask import render_template, Response, jsonify
from const import snap_path
from utils import get_new_brand, check_file_exist, get_list_unknown_img, rename
from view.data import detect_face
mod = Blueprint('take_shot', __name__)
if os.environ.get('CAMERA'):
    Camera = import_module('camera_' + os.environ['CAMERA']).Camera
else:
    from view.camera_flask.camera_opencv import Camera


@mod.route("/take_shots")
def take_shots():
    return render_template('./take_shot.html')


def generate(camera):
    """Video streaming generator function."""
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@mod.route('/video_feed')
def video_feed():
    return Response(generate(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@mod.route('/snap_shot',  methods=['POST'])
def snap_shot():
    import cv2
    import numpy as np
    img = Camera().get_frame()
    image = np.asarray(bytearray(img), dtype="uint8")
    image = detect_face(cv2.imdecode(image, cv2.IMREAD_COLOR))
    # print(image.shape)
    new_shot = get_new_brand()
    cv2.imwrite(new_shot, image)
    return jsonify(result = check_file_exist(new_shot))


@mod.route('/brandname',  methods=['GET', 'POST'])
def brandname():
    if request.method == 'POST':
        rename(request.form)
        # return render_template("/index.html")
        list_unknown_img = get_list_unknown_img()
        return render_template('./brandname.html', list_unknown_img=list_unknown_img)
    else:
        list_unknown_img = get_list_unknown_img()
        return render_template('./brandname.html', list_unknown_img=list_unknown_img)