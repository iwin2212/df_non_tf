from flask import request, render_template, Blueprint, Response
from importlib import import_module
import os
from flask import render_template, Response, jsonify
from const import snap_path
from utils import get_new_brand
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
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    cv2.imwrite(get_new_brand(), image)
    return jsonify()
