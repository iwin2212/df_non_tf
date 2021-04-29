from flask import request, render_template, Blueprint, Response
from importlib import import_module
import os
from flask import Flask, render_template, Response
mod = Blueprint('take_shot', __name__)
if os.environ.get('CAMERA'):
    Camera = import_module('camera_' + os.environ['CAMERA']).Camera
else:
    from view.camera_flask.camera_opencv import Camera


@mod.route("/take_shots", methods=['GET', 'POST'])
def take_shots():
    return render_template('./take_shot.html')


def gen(camera):
    """Video streaming generator function."""
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@mod.route('/video_feed')
def video_feed():
    return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
