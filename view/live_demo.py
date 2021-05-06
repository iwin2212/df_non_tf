from flask import render_template, Blueprint, Response
import os
from importlib import import_module
if os.environ.get('CAMERA'):
    Camera = import_module('camera_' + os.environ['CAMERA']).Camera
else:
    from view.camera_flask.camera_opencv import Camera
import numpy as np
import cv2
from view.utils.stream import preprocess
mod = Blueprint('live_demo', __name__)


def generate(camera):
	face_cascade, threshold = preprocess()
	while True:
		frame = np.asarray(bytearray(camera.get_frame()), dtype="uint8")
		img = cv2.imdecode(frame, cv2.IMREAD_COLOR)
		faces = face_cascade.detectMultiScale(img,  1.3, 5)
		if len(faces) == 0:
			face_included_frames = 0
		yield (b'--frame\r\n'
				b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')


@mod.route('/video_feed')
def video_feed():
    return Response(generate(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@mod.route("/live_demo")
def live_demo():
	return render_template('./live_demo.html')

