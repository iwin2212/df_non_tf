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
import time
mod = Blueprint('live_demo', __name__)
prev = 0

def gen(camera):
	while True:
		frame = np.asarray(bytearray(camera.get_frame()), dtype="uint8")
		img = cv2.imdecode(frame, cv2.IMREAD_COLOR)
		try:
			image = preprocess(img=img, frame_rate=3, prev=prev)
			yield (b'--frame\r\n'
				b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpeg', image)[1].tobytes() + b'\r\n')
		except Exception as error:
			print('Error in gen(camera): {}'.format(error))
			return ""
		


@mod.route('/live_stream')
def live_stream():
    time.sleep(0.01)
    return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@mod.route("/live_demo")
def live_demo():
	return render_template('./live_demo.html')

