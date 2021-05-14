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
import pandas as pd
from const import embedding_path, distance_metric
from custom_deepface.deepface.commons import functions
import logging
mod = Blueprint('live_demo', __name__)


def gen(camera):
    # loading database
    embeddings = np.load(embedding_path, allow_pickle=True)
    df = pd.DataFrame(embeddings, columns=['employee', 'embedding'])
    df['distance_metric'] = distance_metric
    # -----------------------
    opencv_path = functions.get_opencv_path()
    face_detector_path = opencv_path+"haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(face_detector_path)
    while True:
        pTime = time.time()
        frame = np.asarray(bytearray(camera.get_frame()), dtype="uint8")
        img = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        try:
            image = preprocess(img, face_cascade, df)

            cTime = time.time()
            # print("fps: {}".format(1/(cTime-pTime)))
            # print("duration: {}".format((cTime-pTime)))
            fps = 1/(cTime - pTime)
            pTime = cTime
            cv2.putText(image, str(int(fps)), (60, 40),
                        cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpeg', image)[1].tobytes() + b'\r\n')
        except Exception as error:
            logging.warning('Error in gen(camera): {}'.format(error))
            return ""


@mod.route('/live_stream')
def live_stream():
    time.sleep(0.01)
    return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@mod.route("/live_demo")
def live_demo():
    return render_template('./live_demo.html')
