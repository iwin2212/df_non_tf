import os
import cv2
from view.camera_flask.base_camera import BaseCamera
from const import video_source


class Camera(BaseCamera):
    video_source = video_source
    def __init__(self):
        if os.environ.get('OPENCV_CAMERA_SOURCE'):
            Camera.set_video_source(int(os.environ['OPENCV_CAMERA_SOURCE']))
        super(Camera, self).__init__()

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        camera = cv2.VideoCapture(Camera.video_source)
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        while True:
            # read current frame
            _, img = camera.read()
            resolution_x = img.shape[1]
            resolution_y = img.shape[0]
            while(img.shape[0]>600):
                img = cv2.resize(img, (int(resolution_x/2),int(resolution_y/2)))
            # encode as a jpeg image and return it
            yield cv2.imencode('.jpg', img)[1].tobytes()
