from custom_deepface.deepface.commons import functions, distance as dst
from view.utils.lite_predict import predict_tfmodel
import os
import numpy as np
import pandas as pd
import cv2
import time
import os
from const import embedding_path, input_shape_x, input_shape_y, text_color, distance_metric
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def preprocess(model_name='Facenet', detector_backend='opencv'):
    functions.initialize_detector(detector_backend)
    threshold = dst.findThreshold(model_name, distance_metric)-0.1

    # loading database
    embeddings = np.load(embedding_path, allow_pickle=True)
    df = pd.DataFrame(embeddings, columns=['employee', 'embedding'])
    df['distance_metric'] = distance_metric
    # -----------------------

    pivot_img_size = 112  # face recognition result image

    # -----------------------
    opencv_path = functions.get_opencv_path()
    face_detector_path = opencv_path+"haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(face_detector_path)
    # -----------------------
    return face_cascade, threshold



