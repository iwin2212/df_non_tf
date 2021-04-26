from numpy.lib.utils import source
from custom_deepface.deepface.commons import functions, distance as dst
from lite_predict import predict_tfmodel
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2
import time
import re
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



cap = cv2.VideoCapture(source)  # webcam
ret, img = cap.read()
custom_face = base_img[y:y+h, x:x+w]
custom_face = functions.preprocess_face(img=custom_face, target_size=(
                            input_shape_y, input_shape_x), enforce_detection=False)
img1_representation = predict_tfmodel(custom_face)[0, :]
def findDistance(row):
    distance_metric = row['distance_metric']
    img2_representation = row['embedding']

    distance = 1000  # initialize very large value
    if distance_metric == 'cosine':
        distance = dst.findCosineDistance(
            img1_representation, img2_representation)
    elif distance_metric == 'euclidean':
        distance = dst.findEuclideanDistance(
            img1_representation, img2_representation)
    elif distance_metric == 'euclidean_l2':
        distance = dst.findEuclideanDistance(dst.l2_normalize(
            img1_representation), dst.l2_normalize(img2_representation))

    return distance

df['distance'] = df.apply(findDistance, axis=1)