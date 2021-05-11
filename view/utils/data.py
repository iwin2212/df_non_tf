import numpy as np
from custom_deepface.deepface.commons import functions, distance as dst
from view.test.lite_predict import predict_tfmodel
import cv2
import os
from const import embedding_path, input_shape_x, input_shape_y


def detect_face(img_path, enforce_detection=True, detector_backend='opencv'):
    cut_img, img, region = functions.preprocess_face(img=img_path, target_size=(
        input_shape_y, input_shape_x), enforce_detection=enforce_detection, detector_backend=detector_backend, return_region=True)
    return cut_img


def represent(img_path, enforce_detection=True, detector_backend='opencv', grayscale=False):
    img = functions.load_image(img_path)
    # detect and align
    # img = functions.preprocess_face(img=img_path, target_size=(
    #     input_shape_y, input_shape_x), enforce_detection=enforce_detection, detector_backend=detector_backend)
    # post-processing

    # post-processing
    if grayscale == True:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_pixels = get_face_pixels(img)
    embedding = predict_tfmodel(img_pixels)[0].tolist()
    return embedding


def get_face_pixels(img):
    img = cv2.resize(img, (input_shape_x, input_shape_y))
    img_pixels = np.array(img, dtype=np.float32)
    img_pixels = np.expand_dims(img_pixels, axis=0)
    img_pixels /= 255  # normalize input in [0, 1]
    return img_pixels


def add_img2db(img_path, label: str):
    if os.path.isfile(embedding_path):
        embeddings = np.load(embedding_path, allow_pickle=True)
    else:
        embeddings = np.zeros(shape=(0, 2))
    embedding = np.array(represent(img_path))
    new_embeddings = np.concatenate(
        [embeddings, np.array([label, embedding]).reshape(1, 2)], axis=0)
    np.save(embedding_path, new_embeddings)


def add_embedding2db(embedding, label: str):
    if os.path.isfile(embedding_path):
        embeddings = np.load(embedding_path, allow_pickle=True)
    else:
        embeddings = np.zeros(shape=(0, 2))
    embedding = np.array(embedding)
    new_embeddings = np.concatenate(
        [embeddings, np.array([label, embedding]).reshape(1, 2)], axis=0)
    np.save(embedding_path, new_embeddings)
