from pathlib import Path
import io
from const import snap_path, ROOT_DIR, video_source, img_path, result_path, w_min
import os
from view.utils.data import add_img2db
import yaml
import time
import cv2
from custom_deepface.deepface.commons import distance as dst
import pandas as pd
from view.utils.data import get_face_pixels
from view.utils.lite_predict import predict_tfmodel
from const import input_shape, embedding_path, distance_metric, input_shape_size
import numpy as np
import json
from custom_deepface.deepface.commons import functions


def get_new_brand():
    return os.path.join(snap_path, str(int(time.time())) + '.jpg')


def check_file_exist(file_path):
    return Path(file_path).is_file()


def get_list_unknown_img():
    list_path = []
    for root, dirs, files in os.walk(snap_path, topdown=False):
        for name in files:
            path = "./" + os.path.join(root, name)
            list_path.append(path)
    return list_path


def get_current_folder():
    return os.path.dirname(os.path.realpath(__file__))


def rename(key, val):
    if (val == 'delete'):
        os.remove(key)
    else:
        try:
            add_img2db(key, val)
            os.remove(key)
        except Exception as error:
            print("Error in rename: {}".format(error))


def yaml2dict(filename):
    f = open(filename, 'r', encoding='utf8')
    res = yaml.load(f, Loader=yaml.FullLoader)
    f.close()
    return res


def dict2yaml(dict_, filename):
    with io.open(filename, 'w', encoding='utf-8') as outfile:
        yaml.dump(dict_, outfile, default_flow_style=False, allow_unicode=True)


def data2json(data, json_file_path):
    with open(json_file_path, 'w') as outfile:
        json.dump(data, outfile, indent=4)


def json2data(json_file_path):
    with open(json_file_path) as json_file:
        data = json.load(json_file)
    return data


def get_token():
    secret_file = os.path.join(ROOT_DIR, 'secrets.yaml')
    data = yaml2dict(secret_file)
    authen_code = data['token']
    return authen_code


def destroy_camera():
    cv2.VideoCapture(video_source).release()


def get_list_img_path(file_path):
    for root, dirs, files in os.walk(file_path, topdown=False):
        return [os.path.join(root, img)
                for img in files if (".jpg") in img]


def predict_snapshot(img_path=img_path):
    pTime = time.time()
    # case 1: snapshot from home assistant
    if ("/usr/share" in img_path):
        # list_img_path = get_list_img_path(img_path)
        list_img_path = img_path
        print(time.time())
        df, face_cascade = load_database()
        print(time.time(), "load database")
        identity = predict_img_ha(list_img_path, df, face_cascade)
    # case 2: snapshot from service via apis
    else:  # have not done yet############
        df, face_cascade = load_database()

        identity = predict_img_local(img_path, df)

    duration = time.time() - pTime
    data = {"identity": identity, "duration": duration}
    data2json(data, result_path)
    return data


def predict_img_local(img_path, df):
    list_identity = []
    for img in img_path:
        list_candidate = []

        face_pixels = get_face_pixels(img)
        time.sleep(0.05)
        if face_pixels.shape[1:3] == input_shape:
            if df.shape[0] > 0:
                img1_representation = predict_tfmodel(face_pixels)[
                    0, :]

                def findDistance(row):
                    img2_representation = row['embedding']
                    distance = dst.findCosineDistance(
                        img1_representation, img2_representation)
                    return distance

                df['distance'] = df.apply(findDistance, axis=1)
                df = df.sort_values(by=["distance"])
                time.sleep(0.05)

                list_candidate = []
                for i in range(3):
                    candidate = df.iloc[i]
                    candidate_label = candidate['employee']
                    if (candidate_label in list_candidate):
                        break
                    else:
                        list_candidate.append(candidate_label)
                        candidate_label = 'unknown'
                time.sleep(0.05)
        list_identity.append(candidate_label)
    return max(list_identity, key=list_identity.count)


def predict_img_ha(list_img_path, df, face_cascade):
    candidate_label = 'unknown'
    if (str(type(list_img_path)) == "<class 'str'>"):
        img = cv2.imread(list_img_path)
        while(img.shape[0] > input_shape_size):
            img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))
        faces = face_cascade.detectMultiScale(img,  1.3, 5)
        for (x, y, w, h) in faces:
            if w > w_min:  # discard small detected faces
                # -------------------------------
                # apply deep learning for custom_face
                base_img = img.copy()
                img, region = functions.detect_face(
                    img=img, enforce_detection=False)
                # --------------------------

                if img.shape[0] > 0 and img.shape[1] > 0:
                    img = functions.align_face(img=img)
                else:
                    img = base_img.copy()
                # --------------------------
                # post-processing
                img = cv2.resize(img, input_shape)
                img_pixels = np.array(img, dtype=np.float32)
                face_pixels = np.expand_dims(img_pixels, axis=0)
                face_pixels /= 255  # normalize input in [0, 1]

                # print(time.time(), "face_pixels")

                # check preprocess_face function handled
                if face_pixels.shape[1:3] == input_shape:
                    if df.shape[0] > 0:
                        img1_representation = predict_tfmodel(face_pixels)[
                            0, :]

                        def findDistance(row):
                            img2_representation = row['embedding']
                            distance = dst.findCosineDistance(
                                img1_representation, img2_representation)
                            return distance
                        df['distance'] = df.apply(findDistance, axis=1)
                        # print(time.time(), "predict")
                        df = df.sort_values(by=["distance"])
                        # print(time.time(), "sort")
                        # print(df)
                        # print('--------------------')
                        list_candidates = df.iloc[0:3]['employee'].tolist()
                        if list_candidates.count(list_candidates[0]) >= 2:
                            candidate_label = list_candidates[0]
                        elif list_candidates.count(list_candidates[1]) >= 2:
                            candidate_label = list_candidates[1]
                        else:
                            candidate_label = 'unknown'
                        # print(time.time(), "get 1 in 3")
        result = candidate_label
    else:
        list_identity = []
        for img in list_img_path[:1]:
            print(time.time())
            img = cv2.imread(img)
            print(img.shape)
            while(img.shape[0] > input_shape_size):
                img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))
            faces = face_cascade.detectMultiScale(img,  1.3, 5)
            for (x, y, w, h) in faces:
                if w > w_min:  # discard small detected faces
                    # -------------------------------
                    # apply deep learning for custom_face
                    base_img = img.copy()
                    img, region = functions.detect_face(
                        img=img, enforce_detection=False)
                    # --------------------------

                    if img.shape[0] > 0 and img.shape[1] > 0:
                        img = functions.align_face(img=img)
                    else:
                        img = base_img.copy()
                    # --------------------------
                    # post-processing
                    img = cv2.resize(img, input_shape)
                    img_pixels = np.array(img, dtype=np.float32)
                    face_pixels = np.expand_dims(img_pixels, axis=0)
                    face_pixels /= 255  # normalize input in [0, 1]

                    print(time.time(), "face_pixels")

                    # check preprocess_face function handled
                    if face_pixels.shape[1:3] == input_shape:
                        if df.shape[0] > 0:
                            img1_representation = predict_tfmodel(face_pixels)[
                                0, :]

                            def findDistance(row):
                                img2_representation = row['embedding']
                                distance = dst.findCosineDistance(
                                    img1_representation, img2_representation)
                                return distance
                            df['distance'] = df.apply(findDistance, axis=1)
                            print(time.time(), "predict")
                            df = df.sort_values(by=["distance"])
                            print(time.time(), "sort")
                            print(df)
                            print('--------------------')
                            list_candidates = df.iloc[0:3]['employee'].tolist()
                            if list_candidates.count(list_candidates[0]) >= 2:
                                candidate_label = list_candidates[0]
                            elif list_candidates.count(list_candidates[1]) >= 2:
                                candidate_label = list_candidates[1]
                            else:
                                candidate_label = 'unknown'
                            print(time.time(), "get 1 in 3")
            list_identity.append(candidate_label)
        result = max(list_identity, key=list_identity.count)
    return result


def get_result():
    return json2data(result_path)


def load_database():
    # loading database
    embeddings = np.load(embedding_path, allow_pickle=True)
    df = pd.DataFrame(embeddings, columns=['employee', 'embedding'])
    df['distance_metric'] = distance_metric
    # -----------------------
    opencv_path = functions.get_opencv_path()
    face_detector_path = opencv_path+"haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(face_detector_path)
    return df, face_cascade
