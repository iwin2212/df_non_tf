from pathlib import Path
import io
from const import snap_path, ROOT_DIR, video_source
import os
from view.utils.data import add_img2db
import yaml
import time
import cv2
from custom_deepface.deepface.commons import distance as dst
import pandas as pd
from view.utils.data import get_face_pixels
from view.utils.lite_predict import predict_tfmodel
from const import input_shape, embedding_path, distance_metric, model_name
import numpy as np

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
            # print("{} added to database".format(val))
            os.remove(key)
            # print("removed {}".format(key))
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


def get_token():
    secret_file = os.path.join(ROOT_DIR, 'secrets.yaml')
    data = yaml2dict(secret_file)
    authen_code = data['token']
    return authen_code


def destroy_camera():
    cv2.VideoCapture(video_source).release()


def predict_snapshot(list_img):
    pTime = time.time()
    list_candidate = []
    # loading database
    embeddings = np.load(embedding_path, allow_pickle=True)
    df = pd.DataFrame(embeddings, columns=['employee', 'embedding'])
    df['distance_metric'] = distance_metric
    threshold = dst.findThreshold(model_name, distance_metric)-0.1

    face_pixels = get_face_pixels()
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
            list_distance = []
            for i in range(3):
                candidate = df.iloc[i]
                candidate_label = candidate['employee']
                best_distance = candidate['distance']
                list_candidate.append(candidate_label)
                list_distance.append(best_distance)
            candidate_label = 'unknown'
            best_distance = 0
            for i in list_candidate:
                distance = list_distance[list_candidate.index(
                    i)]
                if (list_candidate.count(i) >= 2 and distance < threshold):
                    candidate_label = i
                    best_distance = distance
                    break
            list_candidate.append(candidate_label)
            time.sleep(0.05)

    identity = max(list_candidate,key=list_candidate.count)
    duration = time.time()
    return identity, duration
