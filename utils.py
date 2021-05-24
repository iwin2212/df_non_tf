from pathlib import Path
import io

from PIL.Image import new
from const import snap_path, restart_api, ROOT_DIR, video_source, img_path, result_path, w_min, threshold
import os
from view.utils.data import add_img2db, detect_face
import yaml
import time
import cv2
from custom_deepface.deepface.commons import distance as dst
import pandas as pd
from view.utils.data import get_face_pixels
from view.utils.lite_predict import predict_tfmodel
from const import input_shape, snapshot_api, embedding_path, distance_metric, input_shape_size, number_of_snapshots
import numpy as np
import subprocess
import json
from custom_deepface.deepface.commons import functions
import requests
from const import configration_path, ip_addr
from websocket import create_connection
from importlib import import_module
if os.environ.get('CAMERA'):
    Camera = import_module('camera_' + os.environ['CAMERA']).Camera
else:
    from view.camera_flask.camera_opencv import Camera
import logging


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
            logging.warning("Error in rename: {}".format(error))


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


def get_list_img_path(file_path):
    for root, dirs, files in os.walk(file_path, topdown=False):
        return [os.path.join(root, img)
                for img in files if (".jpg") in img]


def predict_snapshot(img_path=img_path):
    pTime = time.time()
    # case 1: snapshot from home assistant
    if ("/usr/share" in img_path):
        df, face_cascade = load_database()
        identity = predict_img_ha(img_path, df, face_cascade)

    # case 2: snapshot from service via api
    else:
        df, face_cascade = load_database()
        identity = predict_img_service(img_path, df, face_cascade)

    duration = time.time() - pTime
    data = {"identity": identity, "duration": duration}
    data2json(data, result_path)

    list_img_path = get_list_img_path('./static/snap_shots')
    for i in list_img_path:
        os.remove(i)
    return data, duration


def predict_img_service(list_img_path, df, face_cascade):
    list_identity = []
    candidate_label = 'unknown'
    for img in list_img_path:
        # print(time.time())
        img = cv2.imread(img)
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

                list_distance = df.iloc[0:3]['distance'].tolist()
                list_candidates = df.iloc[0:3]['employee'].tolist()
                if list_distance[0] > threshold:
                    candidate_label = 'unknown'
                else:
                    if (list_candidates.count(list_candidates[0]) >= 2):
                        candidate_label = list_candidates[0]
                    elif (list_distance[1] < threshold and list_candidates.count(list_candidates[1]) >= 2):
                        candidate_label = list_candidates[1]
                    else:
                        candidate_label = 'unknown'
                # print(time.time(), "get 1 in 3")
        list_identity.append(candidate_label)
    return max(list_identity, key=list_identity.count)


def predict_img_ha(list_img_path, df, face_cascade):
    candidate_label = 'unknown'
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
            # 3333333333333333333333333333333333
            cv2.imwrite(list_img_path, img)
            # --------------------------
            # post-processing
            img = cv2.resize(img, input_shape)
            img_pixels = np.array(img, dtype=np.float32)
            face_pixels = np.expand_dims(img_pixels, axis=0)
            face_pixels /= 255  # normalize input in [0, 1]

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
                    df = df.sort_values(by=["distance"])

                    list_distance = df.iloc[0:3]['distance'].tolist()
                    list_candidates = df.iloc[0:3]['employee'].tolist()
                    if list_distance[0] > threshold:
                        candidate_label = 'unknown'
                    else:
                        if (list_candidates.count(list_candidates[0]) >= 2):
                            candidate_label = list_candidates[0]
                        elif (list_distance[1] < threshold and list_candidates.count(list_candidates[1]) >= 2):
                            candidate_label = list_candidates[1]
                        else:
                            candidate_label = 'unknown'
    return candidate_label


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


def check_configration():
    data = open(configration_path, "r").read()
    if (data.find("/config/tmp/camera") != -1):
        logging.warning(
            "-> File {} is ready to use.".format(configration_path[configration_path.rfind('/')+1:]))
    else:
        logging.warning("Error: File {} is missing '/config/tmp/camera' in 'whitelist_external_dirs'".format(
            configration_path[configration_path.rfind('/')+1:]))


def check_running_condition():
    try:
        logging.warning("Checking condition:")

        logging.warning("- Task: Checking configration.yaml...")
        check_configration()

        logging.warning("- Task: Checking rest_command.yaml exist...")
        create_rest_api('predict_snapshot')

    except Exception as error:
        logging.warning("*** Error: {} ***".format(error))


def create_rest_api(service_name, filename="rest_command.yaml"):
    try:
        command_file_path = os.path.join(ROOT_DIR, filename)

        if not check_file_exist(command_file_path):
            logging.warning(
                "File is not found. Create {} in homeassistant.".format(filename))
            data = {
                service_name: {
                    "url": snapshot_api,
                    "method": "POST",
                    "payload": {"title": "{{ title }}", "message": "{{ message }}"},
                    "content_type": "application/json; charset=utf-8"
                }
            }
            dict2yaml(data, command_file_path)
            logging.warning(
                " *** We need to restart home assistant to take effect ***")
            try:
                restart_ha()
                logging.warning(
                    "Next step, you will persistently wait while restarting HA service.")
            except Exception as error:
                logging.warning("Error: {}".format(error))

        if check_file_exist(command_file_path):
            logging.warning("File {} existed.".format(filename))
            db = yaml2dict(command_file_path)
            new_db = {i: db[i] for i in db if (i != "predict_snapshot")}
            data_form = '{"title": "{{ title }}", "message": "{{ message }}"}'
            data = {
                service_name: {
                    "url": snapshot_api,
                    "method": "POST",
                    "payload": data_form,
                    "content_type": "application/json; charset=utf-8"
                }
            }
            new_db.update(data)
            dict2yaml(new_db, command_file_path)
            logging.warning("-> File {} is ready to use.".format(filename))
            return {"result": True, "route": command_file_path}
    except Exception as error:
        return {"result": False, "reason": error, "whereis": "create_rest_api"}


def restart_ha():
    try:
        subprocess.call(['sv', 'down', 'hass'])
        subprocess.call(['pkill', 'hass'])
        subprocess.call(['sv', 'up', 'hass'])

        result = "Đã restart server"
    except:
        headers = {
            "Authorization": "Bearer " + get_token(),
            "content-type": "application/json"
        }

        res = requests.post(restart_api, headers=headers)

        result = ''
        if res.status_code == 200:
            result = "Đang khởi động lại dịch vụ Javis HC. Xin vui lòng chờ trong giây lát."
    return result


def predict_service_snapshots():
    link_list = []
    while(len(link_list) < number_of_snapshots):
        img = Camera().get_frame()
        image = np.asarray(bytearray(img), dtype="uint8")
        try:
            image = detect_face(cv2.imdecode(image, cv2.IMREAD_COLOR))
            new_shot = get_new_brand()
            cv2.imwrite(new_shot, image)
            if (new_shot not in link_list):
                link_list.append(new_shot)
            else:
                continue
        except Exception as error:
            logging.warning("Error: {}".format(error))
            continue
    return predict_snapshot(link_list)


def read_result(entity_id, message = "{% if states('sensor.check_person') != 'unknown' %}Chào mừng {{ states('sensor.check_person') }} đã trở về nhà. Nhiệt độ trong phòng đang là 30 độ. Điều hoà đã được bật 26 độ mát. Xin hãy nghỉ ngơi và thư giãn.{% else %} Không nhận dạng được khuôn mặt. Xin hãy thử lại.{% endif %}"):
    headers = {
        "Authorization": "Bearer " + get_token(),
        'Content-Type': 'application/json'
    }
    url_flow = 'http://'+ip_addr+'/api/services/tts/google_translate_say'
    payload = {
        "entity_id": entity_id,
        "message": message,
        "language": "vi"
    }
    res = requests.post(url_flow, data=json.dumps(payload), headers=headers)
    if res.status_code == 200:
        return {"result": "done"}
    else:
        return {"result": res.status_code}


def get_tts_devices():
    headers = {
        "Authorization": "Bearer " + get_token(),
        'Content-Type': 'application/json'
    }
    payload = ""
    url_flow = 'http://'+ip_addr+'/api/states'
    res = requests.request("GET", url_flow, headers=headers, data=payload)
    data = res.json()
    tts_devices = [dev['entity_id'] for dev in data if ("media_player" in dev['entity_id'])]

    if (len(tts_devices) == 1):
        read_result(tts_devices[0])
    logging.warning(tts_devices)

