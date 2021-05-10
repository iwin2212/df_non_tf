from flask import request, render_template, Blueprint, Response
from importlib import import_module
import os
from view.utils.lite_predict import predict_tfmodel
from flask import render_template, Response, jsonify
from const import input_shape, embedding_path, distance_metric, model_name
from utils import get_new_brand, check_file_exist, get_list_unknown_img, rename
from view.utils.data import detect_face
from ast import literal_eval
from view.utils.data import get_face_pixels
if os.environ.get('CAMERA'):
    Camera = import_module('camera_' + os.environ['CAMERA']).Camera
else:
    from view.camera_flask.camera_opencv import Camera
import cv2
import numpy as np
import time
from custom_deepface.deepface.commons import distance as dst
import pandas as pd
from view.utils.stream import draw_retangle
mod = Blueprint('take_shot', __name__)


@mod.route("/take_shots")
def take_shots():
    return render_template('./take_shot.html')


def generate(camera):
    """Video streaming generator function."""
    while True:
        frame = np.asarray(bytearray(camera.get_frame()), dtype="uint8")
        img = cv2.imdecode(frame, cv2.IMREAD_COLOR)

        try:
            image = draw_retangle(img=img)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', image)[1].tobytes() + b'\r\n')
        except Exception as error:
            print('Error in generate(camera): {}'.format(error))
            return ""


@mod.route('/video_feed')
def video_feed():
    return Response(generate(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@mod.route('/snap_shot',  methods=['POST'])
def snap_shot():
    img = Camera().get_frame()
    image = np.asarray(bytearray(img), dtype="uint8")
    image = detect_face(cv2.imdecode(image, cv2.IMREAD_COLOR))
    # print(image.shape)
    new_shot = get_new_brand()
    cv2.imwrite(new_shot, image)
    return jsonify(result=check_file_exist(new_shot))



@mod.route('/snapshot',  methods=['POST'])
def snapshot():
    link_list = []
    while(len(link_list)<=10):
        img = Camera().get_frame()
        image = np.asarray(bytearray(img), dtype="uint8")
        prev  = time.time()
        try:
            image = detect_face(cv2.imdecode(image, cv2.IMREAD_COLOR))
            new_shot = get_new_brand()
            # cv2.imshow(new_shot, image)
            cv2.imwrite(new_shot, image)
            link_list.append(new_shot)
            time.sleep(0.1)
        except Exception as error:
            print("Error: {}".format(error))
            now = time.time()
            
            if ((now-prev) > 10):
                print("Exceeded the time limit : {} > 10 (s)".format((now-prev)))
                return {"result" : "Connection denied" , "reason" : "Exceeded the time limit", "file_list": link_list}
            continue
    cv2.destroyAllWindows()
    return {"result" : "Success" , "file_list": link_list}



@mod.route('/predict_snapshot',  methods=['POST'])
def predict_snapshot():
    # loading database
    embeddings = np.load(embedding_path, allow_pickle=True)
    df = pd.DataFrame(embeddings, columns=['employee', 'embedding'])
    df['distance_metric'] = distance_metric
    threshold = dst.findThreshold(model_name, distance_metric)-0.1
    
    face_pixels = get_face_pixels()
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
            print(
                "\n-------------> {} - {}\n".format(candidate_label, threshold))
    return {}


@mod.route('/brandname')
def brandname():
    list_unknown_img = get_list_unknown_img()
    return render_template('./brandname.html', list_unknown_img=list_unknown_img)


@mod.route("/readdress",  methods=['POST'])
def readdress():
    rename_list = request.args.get("rename_list")
    data = literal_eval(rename_list)
    for key, value in data.items():
        rename(key, value)
    return jsonify()
