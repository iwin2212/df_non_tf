import numpy as np
from const import embedding_path, distance_metric, input_shape_x, input_shape_y, input_shape, model_name, text_color
from view.utils.lite_predict import predict_tfmodel
from custom_deepface.deepface.commons import functions, distance as dst
import cv2
import pandas as pd


def preprocess(img):
    # loading database
    embeddings = np.load(embedding_path, allow_pickle=True)
    df = pd.DataFrame(embeddings, columns=['employee', 'embedding'])
    df['distance_metric'] = distance_metric
    # -----------------------
    opencv_path = functions.get_opencv_path()
    face_detector_path = opencv_path+"haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(face_detector_path)
    faces = face_cascade.detectMultiScale(img,  1.3, 5)

    for (x, y, w, h) in faces:
        if w > 130:  # discard small detected faces
            # draw rectangle to main image
            cv2.rectangle(img, (x, y), (x+w, y+h), (40, 180, 240), 1)
            # -------------------------------------
            # -------------------------------
            # apply deep learning for custom_face
            cut_img, face_pixels, region = functions.preprocess_face(img=img[y:y+h, x:x+w], target_size=(
                input_shape_y, input_shape_x), enforce_detection=False, return_region=True)

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
                    # print(df)
                    list_candidate = []
                    for i in range(3):
                        candidate = df.iloc[i]
                        candidate_label = candidate['employee']
                        if (candidate_label in list_candidate):
                            break
                        else:
                            list_candidate.append(candidate_label)
                            candidate_label = 'unknown'
                    # print(
                    #     "\n-------------> {} - {}\n".format(candidate_label, threshold))
                    # show name
                    cv2.putText(
                        img, candidate_label, (x+int(w/2), y+h + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    return img


def draw_retangle(img):
    opencv_path = functions.get_opencv_path()
    face_detector_path = opencv_path+"haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(face_detector_path)
    faces = face_cascade.detectMultiScale(img,  1.3, 5)

    for (x, y, w, h) in faces:
        if w > 130:  # discard small detected faces
            # draw rectangle to main image
            cv2.rectangle(img, (x, y), (x+w, y+h), (40, 180, 240), 1)
    return img
