from custom_deepface.deepface.commons import functions, distance as dst
from lite_predict import predict_tfmodel
import os
import numpy as np
import pandas as pd
import cv2
import time
import os
from const import embedding_path, w_min
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def test_analysis(distance_metric, source=0, time_threshold=5, frame_threshold=5):
    text_color = (255, 255, 255)
    # ------------------------
    input_shape = input_shape_x, input_shape_y = 160, 160
    model_name='Facenet'
    # threshold = dst.findThreshold(model_name, distance_metric)
    threshold = dst.findThreshold(model_name, distance_metric) - 3

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

    freeze = False
    face_detected = False
    face_included_frames = 0  # freeze screen if face detected sequantially 5 frames
    freezed_frame = 0
    tic = time.time()
    cap = cv2.VideoCapture(source)  # webcam
    list_label = {}
    while(True):
        try:
            t0 = time.time()
            ret, img = cap.read()
            while(img.shape[0]>300):
                img = cv2.resize(img, (int(img.shape[1]/2),int(img.shape[0]/2)))
            fps = (1/(time.time() - t0))
            fpstext = "FPS: " + str(fps)[:2]
            if img is None:
                break
            raw_img = img.copy()
            if freeze == False:
                faces = face_cascade.detectMultiScale(img,  1.3, 5)
                if len(faces) == 0:
                    face_included_frames = 0
            else:
                faces = []
            detected_faces = []
            face_index = 0

            for (x, y, w, h) in faces:
                if w > w_min:  # discard small detected faces
                    face_detected = True
                    if face_index == 0:
                        face_included_frames = face_included_frames + \
                            1  # increase frame for a single face

                    # draw rectangle to main image
                    cv2.rectangle(img, (x, y), (x+w, y+h), (67, 67, 67), 1)

                    cv2.putText(img, str(frame_threshold - face_included_frames), (int(x+w/4),
                                int(y+h/1.5)), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 2)
                    # crop detected face
                    detected_face = img[int(y):int(y+h), int(x):int(x+w)]

                    # -------------------------------------
                    detected_faces.append((x, y, w, h))
                    face_index = face_index + 1

                    # -------------------------------------
            if face_detected == True and face_included_frames == frame_threshold and freeze == False:
                freeze = True
                #base_img = img.copy()
                base_img = raw_img.copy()
                detected_faces_final = detected_faces.copy()
                tic = time.time()

            if freeze == True:
                toc = time.time()
                
                if (toc - tic) < time_threshold:
                    if freezed_frame == 0:
                        freeze_img = base_img.copy()
                        # freeze_img = np.zeros(resolution, np.uint8) #here, np.uint8 handles showing white area issue
                        for detected_face in detected_faces_final:
                            x = detected_face[0]
                            y = detected_face[1]
                            w = detected_face[2]
                            h = detected_face[3]
                            # draw rectangle to main image
                            cv2.rectangle(freeze_img, (x, y),
                                          (x+w, y+h), (67, 67, 67), 1)
                            # -------------------------------
                            # apply deep learning for custom_face
                            custom_face = base_img[y:y+h, x:x+w]
                            # -------------------------------
                            # face recognition
                            custom_face = functions.preprocess_face(img=custom_face, target_size=(
                                input_shape_y, input_shape_x), enforce_detection=False)

                            # check preprocess_face function handled
                            if custom_face.shape[1:3] == input_shape:
                                # if there are images to verify, apply face recognition
                                if df.shape[0] > 0:
                                    img1_representation = predict_tfmodel(custom_face)[
                                        0, :]

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

                                    df['distance'] = df.apply(
                                        findDistance, axis=1)
                                    df = df.sort_values(by=["distance"])
                                    print(df)
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
                                        distance = list_distance[list_candidate.index(i)]
                                        if (list_candidate.count(i) >= 2 and distance <threshold):
                                            candidate_label = i
                                            best_distance = distance
                                            break
                                    print("\n-------------> {} - {}\n".format(candidate_label, threshold))

                                    # if True:
                                    if best_distance <= threshold:
                                        label = candidate_label
                                        try:
                                            # top right
                                            if y - pivot_img_size > 0 and x + w + pivot_img_size < resolution_x:

                                                overlay = freeze_img.copy()
                                                opacity = 0.4
                                                cv2.rectangle(
                                                    freeze_img, (x+w, y), (x+w+pivot_img_size, y+20), (46, 200, 255), cv2.FILLED)
                                                cv2.addWeighted(
                                                    overlay, opacity, freeze_img, 1 - opacity, 0, freeze_img)

                                                cv2.putText(
                                                    freeze_img, label, (x+w, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

                                            # bottom left
                                            elif y + h + pivot_img_size < resolution_y and x - pivot_img_size > 0:

                                                overlay = freeze_img.copy()
                                                opacity = 0.4
                                                cv2.rectangle(
                                                    freeze_img, (x-pivot_img_size, y+h-20), (x, y+h), (46, 200, 255), cv2.FILLED)
                                                cv2.addWeighted(
                                                    overlay, opacity, freeze_img, 1 - opacity, 0, freeze_img)

                                                cv2.putText(
                                                    freeze_img, label, (x - pivot_img_size, y+h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

                                            # top left
                                            elif y - pivot_img_size > 0 and x - pivot_img_size > 0:

                                                overlay = freeze_img.copy()
                                                opacity = 0.4
                                                cv2.rectangle(
                                                    freeze_img, (x - pivot_img_size, y), (x, y+20), (46, 200, 255), cv2.FILLED)
                                                cv2.addWeighted(
                                                    overlay, opacity, freeze_img, 1 - opacity, 0, freeze_img)

                                                cv2.putText(
                                                    freeze_img, label, (x - pivot_img_size, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

                                            # bottom righ
                                            elif x+w+pivot_img_size < resolution_x and y + h + pivot_img_size < resolution_y:

                                                overlay = freeze_img.copy()
                                                opacity = 0.4
                                                cv2.rectangle(
                                                    freeze_img, (x+w, y+h-20), (x+w+pivot_img_size, y+h), (46, 200, 255), cv2.FILLED)
                                                cv2.addWeighted(
                                                    overlay, opacity, freeze_img, 1 - opacity, 0, freeze_img)

                                                cv2.putText(
                                                    freeze_img, label, (x+w, y+h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

                                        except Exception as err:
                                            print(str(err))

                                        # save label to list & return if number of a label is 15
                                        if (list_label.get(label) == None):
                                            list_label[label] = 1
                                        else:
                                            num = list_label[label] + 1
                                            if (num == 100):
                                                cv2.imwrite("result.png", freeze_img)
                                                return list_label
                                            list_label[label] = num
                                        print(list_label)
                            tic = time.time()  # in this way, freezed image can show 5 seconds

                            # -------------------------------
                    cv2.putText(freeze_img, fpstext, (40, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    cv2.imshow('img', freeze_img)
                    freezed_frame = freezed_frame + int(fps)
                else:
                    face_detected = False
                    face_included_frames = 0
                    freeze = False
                    freezed_frame = 0
            else:
                cv2.putText(img, fpstext, (40, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.imshow('img', img)

            if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
                break
        except Exception as error:
            print("An error has just occured: {}.".format(error))
            continue
    # kill open cv things
    cap.release()
    cv2.destroyAllWindows()


def test_stream(distance_metric='cosine', source=0, time_threshold=5, frame_threshold=5):
    """
    This function applies real time face recognition and facial attribute analysis

    Parameters:
            db_path (string): facial database path. You should store some .jpg files in this folder.

            model_name (string): VGG-Face, Facenet, OpenFace, DeepFace, DeepID, Dlib or Ensemble

            distance_metric (string): cosine, euclidean, euclidean_l2

            enable_facial_analysis (boolean): Set this to False to just run face recognition

            source: Set this to 0 for access web cam. Otherwise, pass exact video path.

            time_threshold (int): how many second analyzed image will be displayed

            frame_threshold (int): how many frames required to focus on face

            no_label (int): threshold the number of label appear

    """

    # if time_threshold < 1:
    #     raise ValueError(
    #         "time_threshold must be greater than the value 1 but you passed "+str(time_threshold))

    if frame_threshold < 1:
        raise ValueError(
            "frame_threshold must be greater than the value 1 but you passed "+str(frame_threshold))

    functions.initialize_detector(detector_backend='opencv')

    test_analysis(distance_metric, source=source, time_threshold=time_threshold, frame_threshold=frame_threshold)


test_stream(distance_metric='euclidean', time_threshold=0.02, frame_threshold=1,
            source="rtsp://admin:ECSIAQ@192.168.1.47:554")
# test_stream(distance_metric='euclidean', time_threshold=0.02, frame_threshold=1, source=0)

# save_images(db_path=database_path, model_name="ArcFace", distance_metric='cosine')
