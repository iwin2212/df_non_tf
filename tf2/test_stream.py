from numpy.lib.utils import source
from deepface import DeepFace
from deepface.commons import functions, distance as dst
from deepface.basemodels import VGGFace, OpenFace, Facenet, FbDeepFace, DeepID, DlibWrapper, ArcFace
from deepface import DeepFace
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2
import time
import re
import os
from mtcnn import MTCNN
from pathlib import Path
import gdown
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

database_path = "E:/tf2/database"


def save_images(db_path, model_name, distance_metric):
    employees = []
    # check passed db folder exists
    if os.path.isdir(db_path) == True:
        for r, d, f in os.walk(db_path):  # r=root, d=directories, f = files
            for file in f:
                if ('.jpg' in file):
                    #exact_path = os.path.join(r, file)
                    exact_path = r + "/" + file
                    # print(exact_path)
                    employees.append(exact_path)

    if len(employees) == 0:
        print("WARNING: There is no image in this path ( ", db_path,
              ") . Face recognition will not be performed.")

    # ------------------------

    if len(employees) > 0:

        model = DeepFace.build_model(model_name)
        print(model_name, " is built")

        # ------------------------

        input_shape = functions.find_input_shape(model)
        input_shape_x = input_shape[0]
        input_shape_y = input_shape[1]

    # ------------------------
    # facial attribute analysis models

    # ------------------------

    # find embeddings for employee list

    tic = time.time()

    pbar = tqdm(range(0, len(employees)), desc='Finding embeddings')

    embeddings = []
    # for employee in employees:
    for index in pbar:
        employee = employees[index]
        pbar.set_description("Finding embedding for %s" %
                             (employee.split("/")[-1]))
        embedding = []
        img = functions.preprocess_face(img=employee, target_size=(
            input_shape_y, input_shape_x), enforce_detection=False)
        img_representation = model.predict(img)[0, :]

        embedding.append(employee)
        embedding.append(img_representation)
        embeddings.append(embedding)
    np.save("embeddings.npy", np.array([embeddings]))
    toc = time.time()
    print("Embeddings found for given data set in ", toc-tic, " seconds")


def test_findThreshold(model_name, distance_metric):

    base_threshold = {'cosine': 0.40, 'euclidean': 0.55, 'euclidean_l2': 0.75}

    thresholds = {
        'VGG-Face': {'cosine': 0.40, 'euclidean': 0.55, 'euclidean_l2': 0.75},
        'OpenFace': {'cosine': 0.10, 'euclidean': 0.55, 'euclidean_l2': 0.55},
        'Facenet':  {'cosine': 0.40, 'euclidean': 10, 'euclidean_l2': 0.80},
        'DeepFace': {'cosine': 0.23, 'euclidean': 64, 'euclidean_l2': 0.64},
        'DeepID': 	{'cosine': 0.015, 'euclidean': 45, 'euclidean_l2': 0.17},
        'Dlib': 	{'cosine': 0.07, 'euclidean': 0.6, 'euclidean_l2': 0.6},
        'ArcFace':  {'cosine': 0.6871912959056619, 'euclidean': 4.1591468986978075, 'euclidean_l2': 1.1315718048269017}
    }

    threshold = thresholds.get(
        model_name, base_threshold).get(distance_metric, 0.4)

    return threshold


def test_build_model(model_name):
    """
    This function builds a deepface model
    Parameters:
            model_name (string): face recognition or facial attribute model
                    VGG-Face, Facenet, OpenFace, DeepFace, DeepID for face recognition
                    Age, Gender, Emotion, Race for facial attributes

    Returns:
            built deepface model
    """

    models = {
        'VGG-Face': VGGFace.loadModel,
        'OpenFace': OpenFace.loadModel,
        'Facenet': Facenet.loadModel,
        'DeepFace': FbDeepFace.loadModel,
        'DeepID': DeepID.loadModel,
        'Dlib': DlibWrapper.loadModel,
        'ArcFace': ArcFace.loadModel
    }

    model = models.get(model_name)

    if model:
        model = model()
        #print('Using {} model backend'.format(model_name))
        return model
    else:
        raise ValueError('Invalid model_name passed - {}'.format(model_name))


def test_find_input_shape(model):
    # face recognition models have different size of inputs
    # my environment returns (None, 224, 224, 3) but some people mentioned that they got [(None, 224, 224, 3)]. I think this is because of version issue.

    input_shape = model.layers[0].input_shape

    if type(input_shape) == list:
        input_shape = input_shape[0][1:3]
    else:
        input_shape = input_shape[1:3]

    if type(input_shape) == list:  # issue 197: some people got array here instead of tuple
        input_shape = tuple(input_shape)

    return input_shape


def get_opencv_path():
    opencv_home = cv2.__file__
    folders = opencv_home.split(os.path.sep)[0:-1]

    path = folders[0]
    for folder in folders[1:]:
        path = path + "/" + folder

    return path+"/data/"


def test_initialize_detector(detector_backend):

    global face_detector

    home = str(Path.home())

    # eye detector is common for opencv and ssd
    if detector_backend == 'opencv' or detector_backend == 'ssd':
        opencv_path = get_opencv_path()
        eye_detector_path = opencv_path+"haarcascade_eye.xml"

        if os.path.isfile(eye_detector_path) != True:
            raise ValueError(
                "Confirm that opencv is installed on your environment! Expected path ", eye_detector_path, " violated.")

        global eye_detector
        eye_detector = cv2.CascadeClassifier(eye_detector_path)

    # ------------------------------
    # face detectors
    if detector_backend == 'opencv':
        opencv_path = get_opencv_path()
        face_detector_path = opencv_path+"haarcascade_frontalface_default.xml"

        if os.path.isfile(face_detector_path) != True:
            raise ValueError(
                "Confirm that opencv is installed on your environment! Expected path ", face_detector_path, " violated.")

        face_detector = cv2.CascadeClassifier(face_detector_path)

    elif detector_backend == 'ssd':

        # check required ssd model exists in the home/.deepface/weights folder

        # model structure
        if os.path.isfile(home+'/.deepface/weights/deploy.prototxt') != True:

            print("deploy.prototxt will be downloaded...")

            url = "https://github.com/opencv/opencv/raw/3.4.0/samples/dnn/face_detector/deploy.prototxt"

            output = home+'/.deepface/weights/deploy.prototxt'

            gdown.download(url, output, quiet=False)

        # pre-trained weights
        if os.path.isfile(home+'/.deepface/weights/res10_300x300_ssd_iter_140000.caffemodel') != True:

            print("res10_300x300_ssd_iter_140000.caffemodel will be downloaded...")

            url = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

            output = home+'/.deepface/weights/res10_300x300_ssd_iter_140000.caffemodel'

            gdown.download(url, output, quiet=False)

        face_detector = cv2.dnn.readNetFromCaffe(
            home+"/.deepface/weights/deploy.prototxt",
            home+"/.deepface/weights/res10_300x300_ssd_iter_140000.caffemodel"
        )

    elif detector_backend == 'dlib':
        import dlib  # this is not a must library within deepface. that's why, I didn't put this import to a global level. version: 19.20.0

        global sp

        face_detector = dlib.get_frontal_face_detector()

        # check required file exists in the home/.deepface/weights folder
        if os.path.isfile(home+'/.deepface/weights/shape_predictor_5_face_landmarks.dat') != True:

            print("shape_predictor_5_face_landmarks.dat.bz2 is going to be downloaded")

            url = "http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2"
            output = home+'/.deepface/weights/'+url.split("/")[-1]

            gdown.download(url, output, quiet=False)

            zipfile = bz2.BZ2File(output)
            data = zipfile.read()
            newfilepath = output[:-4]  # discard .bz2 extension
            open(newfilepath, 'wb').write(data)

        sp = dlib.shape_predictor(
            home+"/.deepface/weights/shape_predictor_5_face_landmarks.dat")

    elif detector_backend == 'mtcnn':
        face_detector = MTCNN()


def test_analysis(db_path, model_name, distance_metric, enable_face_analysis=True, source=0, time_threshold=5, frame_threshold=5, no_label=15):
    text_color = (255, 255, 255)
    # ------------------------

    model = test_build_model(model_name)
    print(model_name, " is built")

    # ------------------------

    input_shape = test_find_input_shape(model)
    input_shape_x = input_shape[0]
    input_shape_y = input_shape[1]

    # tuned thresholds for model and metric pair
    threshold = test_findThreshold(model_name, distance_metric)

    # loading database
    try:
        embeddings = np.load("embeddings.npy", allow_pickle=True)[0]
    except:
        save_images(db_path=database_path,
                    model_name="ArcFace", distance_metric='cosine')
        embeddings = np.load("embeddings.npy", allow_pickle=True)[0]
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
    fps = cap.get(cv2.CAP_PROP_FPS)

    list_label = {}
    while(True):
        ret, img = cap.read()

        if img is None:
            break

        #cv2.namedWindow('img', cv2.WINDOW_FREERATIO)
        #cv2.setWindowProperty('img', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        raw_img = img.copy()
        resolution = img.shape

        resolution_x = img.shape[1]
        resolution_y = img.shape[0]

        if freeze == False:
            faces = face_cascade.detectMultiScale(img, 1.3, 5)
            if len(faces) == 0:
                face_included_frames = 0
        else:
            faces = []
        detected_faces = []
        face_index = 0
        for (x, y, w, h) in faces:
            if w > 130:  # discard small detected faces

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
            # print('--------------------> ', (toc - tic))
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
                        # facial attribute analysis

                        # -------------------------------
                        # face recognition

                        custom_face = functions.preprocess_face(img=custom_face, target_size=(
                            input_shape_y, input_shape_x), enforce_detection=False)

                        # check preprocess_face function handled
                        if custom_face.shape[1:3] == input_shape:
                            # if there are images to verify, apply face recognition
                            if df.shape[0] > 0:
                                img1_representation = model.predict(custom_face)[
                                    0, :]

                                #print(freezed_frame," - ",img1_representation[0:5])

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
                                df = df.sort_values(by=["distance"])

                                candidate = df.iloc[0]
                                employee_name = candidate['employee']
                                best_distance = candidate['distance']

                                #print(candidate[['employee', 'distance']].values)

                                # if True:
                                if best_distance <= threshold:
                                    # print(employee_name)
                                    display_img = cv2.imread(employee_name)

                                    display_img = cv2.resize(
                                        display_img, (pivot_img_size, pivot_img_size))

                                    label = employee_name.split(
                                        "/")[-1].replace(".jpg", "")
                                    label = re.sub('[0-9]', '', label)
                                    try:
                                        # top right
                                        if y - pivot_img_size > 0 and x + w + pivot_img_size < resolution_x:
                                            # freeze_img[y - pivot_img_size:y, x +
                                            #            w:x+w+pivot_img_size] = display_img

                                            overlay = freeze_img.copy()
                                            opacity = 0.4
                                            cv2.rectangle(
                                                freeze_img, (x+w, y), (x+w+pivot_img_size, y+20), (46, 200, 255), cv2.FILLED)
                                            cv2.addWeighted(
                                                overlay, opacity, freeze_img, 1 - opacity, 0, freeze_img)

                                            cv2.putText(
                                                freeze_img, label, (x+w, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

                                            # connect face and text
                                            # cv2.line(
                                            #     freeze_img, (x+int(w/2), y), (x+3*int(w/4), y-int(pivot_img_size/2)), (67, 67, 67), 1)
                                            # cv2.line(freeze_img, (x+3*int(w/4), y-int(pivot_img_size/2)),
                                            #          (x+w, y - int(pivot_img_size/2)), (67, 67, 67), 1)

                                        # bottom left
                                        elif y + h + pivot_img_size < resolution_y and x - pivot_img_size > 0:
                                            # freeze_img[y+h:y+h+pivot_img_size,
                                            #            x-pivot_img_size:x] = display_img

                                            overlay = freeze_img.copy()
                                            opacity = 0.4
                                            cv2.rectangle(
                                                freeze_img, (x-pivot_img_size, y+h-20), (x, y+h), (46, 200, 255), cv2.FILLED)
                                            cv2.addWeighted(
                                                overlay, opacity, freeze_img, 1 - opacity, 0, freeze_img)

                                            cv2.putText(
                                                freeze_img, label, (x - pivot_img_size, y+h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

                                            # connect face and text
                                            # cv2.line(freeze_img, (x+int(w/2), y+h), (x+int(w/2)-int(
                                            #     w/4), y+h+int(pivot_img_size/2)), (67, 67, 67), 1)
                                            # cv2.line(freeze_img, (x+int(w/2)-int(w/4), y+h+int(
                                            #     pivot_img_size/2)), (x, y+h+int(pivot_img_size/2)), (67, 67, 67), 1)

                                        # top left
                                        elif y - pivot_img_size > 0 and x - pivot_img_size > 0:
                                            # freeze_img[y-pivot_img_size:y,
                                            #            x-pivot_img_size:x] = display_img

                                            overlay = freeze_img.copy()
                                            opacity = 0.4
                                            cv2.rectangle(
                                                freeze_img, (x - pivot_img_size, y), (x, y+20), (46, 200, 255), cv2.FILLED)
                                            cv2.addWeighted(
                                                overlay, opacity, freeze_img, 1 - opacity, 0, freeze_img)

                                            cv2.putText(
                                                freeze_img, label, (x - pivot_img_size, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

                                            # connect face and text
                                            # cv2.line(freeze_img, (x+int(w/2), y), (x+int(w/2)-int(
                                            #     w/4), y-int(pivot_img_size/2)), (67, 67, 67), 1)
                                            # cv2.line(freeze_img, (x+int(w/2)-int(w/4), y-int(
                                            #     pivot_img_size/2)), (x, y - int(pivot_img_size/2)), (67, 67, 67), 1)

                                        # bottom righ
                                        elif x+w+pivot_img_size < resolution_x and y + h + pivot_img_size < resolution_y:
                                            # freeze_img[y+h:y+h+pivot_img_size,
                                            #            x+w:x+w+pivot_img_size] = display_img

                                            overlay = freeze_img.copy()
                                            opacity = 0.4
                                            cv2.rectangle(
                                                freeze_img, (x+w, y+h-20), (x+w+pivot_img_size, y+h), (46, 200, 255), cv2.FILLED)
                                            cv2.addWeighted(
                                                overlay, opacity, freeze_img, 1 - opacity, 0, freeze_img)

                                            cv2.putText(
                                                freeze_img, label, (x+w, y+h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

                                            # connect face and text
                                            # cv2.line(freeze_img, (x+int(w/2), y+h), (x+int(w/2)+int(
                                            #     w/4), y+h+int(pivot_img_size/2)), (67, 67, 67), 1)
                                            # cv2.line(freeze_img, (x+int(w/2)+int(w/4), y+h+int(
                                            #     pivot_img_size/2)), (x+w, y+h+int(pivot_img_size/2)), (67, 67, 67), 1)
                                    except Exception as err:
                                        print(str(err))

                                    # save label to list & return if number of a label is 15
                                    if (list_label.get(label) == None):
                                        list_label[label] = 1
                                    else:
                                        no = list_label[label] + 1
                                        if (no == no_label):
                                            break
                                        list_label[label] = no
                                    # print(list_label)
                        tic = time.time()  # in this way, freezed image can show 5 seconds

                        # -------------------------------

                # time_left = int(time_threshold - (toc - tic) + 1)

                # cv2.rectangle(freeze_img, (10, 10),
                #               (90, 50), (67, 67, 67), -10)
                # cv2.putText(freeze_img, str(time_left), (40, 40),
                #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

                cv2.putText(img, "fps: " + str(fps), (40, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow('img', freeze_img)

                freezed_frame = freezed_frame + 1
            else:
                face_detected = False
                face_included_frames = 0
                freeze = False
                freezed_frame = 0
                # cv2.putText(img, "fps: " + str(fps), (40, 40),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                # cv2.imshow('img', img)
        else:
            cv2.putText(img, "fps: " + str(fps), (40, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imshow('img', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
            break

    # kill open cv things
    cap.release()
    cv2.destroyAllWindows()


def test_stream(db_path='', model_name='Facenet', distance_metric='cosine', source=0, time_threshold=5, frame_threshold=5, no_label=15):
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

    test_initialize_detector(detector_backend='opencv')

    test_analysis(db_path, model_name, distance_metric, source=source,
                  time_threshold=time_threshold, frame_threshold=frame_threshold, no_label=no_label)


test_stream(db_path=database_path,
            model_name="Facenet", time_threshold=0.02, frame_threshold=1, source="rtsp://admin:ECSIAQ@192.168.1.48:554")

# save_images(db_path=database_path,
#             model_name="Facenet", distance_metric='cosine')
