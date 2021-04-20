# from deepface import DeepFace


# result = DeepFace.verify("database/dung 86/1.jpg", "database/dung 86/2.jpg", distance_metric = "cosine",  model_name = 'Facenet')
# print(result)
# from deepface import DeepFace
# DeepFace.stream(enable_face_analysis=False,  db_path = "./", model_name = 'Facenet',  source="rtsp://admin:ECSIAQ@192.168.1.48:554", time_threshold=1, frame_threshold=1)

# from test_stream import test_build_model

# model = test_build_model('Facenet')
# model.summary()
import tensorflow as tf
from deepface.basemodels.Facenet import InceptionResNetV2

model = InceptionResNetV2()
model.load_weights("facenet_weights.h5")
model.summary()

# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = converter.convert()

# open("facenet.tflite", "wb").write(tflite_model)