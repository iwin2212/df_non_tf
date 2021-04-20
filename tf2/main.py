from models import get_model
import tensorflow as tf
import keras
print(tf.__version__)
print(keras.__version__)
args_model = "biLSTM"
input_dim = 940
no_activities = 7
model = get_model(args_model, input_dim, no_activities)
model.load_weights("biLSTM-cairo-20210409-073659.h5")
model.summary()

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

open("model.tflite", "wb").write(tflite_model)
