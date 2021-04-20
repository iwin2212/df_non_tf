import numpy as np 
from custom_deepface.deepface.commons import functions, distance as dst
from lite_predict import predict_tfmodel
import os

def represent(img_path, enforce_detection = True, detector_backend = 'opencv'):
	#decide input shape
	input_shape =  input_shape_x, input_shape_y= 160, 160

	#detect and align
	img = functions.preprocess_face(img = img_path
		, target_size=(input_shape_y, input_shape_x)
		, enforce_detection = enforce_detection
		, detector_backend = detector_backend)

	#represent
	embedding = predict_tfmodel(img)[0].tolist()
	return embedding

def add_img2db(img_path, label:str):
    if os.path.isfile("embedding.npy"):
        embeddings = np.load("embedding.npy", allow_pickle=True)
    else:
        embeddings = np.zeros(shape=(0, 2))
    embedding = np.array(represent(img_path))
    new_embeddings = np.concatenate([embeddings, np.array([label, embedding]).reshape(1, 2)], axis=0)
    np.save("embedding.npy", new_embeddings)

def add_embedding2db(embedding, label:str):
    if os.path.isfile("embedding.npy"):
        embeddings = np.load("embedding.npy", allow_pickle=True)
    else:
        embeddings = np.zeros(shape=(0, 2))
    embedding = np.array(embedding)
    new_embeddings = np.concatenate([embeddings, np.array([label, embedding]).reshape(1, 2)], axis=0)
    np.save("embedding.npy", new_embeddings)