import numpy as np 
from lite_predict import predict_tfmodel
from custom_deepface.deepface.commons import functions, distance as dst

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

def verify(img1_path, img2_path, distance_metric = 'cosine', detector_backend = 'mtcnn'):
    model_name = "facenet"
    img_list, bulkProcess = functions.initialize_input(img1_path, img2_path)
    functions.initialize_detector(detector_backend = detector_backend)
    resp_objects = []
    disable_option = False if len(img_list) > 1 else True
    img1_representation = represent(img1_path)
    img2_representation = represent(img2_path)
    if distance_metric == 'cosine':
        distance = dst.findCosineDistance(img1_representation, img2_representation)
    elif distance_metric == 'euclidean':
        distance = dst.findEuclideanDistance(img1_representation, img2_representation)
    elif distance_metric == 'euclidean_l2':
        distance = dst.findEuclideanDistance(dst.l2_normalize(img1_representation), dst.l2_normalize(img2_representation))
    else:
        raise ValueError("Invalid distance_metric passed - ", distance_metric)

    distance = np.float64(distance) #causes trobule for euclideans in api calls if this is not set (issue #175)
    #----------------------
    threshold = dst.findThreshold(model_name, distance_metric)
    if distance <= threshold:
        identified = True
    else:
        identified = False

    resp_obj = {
        "verified": identified
        , "distance": distance
        , "max_threshold_to_verify": threshold
        , "model": model_name
        , "similarity_metric": distance_metric
    }
    if bulkProcess == True:
        resp_objects.append(resp_obj)
    else:
        return resp_obj


print(verify("/home/iwin/Desktop/df_non_tf/database/dung 86/1.jpg", "/home/iwin/Desktop/df_non_tf/database/dung 86/2.jpg"))