import os

model_path = "model/facenet.tflite"
embedding_path = "database/embb.npy"

# video_source = "rtsp://admin:ECSIAQ@192.168.1.47:554"
video_source = 0

snap_path = "static/snap_shots"
UPLOAD_FOLDER = os.path.join(os.path.dirname(
    os.path.realpath(__file__)).split('/view')[0], "upload")
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
input_shape = input_shape_x, input_shape_y = 160, 160

text_color = (255, 255, 255)
distance_metric = 'cosine'
model_name = 'Facenet'