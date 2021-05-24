import os
from custom_deepface.deepface.commons import distance as dst

model_path = "model/facenet.tflite"
# model_path = "model/arcface.tflite"
embedding_path = "database/embeddings.npy"
# embedding_path = "database/test.npy"
# embedding_path = "database/gray_scale.npy"


# video_source = "rtsp://admin:javis2021@192.168.1.109:554"
# video_source = "rtsp://admin:ECSIAQ@192.168.1.47:554"
video_source = 0

snap_path = "static/snap_shots"
UPLOAD_FOLDER = os.path.join(os.path.dirname(
    os.path.realpath(__file__)).split('/view')[0], "upload")
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
# input_shape = input_shape_x, input_shape_y = 224, 224
input_shape = input_shape_x, input_shape_y = 160, 160
# input_shape = input_shape_x, input_shape_y = 112, 112
w_min = 100

text_color = (255, 255, 255)
distance_metric = 'cosine'
model_name = 'Facenet'
# model_name = 'ArcFace'

token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiIxZGQwYjNlNWE2ZTc0ZTY5YTM5NzdlZDAxMWE2Mjk5OCIsImlhdCI6MTU5NTE4MDYyOCwiZXhwIjoxOTEwNTQwNjI4fQ.9IytiXHV98pS4x5nxhH7z1QAq91ZXzBQaeJsZ8U2ZAQ"

ROOT_DIR = "/usr/share/hassio/homeassistant/"
configration_path = "/usr/share/hassio/homeassistant/configuration.yaml"

ip_addr  = "127.0.0.1:8123"
# ip_addr  = "192.168.0.110:8123"

states_api = "http://"+ ip_addr +"/api/states/{}"
config_api = "http://"+ ip_addr +"/api/config/core/check_config"
restart_api = "http://"+ ip_addr +"/api/services/homeassistant/restart"
snapshot_api = "http://"+ ip_addr +"/predict_snapshots"
automation_api = "http://"+ ip_addr +"/api/config/automation/config/{}"

result_path = "./static/result/result.json"
img_path = "/usr/share/hassio/homeassistant/tmp/camera"

input_shape_size = 500

number_of_snapshots = 3

# tuned thresholds for model and metric pair
threshold = dst.findThreshold(model_name, distance_metric) - 0.1
