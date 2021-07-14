from flask import Flask, render_template, request
from view import upload, take_shot, live_demo
from utils import get_result, predict_snapshot, check_file_exist, check_running_condition, predict_service_snapshots
import os
from const import img_path
import logging
app = Flask(__name__)
app.config['SECRET_KEY'] = 'face_regconition'

app.register_blueprint(upload.mod)
app.register_blueprint(take_shot.mod)
app.register_blueprint(live_demo.mod)


@app.route('/')
def index():
    return render_template("/index.html")


@app.route('/get_results', methods=['GET', 'POST'])
def get_results():
    return get_result()


@app.route('/predict_snapshots', methods=['POST'])
def predict_snapshots():
    payload = request.get_json()
    if (payload['message'] == 'service_snapshot'):
        return predict_service_snapshots()
    else:
        img = os.path.join(img_path, payload['title'])
        if check_file_exist(img):
            return predict_snapshot(img)
        else:
            logging.warning("File is not found")
            return "File is not found"


if __name__ == "__main__":
    check_running_condition()
    app.run(host='0.0.0.0', port=2212, debug=False)
