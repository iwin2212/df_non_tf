from flask import Flask, render_template
from view import upload, take_shot, live_demo
from utils import destroy_camera, get_result, predict_snapshot

app = Flask(__name__)
app.config['SECRET_KEY'] = 'face_regconition'

app.register_blueprint(upload.mod)
app.register_blueprint(take_shot.mod)
app.register_blueprint(live_demo.mod)


@app.route('/')
def index():
    destroy_camera()
    return render_template("/index.html")


@app.route('/get_results', methods=['POST'])
def get_results():
    return get_result()

@app.route('/predict_snapshots', methods=['POST'])
def predict_snapshots():
    # predict_snapshot()
    # return get_result()
    return predict_snapshot()

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=2212, debug=True)
