from flask import Flask, render_template
from view import upload
from const import *
app = Flask(__name__)
app.config['SECRET_KEY'] = 'face_regconition'

app.register_blueprint(upload.mod)


@app.route('/')
def index():
    return render_template("/index.html")


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=1234, debug=True)
