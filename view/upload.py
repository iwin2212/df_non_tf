# -*- coding: utf-8 -*-
import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from ketqua import ketqua
UPLOAD_FOLDER = '/home/december/Desktop/web/test'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
	return render_template("index.html")

@app.route("/deeplearning")
def deep():
    return render_template("deep.html") 

@app.route("/uploader", methods = ['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
		file = request.files['file']
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		result = ketqua()
	return render_template("result.html", name = result)

if __name__ == "__main__":
	app.run(debug = True)