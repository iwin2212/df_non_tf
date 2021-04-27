# -*- coding: utf-8 -*-
import os
from flask import Flask, request, render_template, Blueprint
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)).split('/view')[0], "upload")
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

mod = Blueprint('upload', __name__)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@mod.route("/uploader", methods = ['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
		uploaded_files = request.files.getlist("file[]")
		for index, file in enumerate(uploaded_files):
			if file and allowed_file(file.filename):
				filename = request.form['name'] + str(index) + '.jpg'
				file.save(os.path.join(UPLOAD_FOLDER, filename))
		return render_template("./upload_2_db.html", action='uploaded') 
	else:
		return render_template("./upload_2_db.html")

if __name__ == "__main__":
	mod.run(debug = True)