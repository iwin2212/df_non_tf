# -*- coding: utf-8 -*-
import os
from flask import request, render_template, Blueprint
from view.data import add_img2db

from const import embedding_path
import numpy as np
import pandas as pd

UPLOAD_FOLDER = os.path.join(os.path.dirname(
    os.path.realpath(__file__)).split('/view')[0], "upload")
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

mod = Blueprint('upload', __name__)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@mod.route("/uploader", methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        uploaded_files = request.files.getlist("file[]")
        for index, file in enumerate(uploaded_files):
            if file and allowed_file(file.filename):
                name = request.form['name']
                filename = name + str(index) + '.jpg'
                path_img = os.path.join(UPLOAD_FOLDER, filename)
                file.save(path_img)
                add_img2db(path_img, name)

                # print embedding output here

                # embeddings = np.load(embedding_path, allow_pickle=True)
                # df = pd.DataFrame(embeddings, columns=['employee', 'embedding'])
                # print("-----------------------------\n{}\n---------------------------".format(df))

                # remove image in upload folder to optimize space
                os.remove(path_img)
        return render_template("./upload_2_db.html", action='uploaded', name=name)
    else:
        return render_template("./upload_2_db.html")


if __name__ == "__main__":
    mod.run(debug=True)
