from pathlib import Path
from const import snap_path, UPLOAD_FOLDER
import os
from view.data import add_img2db

import numpy as np
from const import embedding_path
import pandas as pd

def get_new_brand():
    name = 'snap'
    index = 0
    while(True):
        new_brand = os.path.join(snap_path, name+str(index) + '.jpg')
        if Path(new_brand).is_file():
            index += 1
        else:
            return new_brand


def check_file_exist(file_path):
    return Path(file_path).is_file()


def get_list_unknown_img():
    list_path = []
    for root, dirs, files in os.walk(snap_path, topdown=False):
        for name in files:
            path = "./" + os.path.join(root, name)
            list_path.append(path)
    return list_path


def get_current_folder():
    return os.path.dirname(os.path.realpath(__file__))


def rename(f):
    for key in f.keys():
        name = f[key]
        if (name != ''):
            try:
                add_img2db(key, name)
            except Exception as error:
                print("Error: {}".format(error))
                continue
            os.remove(key)

    embeddings = np.load(embedding_path, allow_pickle=True)
    df = pd.DataFrame(embeddings, columns=['employee', 'embedding'])
    print("-----------------------------\n{}\n---------------------------".format(df))
