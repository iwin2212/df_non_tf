from pathlib import Path

from numpy import delete
from const import snap_path, UPLOAD_FOLDER
import os
from view.data import add_img2db


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


def rename(key, val):
    if (val == 'delete'):
        os.remove(key)
    else:
        try:
            add_img2db(key, val)
            # print("{} added to database".format(val))
        except Exception as error:
            print("Error: {}".format(error))
        os.remove(key)
        # print("removed {}".format(key))
