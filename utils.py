from pathlib import Path
from const import snap_path
import os


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
