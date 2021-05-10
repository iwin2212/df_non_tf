from pathlib import Path
import io
from const import snap_path, ROOT_DIR
import os
from view.utils.data import add_img2db
import yaml


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


def yaml2dict(filename):
    f = open(filename, 'r', encoding='utf8')
    res = yaml.load(f, Loader=yaml.FullLoader)
    f.close()
    return res


def dict2yaml(dict_, filename):
    with io.open(filename, 'w', encoding='utf-8') as outfile:
        yaml.dump(dict_, outfile, default_flow_style=False, allow_unicode=True)


def get_token():
    secret_file = os.path.join(ROOT_DIR, 'secrets.yaml')
    data = yaml2dict(secret_file)
    authen_code = data['token']
    return authen_code
