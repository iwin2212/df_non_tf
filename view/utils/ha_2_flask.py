from flask import request
from utils import check_file_exist
from const import ROOT_DIR
import os

print(request.headers['Host'])
command_file_path = os.path.join(ROOT_DIR, 'rest_command.yaml')
if not check_file_exist(command_file_path):
	print("File is not found. Create 'rest_command.yaml' in homeassistant.")
	