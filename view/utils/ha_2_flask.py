from utils import check_file_exist, dict2yaml, get_token
from const import ROOT_DIR, snapshot_api, restart_api, config_api, automation_api
import os
import subprocess
import requests
import json


def restart_ha():
    try:
        subprocess.call(['sv', 'down', 'hass'])
        subprocess.call(['pkill', 'hass'])
        subprocess.call(['sv', 'up', 'hass'])

        result = "Đã restart server"
    except:
        headers = {
            "Authorization": "Bearer " + get_token(),
            "content-type": "application/json"
        }

        res = requests.post(restart_api, headers=headers)

        result = ''
        if res.status_code == 200:
            result = "Đang khởi động lại dịch vụ Javis HC. Xin vui lòng chờ trong giây lát."
    return result


def check_config():
    headers = {
        "Authorization": "Bearer " + get_token()
    }

    payload = ""

    response = requests.request(
        "POST", config_api, headers=headers, data=payload)
    return response.text


def create_rest_api(service_name, filename="rest_command.yaml"):
    try:
        command_file_path = os.path.join(ROOT_DIR, filename)

        if not check_file_exist(command_file_path):
            print("File is not found. Create {} in homeassistant.".format(filename))

            data = {
                service_name: {
                    "url": snapshot_api,
                    "method": "POST",
                    "payload": "{}",
                    "content_type": "application/json; charset=utf-8"
                }
            }
            dict2yaml(data, command_file_path)
            print(" *** You need to restart home assistant to take effect ***")
            try:
                restart_ha()
                print("Next step, you will persistently wait while restarting HA service.")
            except Exception as error:
                print("Error: {}".format(error))

        if check_file_exist(command_file_path):
            print("File {} is ready to use.".format(filename))
            return {"result": True, "route": command_file_path}
    except Exception as error:
        return {"result": False, "reason": error, "whereis": "create_rest_api"}


def create_automation(id_, entity_id):
    payload = json.dumps({
    "alias": "thangtest",
    "trigger": [
        {
        "platform": "state",
        "entity_id": entity_id,
        "to": "on"
        }
    ],
    "condition": [],
    "action": [
        {
        "service": "media_player.turn_on",
        "target": {
            "entity_id": "media_player.office"
        }
        }
    ]
    })

    headers = {
        "Authorization": "Bearer " + get_token(),
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", automation_api.format(id_), headers=headers, data=payload)

    return (response.text)

