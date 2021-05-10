import requests
import json
from const import states_api
from utils import get_token


def post_entity_state(entity, state):
    headers = {
        "Authorization": "Bearer " + get_token(),
        "content-type": "application/json"
    }

    payload = json.dumps({
        "state": state
    })

    response = requests.request("POST", states_api.format(
        entity), headers=headers, data=payload)
    return response.text


def get_entity_state(entity):
    headers = {
        "Authorization": "Bearer " + get_token()
    }

    payload = ""

    response = requests.request("GET", states_api.format(
        entity), headers=headers, data=payload)
    return response.text
