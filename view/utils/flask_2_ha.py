import requests
import json

def turn_on_switch(entity):

	url = "http://192.168.0.110:8123/api/states/" + entity

	payload = json.dumps({
	"state": "on"
	})
	headers = {
	'Authorization': 'Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiI3MjMxMzczOGY5NWI0NTFjODRjNjU2M2ViMmVjZjczMyIsImlhdCI6MTYyMDU3MzAzNywiZXhwIjoxNjIwNTc0ODM3fQ.pJUIhhcLS3e1KwC-GrHk98E1Y1Uawc18aedwu-bCRgM',
	'Content-Type': 'application/json'
	}

	response = requests.request("POST", url, headers=headers, data=payload)
	return response.text