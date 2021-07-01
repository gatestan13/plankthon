import base64
import json
import os
import random

import requests

def get_body(count=128):
    path = '../test/images/'
    walk_gen = os.walk(path)
    walk = next(walk_gen)
    images = walk[2]
    random.shuffle(images)
    data = {}
    data['images'] = {}

    for image in images[:count]:
        with open(path + image, 'rb') as f:
            encoded_string = base64.b64encode(f.read())
            data['images'][image] = str(encoded_string)[2:-1]

    with open('body.json', 'w') as f:
        json.dump(data, f)

def get_proba(body='./body.json'):
    body_json = json.loads(open(body).read())
    resp = requests.request("POST", "http://127.0.0.1:8080", json=body_json)

    if resp.status_code == 200:
        resp_json = resp.json()

        with open('response.json', 'w') as f:
            json.dump(resp_json, f)

        return resp.json()
    else:
        raise