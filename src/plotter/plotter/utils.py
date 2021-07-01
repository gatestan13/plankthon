import base64
import json
import os
import random

import pandas as pd
import requests

def get_jsons(num, batch=10, size=128):
    if size < 1 or size > 128:
        raise ValueError("The size must be in [1, 128].")

    path = '../test/images/' # change path to image dir
    walk_gen = os.walk(path)
    walk = next(walk_gen)
    images = walk[2]
    random.shuffle(images)
    big_body = {}
    big_body['images'] = {}
    big_resp = {}
    big_resp['predictions'] = {}

    for i in range(batch):
        print(f'Processing batch #{i + 1}...')
        body = {}
        body['images'] = {}
        offset = i * size

        for image in images[offset:offset + size]:

            with open(path + image, 'rb') as f:
                encoded_string = base64.b64encode(f.read())
                big_body['images'][image] = str(encoded_string)[2:-1]
                body['images'][image] = str(encoded_string)[2:-1]

        r = requests.request("POST", "http://127.0.0.1:8080", json=body) # docker run --rm -p 8080:8080 bunyod16/plankathon:latest

        if r.status_code == 200:
            resp = r.json()
            big_resp['predictions'].update(resp['predictions'])
        else:
            raise    

    with open(f'json/body_{num}.json', 'w') as f:
        json.dump(big_body, f)

    with open(f'json/response_{num}.json', 'w') as f:
        json.dump(big_resp, f)

    return big_body, big_resp

def get_df(resp):
    df = pd.DataFrame.from_dict(resp['predictions'], orient='index')
    df = df.eq(df.where(df != 0).max(1), axis=0).astype(int)
    df2 = df.sum(axis=0).reset_index()
    df2.columns = ['label', 'count']
    df2 = df2[df2['count'] != 0]
    df2['type'] = df2['label'].str.contains(pat='^[A-Z]', regex=True).astype(int)
    df2 = df2.sort_values(['type', 'count', 'label'], ascending=[False, False, True]).reset_index(drop=True)

    for column in df.columns:
        df2.loc[df2['label'] == column, 'sample'] = str(df[column].idxmax())

    return df2

def get_image(body, title):
    encoded_string = body['images'][title]
    decoded_byte = base64.b64decode(encoded_string)
    return decoded_byte