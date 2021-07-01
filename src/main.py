##############################################################################
# File: main.py
# Project: hackathon-plankton
# Created Date: Friday, June 18th 2021, 10:33:29 pm
# Author: ZHANG Xiyu
# -----
# Last Modified: Friday, 25th June 2021 10:03:48 am
# Modified By: ZHANG Xiyu
# -----
# Copyright (c) 2021 Veolia S.A.
# Please add license text here.
# Nope...we're doomed!
# -----
# HISTORY:
# Date      	By	Comments
# ----------	---	----------------------------------------------------------
##############################################################################
"""# Template of Hackathon Plankton Model Serving

To launch the server locally:

```console
    $ python3 -m venv env
    $ source env/bin/activate
    $ pip3 install -r requirements.txt
    $ uvicorn main:app --reload --host=127.0.0.1 --port=8080
```

To do an unit test:

```console
    $ pytest test_local_server.py
```

To launch the server in a container:

```console
    $ chmod +x ./build.sh ./run.sh
    $ ./build.sh
    $ ./run.sh
```

To read docs or try this api in your browser, please navigate to:

- openapi : <http://127.0.0.1:8080/docs#/>
- redoc : <http://127.0.0.1:8080/redoc/>
- openapi as json : <http://127.0.0.1:8080/hackathon-plankton-openapi.json>

To push your image to Docker Hub:

```console

    $ chmod +x ./push.sh
    $ ./push.sh
```

Notes:

- You should replace some codes in `main.py`, `requirements.txt`, `build.sh`, `run.sh`, `push.sh` as asked like this:

```python
### START CODE HERE (replace codes below with your codes) ###
...
...
...
### END CODE HERE ###
```

- If you use the provided baseline(model-3825702330334969856) as yours, you will be removed from the list of candidates. This baseline model just help you to understand how to serve a Tensorflow savedModel within the uvicorn server. That's all.
- Please don't be shy to ask questions to us during this competition.
"""
import base64
import io
import json
import logging
from typing import Dict

from fastapi import FastAPI, Header
from fastapi.encoders import jsonable_encoder
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

LOGGER = logging.getLogger("server_template." + __name__)
LOGGER.setLevel(logging.INFO)

# Here the app variable will be an "instance" of the class FastAPI.
# This will be the main point of interaction to create all your API.
app = FastAPI(openapi_url="/hackathon-plankton-openapi.json")
# load open api doc examples.
DOC_BODY = json.loads(open("./request/body.json").read())
DOC_RESPONSE = json.loads(open("./request/response.json").read())


### START CODE HERE (replace codes below with your codes) ###
# * load tf-saved-model from model's directory.
from io import BytesIO
from PIL import Image
import timm
import torch
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

CHECKPOINT_PATH = "C:\\Users\\gates\\Desktop\\project\\checkpoint-0.pth.tar"
MODEL = timm.create_model('vit_base_patch16_224_miil_in21k', num_classes=84)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device('cpu'))
MODEL.load_state_dict(checkpoint['state_dict'])
MODEL.eval()
config = resolve_data_config({}, model=MODEL)
transform = create_transform(**config)

def get_image_proba(imageinbytes):
	img = Image.open(io.BytesIO(imageinbytes))
	tensor = transform(img).unsqueeze(0)
	with torch.no_grad():
		out = MODEL(tensor)
	probabilities = torch.nn.functional.softmax(out[0], dim=0)
	return (probabilities)

### END CODE HERE ###


def custom_openapi():
    """Utility function to generate the OpenAPI schema.

    Returns:
        `Dict`: A dictionary of Openapi Schema.
    """
    if app.openapi_schema:
        # use the property .openapi_schema as a "cache".
        return app.openapi_schema
    tags_desc_list = [
        {
            "name": "root",
            "description": "Post endpoint predicts the probabilities of the 84 classes for each image of plankton.",
        }
    ]
    openapi_schema = get_openapi(
        title="Hackathon Plankton API",
        version="1.0.0",
        description="This API will give the probabilities of the 84 classes for each image of plankton.",
        routes=app.routes,
        tags=tags_desc_list,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


class ImageBatch(BaseModel):
    """Customized Input Data Model."""

    images: Dict[str, str] = Field(
        ...,
        title="A batch of images",
        description="""A `Dict(key, value)`, \
where the key is the name(`str`) of one jpeg image and \
the value is the encoded string of this image via `base64` algorithm. \
For example, A key is a string `"86177.jpg"`. \
A value is an encoded string `"/9j/4AAQSkZJRgABAQAAAQABAAD...UUUUCP/2Q=="` \
via `base64.b64encode(file_object.read()).decode()`.""",
        example=DOC_BODY["images"],
    )

    @validator("images", pre=True)
    @classmethod
    def check_nums(cls, images):
        """In each request, whether there is at least one image and no more than 128."""
        num_images = len(images)
        if num_images < 1 or num_images > 128:
            raise ValueError("The number of images must be in [1, 128].")
        return images


class ResponseBatch(BaseModel):
    """Customized Output Data Model."""

    predictions: Dict[str, Dict[str, float]] = Field(
        ...,
        title="A batch of predictions",
        description="""A `Dict(key, value)`, \
where the key is the name(`str`) of one jpeg image and \
the value is a class-to-probability dictionary.""",
        example=DOC_RESPONSE["predictions"],
    )


def preprocess_image(image_as_string, max_width=640, max_height=480):
    """Preprocesses input images in string.

    Args:
        image_as_string: a image string from base64.b64decode.
        max_width: The max width for preprocessed images. The max width is 640
            (1024) for Image Classfication (Object Detection) models.
        max_height: The max width for preprocessed images. The max height is
            480 (1024) for Image Classfication (Object Detetion) models.
    Returns:
        The preprocessed image in string.
    """
    im = Image.open(io.BytesIO(image_as_string), mode="r", formats=None)
    width, height = im.size
    processed_byte_io = BytesIO()
    if height > max_height or width > max_width:
        ratio = max(height / float(max_width), width / float(max_height))
        new_height = int(height / ratio + 0.5)
        new_width = int(width / ratio + 0.5)
        im = im.resize(size=(new_width, new_height), resample=Image.NEAREST)
        im.save(processed_byte_io, "JPEG")
    else:
        im.save(processed_byte_io, "JPEG")
    processed_image_as_string = processed_byte_io.getvalue()
    return processed_image_as_string


def get_predictions(image_batch: ImageBatch) -> ResponseBatch:
    """Get predictions for incoming images.
    Args:
        image_batch (ImageBatch): Customized Input Data Model.
    Returns:
        ResponseBatch: Customized Output Data Model.
    """
    ### START CODE HERE (replace codes below with your codes) ###
    input_images = []
    input_keys = []
    for k, v in image_batch.images.items():
        input_keys.append(k)
        img_str = preprocess_image(base64.b64decode(v))
        input_images.append(img_str)

    with open('dictionary','r') as f:
        names = json.load(f)
    
    big_predictions = {}
    i = 0
    for image_name in input_keys:
        predictions = {}
        tensor = get_image_proba(input_images[i])
        for _class, proba in enumerate(tensor):
            predictions[names[str(_class)]] = proba
        big_predictions[image_name] = predictions
        i += 1
    
    # for i, k in enumerate(output_key):
    #     k = k.decode("utf-8")
    #     predictions[k] = {}
    #     for j, l in enumerate(output_labels[i]):
    #         l = l.decode("utf-8")
    #         predictions[k][l] = round(output_scores[i][j], 4)
    ### END CODE HERE ###
    response_batch = ResponseBatch(predictions=predictions)
    return response_batch


@app.post("/", response_model=ResponseBatch, tags=["root"])
def root(image_batch: ImageBatch):
    """This function will call `get_predictions` function to classify images.
    Then, it will generate the final response for a request.
    Args:
        image_batch (ImageBatch): Customized Input Data Model.
    Returns:
        ResponseBatch: Customized Output Data Model.
    """
    response_batch = get_predictions(image_batch)
    response_batch = jsonable_encoder(response_batch)
    response_batch = JSONResponse(response_batch)
    return response_batch


@app.get("/healthz", response_model=str, tags=["healthz"], include_in_schema=False)
def read_root():
    """This router is used to check the health of the running server.

    Returns:
        `str`: yes.
    """
    return "yes"
