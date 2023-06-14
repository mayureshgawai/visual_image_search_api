from DeepImageSearch import Load_Data, Search_Setup
from flask import Flask, request, jsonify
from flask import render_template
from flask_cors import CORS, cross_origin
import matplotlib.pyplot as plt
import urllib.request as req
import numpy as np
from PIL import Image
import cv2
import uuid
import os

app = Flask(__name__)

image_list = Load_Data().from_folder(['image_data'])  # define any folder in image_list (it may don't even exist)
st = Search_Setup(image_list=image_list, model_name='vgg19', pretrained=True, image_count=162)
st.run_index()
metadata = st.get_image_metadata_file()


@app.route("/", methods=['GET', 'POST'])
@cross_origin()
def index():
    return "Hello world"

@app.route('/sendImage', methods=['GET', 'POST'])
@cross_origin()
def sendImage():
    try:
        # value of "image" will be an image read in PIL.Image()

        img_url = request.json["image_url"]
        response = req.urlopen(img_url)
        image_array = np.asarray(bytearray(response.read()), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        random_id = uuid.uuid4()
        name = "image_"+str(random_id)+".jpg"
        path = os.path.join("temp_data", name)
        cv2.imwrite(path, image)
        # plt.imsave(path, image_array)

        ids = []
        for key, val in st.get_similar_images(image_path=path, number_of_images=10).items():
            ids.append(val.split("\\")[-1].split(".")[0])

        return jsonify({"list": ids})

    except Exception as e:
        pass

@app.route("/addIndex", methods=['GET', 'POST'])
@cross_origin()
def addIndex():
    img_url = request.json["image_url"]
    response = req.urlopen(img_url)
    image_array = np.asarray(bytearray(response.read()), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    random_id = uuid.uuid4()
    name = "image_"+str(random_id)+".jpg"
    path = os.path.join("temp_data", name)
    cv2.imwrite(path, image)

    st.add_images_to_index([path])

    return "Done!"

if __name__ == '__main__':
    app.run()