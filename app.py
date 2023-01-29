from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename


app = Flask(__name__)

#import the model

model = load_model('model1.h5')


def model_pred(img_path, model):

    img = image.load_img(img_path, target_size=(224,224))

    img_arr = image.img_to_array(img)
    img_arr = img_arr/255
    img_arr = np.expand_dims(img_arr, axis = 0)
    pred = model.predict(img_arr)

    pred = np.argmax(pred, axis = 1)

    if pred == 0:
        out = "This car is Audi"
    
    elif pred == 1:
        out = "This car is Lambo"

    else:
        out = "This car is Mercedes"

    
    return out 

# defining app methods

@app.route('/', methods = ['GET'])
def index():
    #main webpage

    return render_template('index.html')



#predict method
@app.route('/predict', methods = ['GET', 'POST'])

def upload_picture():
    if request.method == 'POST':
        #USER POSTS IMAGE ON THE PAGE
        file = request.files['file']

        #saving file on uploads folder

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(file.filename))
        file.save(file_path)

        #making predictions

        preds = model_pred(file_path, model)
        result=preds
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)
