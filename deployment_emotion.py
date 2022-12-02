

from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
from tensorflow.keras import Model
import numpy as np
import PIL
from PIL import Image
import efficientnet.keras as efn 
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import os

from functions.visuals import *
from functions.check import score


app = Flask(__name__)
class_names = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']


model = load_model('model.h5')

@app.route('/', methods=["GET"])
def hello():
    return render_template('index.html')


@app.route('/', methods=["POST"])
def predict():
    imagefile = request.files['imagefile']      
    image_path = "/Users/sarbjitmadra/Desktop/Concordia/emotion_detection/assets/" + imagefile.filename
    imagefile.save(image_path)
    
    img = tf.keras.utils.load_img(image_path, target_size=(48, 48))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) 
    
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    classification = class_names[np.argmax(score)]
    return f'The prediction shows {classification} face.'
    
    return render_template('index.html', prediction = classification)


if __name__ == '__main__':
    app.run(debug=True)
