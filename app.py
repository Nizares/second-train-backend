import os
from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from keras.preprocessing import image
import streamlit as st
from io import StringIO
import numpy as np
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__, template_folder='template')

lists = ["A", "B", "C", "D", "E", "F", "G"]

model = load_model('Model_Saved.h5')

# model.make_predict_function()

# def predict_label(img_path):
# 	i = image.load_img(img_path, target_size=(100,100))
# 	i = image.img_to_array(i)/255.0
# 	i = i.reshape(1, 100,100,3)
# 	p = model.predict_classes(i)
# 	return dic[p[0]]


# routes
@app.route("/")
def main():
	return render_template("index.html")

@app.route("/predict", methods = ['GET', 'POST'])
def predict():
    # Get uploaded file using local directory
    file = request.files['file']
    image_path = "./images/" + file.filename
    file.save(image_path)
    
	# load image from local
    image = keras.preprocessing.image.load_img(image_path,target_size=(150,150))
    
	# image to arrray
    x = keras.preprocessing.image.img_to_array(image)

    # Normalize pixel values to 0-1
    x = x / 255.0
    
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])

    pred = model.predict(images)
    pred = lists[np.argmax(pred)]
    
    classify = pred
    # # Make prediction using loaded model
    # y_pred = model.predict(x)[0]

    # # Return prediction result as JSON response
    # return jsonify({'result': int(y_pred)})
    # return jsonify({'result': (pred)}, prediction=classify)
    return render_template('index.html', prediction=classify)
if __name__ =='__main__':
    # app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
    app.run(debug = True)