import os
from flask import Flask, render_template, request, jsonify, url_for
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import tensorflow as tf
from tensorflow import keras
from google.cloud import storage
import tempfile

app = Flask(__name__, template_folder='template')
client = storage.Client.from_service_account_json('keys/serviceaccount.json')
bucket_name = 'model-loads'
bucket = client.bucket(bucket_name)
lists = ["A", "B", "C", "D", "E", "F", "G"]

model = load_model("Model_Saved.h5")

# routes
@app.route("/")
def main():
	return render_template("index.html")

@app.route("/predict", methods = ['GET', 'POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part in the request'

    file = request.files['file']

    if file.filename == '':
        return 'No selected file'

    # Save temp file to server
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    file.save(temp_file.name)
    
	# Save temp file to Google Cloud Storage
    blob = bucket.blob('images/' + file.filename)
    blob.upload_from_filename(temp_file.name)
    # get public url for shown image
    image_url = blob.public_url

    # Get uploaded file using local directory
    # file = request.files['file']
    # image_path = "./images/" + file.filename
    # file.save(image_path)
    
    
	# load image from local
    image = keras.preprocessing.image.load_img(temp_file.name,target_size=(150,150))
    
	# image to arrray
    x = keras.preprocessing.image.img_to_array(image)

    # Normalize pixel values to 0-1
    x = x / 255.0
    
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])

    pred = model.predict(images)
    pred = lists[np.argmax(pred)]
    
    classify = pred
    # return jsonify({'result': (pred)}, prediction=classify)
    temp_file.close()
    # return to fill in variable website
    return render_template('index.html', prediction=classify, image_url=image_url)
if __name__ =='__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
    # app.run(debug = True)