import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, redirect 
from werkzeug.utils import secure_filename
import joblib
from flask import send_from_directory

app = Flask(__name__)


model = tf.keras.models.load_model('modelcn.h5')


additional_data = joblib.load('additional_data.pkl')
class_names = additional_data['class_names']

def preprocess_image(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def make_prediction(img_path):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    predicted_class_name = class_names[predicted_class[0]]
    return predicted_class_name, predictions

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        print("POST request received")  
        if 'file' not in request.files:
            print("No file part in request")  
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            print("No file selected")  
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join('uploads', filename)
            file.save(filepath)
            try:
                predicted_class_name, predictions = make_prediction(filepath)
                return render_template('index.html', prediction=predicted_class_name, filename=filename)
            except Exception as e:
                print(f"Error during prediction: {e}")
                return redirect(request.url)
    return render_template('index.html', prediction=None, filename=None)

if __name__ == "__main__":
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
