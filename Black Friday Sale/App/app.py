from __future__ import division, print_function
import numpy as np
import joblib
from keras.preprocessing import image
from flask import Flask,request, render_template
app = Flask(__name__)

with open('model.h5', 'rb') as model_file:
    model = joblib.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        gender = int(request.form['gender'])
        age = int(request.form['age'])
        occupation = int(request.form['occupation'])
        stay_in_current_city_years = int(request.form['stay_in_current_city_years'])
        marital_status = int(request.form['marital_status'])
        product_category_1 = int(request.form['product_category_1'])
        product_category_2 = int(request.form['product_category_2'])
        product_category_3 = int(request.form['product_category_3'])
        b = int(request.form['b'])
        c = int(request.form['c'])

        input_features = np.array([[gender, age, occupation, stay_in_current_city_years, marital_status, 
                                    product_category_1, product_category_2, product_category_3, b, c]])
        
        prediction = round(model.predict(input_features)[0],2)

        return render_template('result.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
