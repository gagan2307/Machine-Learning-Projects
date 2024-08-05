import re
import string
from flask import Flask, render_template, request
import joblib
from nltk.corpus import stopwords

# Define your text_process function
def text_process(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# Load the model with the custom function
pipeline_loaded = joblib.load("pipeline_model.pkl")

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        message = request.form['message']
        prediction_prob = pipeline_loaded.predict_proba([message])[0]
        spam_prob = prediction_prob[1]
        not_spam_prob = prediction_prob[0]
        print(f"Spam Probability: {spam_prob}, Not Spam Probability: {not_spam_prob}")  # Debug line
        prediction = "Spam" if spam_prob > 0.5 else "Not Spam"
        return render_template('index.html', prediction=prediction, prediction_prob=prediction_prob, message=message)
    return render_template('index.html', prediction=None)


if __name__ == '__main__':
    app.run(debug=True)
