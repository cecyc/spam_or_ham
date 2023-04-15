from flask import Flask, request, render_template
from preprocess import preprocess

import joblib
import model

app = Flask(__name__)

model.load_model()
clf = joblib.load('classifier.pkl')
vectorizer = joblib.load('vectorizer.pkl')

prediction_cache = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    sms = request.form['text']

    if sms in prediction_cache:
        prediction = prediction_cache[sms]
    else:
        preprocessed_sms = preprocess(sms)
        vectorized_text = vectorizer.transform([preprocessed_sms])
        prediction = clf.predict(vectorized_text)[0]
        prediction_cache[sms] = prediction

    return render_template('index.html', result=prediction, sms=sms)

if __name__ == '__main__':
    app.run()
