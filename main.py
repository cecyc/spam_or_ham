from flask import Flask, jsonify, request
from preprocess import preprocess

import joblib

app = Flask(__name__)

clf = joblib.load('classifier.pkl')
vectorizer = joblib.load('vectorizer.pkl')

prediction_cache = {}

@app.route('/classify', methods=['POST'])
def classify():
    sms = request.json['text']

    if sms in prediction_cache:
        prediction = prediction_cache[sms]
    else:
        preprocessed_sms = preprocess(sms)
        vectorized_text = vectorizer.transform([preprocessed_sms])
        prediction = clf.predict(vectorized_text)[0]
        prediction_cache[sms] = prediction

    return jsonify({'result': prediction})

if __name__ == '__main__':
    app.run()
