from flask import Flask, jsonify, request
import nltk

nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

app = Flask(__name__)

@app.route('/classify', methods=['POST'])
def classify():
    sms = request.json['text']
    preprocessed_sms = __preprocess(sms)

    # just return preprocessed text for now
    return jsonify({'word': preprocessed_sms})

def __preprocess(sms):
    text = sms.lower()
    words = nltk.word_tokenize(text)

    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words]

    stemmer = PorterStemmer()
    words = [stemmer.stem(w) for w in words]

    return ' '.join(words)


if __name__ == '__main__':
    app.run()
