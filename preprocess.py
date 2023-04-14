import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

def preprocess(sms):
    text = sms.lower()
    words = nltk.word_tokenize(text)

    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words]

    stemmer = PorterStemmer()
    words = [stemmer.stem(w) for w in words]

    return ' '.join(words)
