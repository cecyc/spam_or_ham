import joblib
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from preprocess import preprocess

CSV_URL = 'https://raw.githubusercontent.com/cecyc/spam_or_ham/main/data/sms_spam.csv'


def load_model():
    data = pd.read_csv(CSV_URL)

    # Split the data into test and train
    train_data = data.sample(frac=0.8, random_state=1)
    test_data = data.drop(train_data.index)

    # Vectorize the train and test data
    vectorizer = CountVectorizer(binary=True, preprocessor=preprocess)

    X_train = vectorizer.fit_transform(train_data['text'])
    y_train = train_data['type']

    X_test = vectorizer.transform(test_data['text'])
    y_test = test_data['type']

    # Train the model
    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    # Save the trained model to a file
    joblib.dump(clf, 'classifier.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
