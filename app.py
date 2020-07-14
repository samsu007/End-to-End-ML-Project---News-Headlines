from flask import Flask, request, jsonify, render_template
import numpy as np
from sklearn.exceptions import NotFittedError
import pickle
from sklearn.feature_extraction.text import CountVectorizer
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
loaded_vectorizer = pickle.load(open('vectorizer.pickle', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    output = list(request.form.values())
    test_headlines = []
    test_headlines.append("this is a heading")
    print(test_headlines)
    try:
        countvector = CountVectorizer(ngram_range=(2, 2))
        testdataset = loaded_vectorizer.transform(test_headlines)
        print(testdataset)
        predictions = model.predict(testdataset)
        print(predictions)
    except NotFittedError as e:
        print(repr(e))
    return render_template('index.html', prediction_value="{}".format(predictions[0]))


if __name__ == '__main__':
    app.run(debug=True)
