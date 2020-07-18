from flask import Flask, jsonify, request, render_template
import pickle
import numpy as np
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

model=pickle.load(open('model.pkl','rb'))

app =Flask(__name__)


@app.route('/')
def Home():
    return render_template('index.html')



def new_prediction(new_review):
    new_review = re.sub('[^a-zA-Z]', ' ', new_review)
    new_review = new_review.lower()
    new_review = new_review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    new_review = [ps.stem(word) for word in new_review if not word in set(all_stopwords)]
    new_review = ' '.join(new_review)
    new_corpus = [new_review]
    cv = CountVectorizer()
    new_X_test = cv.fit_transform(new_corpus).toarray()
    new_y_pred = model.predict(new_X_test)
    return(new_y_pred)

@app.route('/predict',methods=['POST','GET'])
def predict_review():
    if request.method == 'POST':
        review = request.form['review']
        print(review)
        result = new_prediction(review)
        print(int(result[0]))
        return render_template('index.html',pred="done")
        #print(int(result[0]))
        if(int(result[0])==1):
            return render_template('index.html',pred='This is a possitive review')
        else:
            return render_template('index.html',pred='This is a negative review')

app.run() 
