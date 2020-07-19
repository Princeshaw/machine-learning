from flask import Flask, jsonify, request, render_template
import pickle
import numpy as np
import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib

NB_spam_model = open('model.pkl','rb')
model = joblib.load(NB_spam_model)


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
    
    model=pickle.load(open('model.pkl','rb'))
    new_X_test = cv.transform(new_corpus).toarray()
    new_y_pred = model.predict(new_X_test)
    return(new_y_pred[0])
    
@app.route('/predict',methods=['POST','GET'])
def predict_review():
    dataset = pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)
    corpus = []
    for i in range(0,1000):
        review = re.sub('[^a-zA-z]',' ',dataset['Review'][i])
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        all_stopwords = stopwords.words('english')
        all_stopwords.remove('not')
        review = [ps.stem(word) for word in review if word not in set(all_stopwords)]
        review = " ".join(review)
        corpus.append(review)
    cv = CountVectorizer()
    X = cv.fit_transform(corpus).toarray()
    y = dataset.iloc[:,-1].values  
    
    if request.method == 'POST':
        new_review = request.form['review']
        #print(review)
        new_review = re.sub('[^a-zA-Z]', ' ', new_review)
        new_review = new_review.lower()
        new_review = new_review.split()
        ps = PorterStemmer()
        all_stopwords = stopwords.words('english')
        all_stopwords.remove('not')
        new_review = [ps.stem(word) for word in new_review if not word in set(all_stopwords)]
        new_review = ' '.join(new_review)
        new_corpus = [new_review]
        new_X_test = cv.transform(new_corpus).toarray()
        result  = model.predict(new_X_test)
        
        #return render_template('index.html',pred="done")
        #print(int(result[0]))
        if(int(result[0])==1):
            return render_template('index.html',pred='This is a positive review')
        else:
            return render_template('index.html',pred='This is a negative review')

app.run() 
