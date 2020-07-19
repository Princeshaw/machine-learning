import numpy as np   
import pandas as pd  
import matplotlib.pyplot as plt
import warnings
import pickle
warnings.filterwarnings("ignore")

dataset = pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
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
    
print(corpus)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,-1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)


pickle.dump(gnb,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))    