import re
import nltk
import pandas as pd
from pandas.io.json import json_normalize
import json

nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer 

json_data = {}

with open('datasets/yp_leilanis-lahaina-2_rws.json') as f:
    json_data = json.loads(f.read())

dataset = json_normalize(json_data['reviews'])

reviews = dataset.iloc[:, 2].values


# clean text
corpus = []

for review in reviews:
    review = re.sub('[^a-zA-Z]', ' ', review)

    # convert all cases to lower cases 
    review = review.lower()  
      
    # split to array(default delimiter is " ") 
    review = review.split()  
      
    # creating PorterStemmer object to 
    # take main stem of each word 
    ps = PorterStemmer()  

    #lem = WordNetLemmatizer()

    #lem.lemmatize(word, "v")
    
    # loop for stemming each word 
    # in string array at ith row     
    review = [ps.stem(word) for word in review 
                if not word in set(stopwords.words('english'))]  
                  
    # rejoin all string array elements 
    # to create back into a string 
    review = ' '.join(review)   
      
    # append each string to create 
    # array of clean text  
    corpus.append(review)  


# Creating the Bag of Words model 
from sklearn.feature_extraction.text import CountVectorizer 
  
# To extract max 1500 feature. 
# "max_features" is attribute to 
# experiment with to get better results 
cv = CountVectorizer(max_features = 1500)  
  
# X contains corpus (dependent variable) 
X = cv.fit_transform(corpus).toarray()  
  
# y contains answers if review 
# is positive or negative 
y = dataset.iloc[:, 3].apply(lambda x: 1 if x > 3 else 0)

# Splitting the dataset into 
# the Training set and Test set 
from sklearn.cross_validation import train_test_split 

# experiment with "test_size" 
# to get better results 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25) 


# Fitting Random Forest Classification 
# to the Training set 
from sklearn.ensemble import RandomForestClassifier 

# n_estimators can be said as number of 
# trees, experiment with n_estimators 
# to get better results 
model = RandomForestClassifier(n_estimators = 501, 
							criterion = 'entropy') 
							
model.fit(X_train, y_train) 

# Predicting the Test set results 
y_pred = model.predict(X_test) 

y_pred 

# Making the Confusion Matrix 
from sklearn.metrics import confusion_matrix 

cm = confusion_matrix(y_test, y_pred) 

cm 
