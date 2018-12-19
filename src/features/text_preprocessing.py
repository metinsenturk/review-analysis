import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import json

from textblob import TextBlob
from textblob import Word
import nltk
from nltk.stem import PorterStemmer
try:
    from nltk.corpus import stopwords
except:
    nltk.download('stopwords')

class preprocessing:
    def __init__(self, text_list):
        freq = list()
        rare = list()
        word_list = ' '.join(text_list).split()

        self.freq = pd.Series(word_list).value_counts()[:10]
        self.freq = list(freq.index)
        self.rare = pd.Series(word_list).value_counts()[-10:]
        self.rare = list(rare.index)

        self.st = PorterStemmer()    

    def lower(self, text):
        text_arr = [x.lower() for x in text.split()]
        return(' '.join(text_arr))

    def punctuation(self, text):
        text = text.replace("[^a-zA-Z#]", " ")
        # text = text.replace('[^\w\s]','')
        return(text)

    def shortwords(self, text):        
        return (' '.join([w for w in text.split() if len(w)>2]))

    def stopwords(self, text):
        stopwords_list = set(stopwords.words('english'))
        text = ' '.join(x for x in text.split() if x not in stopwords_list)
        return(text)

    def freqwords(self, text):
        text = " ".join(x for x in text.split() if x not in self.freq)
        return(text)
    
    def rarewords(self, text):
        text = " ".join(x for x in text.split() if x not in self.rare)

    def spellcheck(self, text):
        return(str(TextBlob(text).correct()))

    def tokenize(self, text):
        return(" ".join(TextBlob(str(text)).words))

    def stemming(self, text):
        text = " ".join([self.st.stem(word) for word in text.split()])
        return(text)

    def lemnatize(self, text):
        text = " ".join([Word(word).lemmatize() for word in text.split()])
        return(text)