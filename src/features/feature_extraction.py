import numpy as np
import pandas as pd

from nltk import word_tokenize, sent_tokenize
from nltk import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
from string import punctuation

lemmatizer = WordNetLemmatizer()
porter = PorterStemmer()
stops = stopwords.words('english')

def _processing(word):
    """ remove punctuation and stopwords from sentences. """
    word = word.lower()
    if word not in stops and word not in punctuation:
        return porter.stem(lemmatizer.lemmatize(word))

def get_sent_tokens(doc: str):
    """ get str documents and returns sent tokens as list. """
    return sent_tokenize(doc)

def get_word_tokens(doc: str):
    """ get str documents and return word tokens as list. """
    return word_tokenize(doc)

def get_norm_tokens(doc: list, stemming=True, lemmatizer=True, lower=True, punctuation=True, stopwords=True):
    """ gets a *list* of words and returns normalized versions of the words. """
    for word in doc:
        if stopwords:
            if word in stops:
                continue
            else:
                pass

        if punctuation:
            if word in punctuation:
                continue
            else:
                pass

        if lower:
            word = word.lower()
        
        if lemmatizer:
            word = lemmatizer.lemmatize(word)

        if stemming:
            word = porter.stem(word)
    
    return doc



