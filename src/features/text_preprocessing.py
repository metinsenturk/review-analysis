import numpy as np
import pandas as pd

from nltk import word_tokenize
from nltk import WordNetLemmatizer, PorterStemmer

try:
    from nltk.corpus import stopwords
except:
    nltk.download('stopwords')
    from nltk.corpus import stopwords


class preprocessing:
    def __init__(self, text_list: list):
        self.stopwords_list = stopwords.words('english')

        word_list = ' '.join(text_list).split()
        word_list = [x for x in word_list if x not in self.stopwords_list]

        freq = pd.Series(word_list).value_counts()[:10]
        self.freq = list(freq.index)

        rare = pd.Series(word_list).value_counts()[-10:]
        self.rare = list(rare.index)

        self.st = PorterStemmer()
        self.lm = WordNetLemmatizer()

    def lower(self, text):
        text_arr = [x.lower() for x in text.split()]
        return(' '.join(text_arr))

    def punctuation(self, text):
        text = text.replace("[^a-zA-Z#]", " ")
        # text = text.replace('[^\w\s]','')
        return(text)

    def shortwords(self, text):
        return (' '.join([w for w in text.split() if len(w) > 2]))

    def stopwords(self, text):
        text = ' '.join(x for x in text.split()
                        if x not in self.stopwords_list)
        return(text)

    def freqwords(self, text):
        text = " ".join(x for x in text.split() if x not in self.freq)
        return(text)

    def rarewords(self, text):
        text = " ".join(x for x in text.split() if x not in self.rare)
        return(text)

    def spellcheck(self, text):
        # below code will work. TextBlob needed.
        # return(str(TextBlob(text).correct()))
        pass

    def tokenize(self, text):
        return(" ".join(word_tokenize(text)))

    def stemming(self, text):
        text = " ".join([self.st.stem(word) for word in text.split()])
        return(text)

    def lemnatize(self, text):
        text = " ".join(self.lm.lemmatize(word) for word in text.split())
        return(text)
