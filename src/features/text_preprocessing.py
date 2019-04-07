import numpy as np
import pandas as pd

import spacy

from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk import WordNetLemmatizer, PorterStemmer


class SpaCyProcessing:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm", disable=['ner'])

    def _token_cleanup(self, token):
        """ token cleanup. return clean token or None. """
        removal = ['ADV', 'PRON', 'CCONJ',
                   'PUNCT', 'PART', 'DET', 'ADP', 'SPACE']
        if token.is_stop == False and token.is_alpha and len(token) > 3 and token.pos_ not in removal:
            lemma = token.lemma_
            return lemma

    def doc_clean_up(self,  text):
        """ clean up tokens by documents """
        doc = self.nlp(text)
        text_out = []

        for token in doc:
            token_clean = self._token_cleanup(token)
            if token_clean is not None:
                text_out.append(token_clean)
        return text_out

    def doc_sent_clean_up(self, text, clean_up=True):
        """ clean up tokens by sents in documents """
        doc = self.nlp(text)
        texts = []

        sent_current = ""
        for token in doc:
            # check for tokens current sent
            if sent_current != token.sent.text:
                # add it to texts, if it is not initially
                if sent_current != "":
                    texts.append(sent)
                # update current sent index
                sent_current = token.sent.text
                # create sent list and add first token
                sent = []
                if clean_up:
                    token_clean = self._token_cleanup(token)
                    if token_clean is not None:
                        sent.append(token_clean)
                else:
                    sent.append(self._token_cleanup(token))
            else:
                # add same sent tokens to the sent list
                if clean_up:
                    token_clean = self._token_cleanup(token)
                    if token_clean is not None:
                        sent.append(self._token_cleanup(token))
                else:
                    sent.append(token)
        
        # add the last sentence to the list
        texts.append(sent)

        return texts


class NLTKProcessing:
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
