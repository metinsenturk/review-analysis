import pickle
import logging

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier

logger = logging.getLogger(__name__)

def _save_model(model, name):
    try:
        with open(f'../model/{name}.pkl', 'wb') as f:
            pickle.dump(model, f)
    except Exception as ex:
        logger.warning(f"{name} could not be saved.", exc_info=ex)

def _build_pipe(model, tfidf=True):
    steps = []
    steps.append(('cnt', CountVectorizer()))

    if tfidf:
        steps.append(('tdf', TfidfTransformer()))
    
    steps.append(model)
    clf = Pipeline(steps=steps)

    return clf

def sgd_model(X, y, name):
    clf = _build_pipe(('sgd', SGDClassifier(n_jobs=-1)))
    clf.fit(X, y)

    _save_model(clf, f'sgd_model/{name}')

    return clf

def log_model(X, y, name):
    clf = _build_pipe(('log', LogisticRegression(n_jobs=-1)))
    clf.fit(X, y)

    _save_model(clf, f'log_model/{name}')

    return clf

def mnb_model(X, y, name):
    clf = _build_pipe(('mnb', MultinomialNB()))
    clf.fit(X, y)

    _save_model(clf, f'mnb_model/{name}')

    return clf

def rdg_model(X, y, name):
    clf = _build_pipe(('rdg', RidgeClassifier()))
    clf.fit(X, y)

    _save_model(clf, f'rdg_model/{name}')

    return clf

def get_document_sentiments(model, X, y):
    y_pred = model.predict(X)

    return y_pred
    