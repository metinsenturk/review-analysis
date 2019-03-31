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


def get_pipe(tfidf=True):
    steps = []
    steps.append(('cnt', CountVectorizer()))

    if tfidf:
        steps.append(('tdf', TfidfTransformer()))

    pipe = Pipeline(steps=steps)

    return pipe


def sgd_model(X, y, pipe, name):
    pipe.steps.append((name, SGDClassifier(
        n_jobs=-1,
        loss='hinge',
        penalty='l2',
        max_iter=1000,
        tol=0.05
    )))
    pipe.fit(X, y)

    _save_model(pipe, f'sgd_model/{name}')

    return pipe


def log_model(X, y, pipe, name):
    pipe.steps.append((name, LogisticRegression(
        n_jobs=-1,
        solver='sag',
    )))
    pipe.fit(X, y)

    _save_model(pipe, f'log_model/{name}')

    return pipe


def mnb_model(X, y, pipe, name):
    pipe.steps.append((name, MultinomialNB()))
    pipe.fit(X, y)

    _save_model(pipe, f'mnb_model/{name}')

    return pipe


def rdg_model(X, y, pipe, name):
    pipe.steps.append((name, RidgeClassifier()))
    pipe.fit(X, y)

    _save_model(pipe, f'rdg_model/{name}')

    return pipe


def get_document_sentiments(model, X, y):
    y_pred = model.predict(X)

    return y_pred
