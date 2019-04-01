import time
import pickle
import logging

import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier

logger = logging.getLogger(__name__)
params = {
    'optimum': False, 
    'max_iter': [100, 50], #, 1000], # 2000, 5000], 
    'alpha': np.logspace(-5, 5, 2), 
    'C': [0.001], #, 0.01, 0.1, 1, 10, 100, 1000], 
    'tol':[0.001] #, 0.1, 1]
}


def _save_model(model, name):
    try:
        with open(f'../model/{name}.pkl', 'wb') as f:
            pickle.dump(model, f)
            logger.info(f'{name} is saved.')
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
    if params['optimum']:
        sgd_params = dict(
            sgd__loss=['log', 'modified_huber'],
            # sgd__learning_rate= ['optimal'], # ['constant', 'optimal', 'invscaling', 'adaptive'],
            sgd__alpha=params['alpha'],
            sgd__max_iter=params['max_iter'],
            sgd__tol=params['tol'],
        )
        pipe.steps.append((name, SGDClassifier()))
        pipe = get_gridsearchcsv_model(X, y, pipe, sgd_params, f'sgd_model/{name}')
    else:
        pipe.steps.append((name, SGDClassifier(
            n_jobs=-1,
            loss='hinge',
            penalty='l2',
            max_iter=1000,
            tol=0.005
        )))
        pipe.fit(X, y)

    logger.info(f'pipe is created for {name}.')
    _save_model(pipe, f'sgd_model/{name}')

    return pipe


def log_model(X, y, pipe, name):
    if params['optimum']:
        log_params = dict(
            log__C=params['C'], 
            log__tol=params['tol'], 
            log__max_iter=params['max_iter'],
            log__penalty=['l2'], 
            log__solver=['newton-cg', 'lbfgs', 'sag'], 
        )
        pipe.steps.append((name, LogisticRegression()))
        pipe = get_gridsearchcsv_model(X, y, pipe, log_params, f'log_model/{name}')
    else:
        pipe.steps.append((name, LogisticRegression(
            n_jobs=-1,
            solver='sag',
        )))
        pipe.fit(X, y)

    logger.info(f'pipe is created for {name}.')
    _save_model(pipe, f'log_model/{name}')

    return pipe


def mnb_model(X, y, pipe, name):
    if params['optimum']:
        mnb_params = dict(
            mnb__alpha=params['alpha']
        )
        pipe.steps.append((name, MultinomialNB()))
        pipe = get_gridsearchcsv_model(X, y, pipe, mnb_params, f'mnb_model/{name}')
    else:
        pipe.steps.append((name, MultinomialNB()))
        pipe.fit(X, y)

    logger.info(f'pipe is created for {name}.')
    _save_model(pipe, f'mnb_model/{name}')

    return pipe


def rdg_model(X, y, pipe, name):
    if params['optimum']:
        rdg_params = dict(
            # rdg__solver=['auto'],
            rdg__alpha=params['alpha'],
            rdg__tol=params['tol'],
            rdg__max_iter=params['max_iter'],
        )
        pipe.steps.append((name, RidgeClassifier()))
        pipe = get_gridsearchcsv_model(X, y, pipe, rdg_params, f'rdg_model/{name}')
    else:
        pipe.steps.append((name, RidgeClassifier()))
        pipe.fit(X, y)

    logger.info(f'pipe is created for {name}.')
    _save_model(pipe, f'rdg_model/{name}')

    return pipe


def get_gridsearchcsv_model(X, y, pipe, model_params, name):
    pipe_params = dict(
        # cnt__max_df=[0.05, 0.01],
        # cnt__min_df=[0.05, 0.01],
        cnt__ngram_range=[(1,1)] #, (1,2), (1,3)],
        # cnt__stop_words=[None, 'english']
    )

    gscv_params = {**pipe_params, **model_params}
    scoring = ('accuracy', 'balanced_accuracy', 'average_precision', 'brier_score_loss', 'recall', 'roc_auc') # 'f1_micro', 'f1_macro', 'f1_weighted', 'f1', 'precision',
    
    logger.info('starting grid search cv..')
    gscv = GridSearchCV(
        estimator=pipe,
        param_grid=gscv_params,
        scoring=scoring,
        refit='accuracy',
        iid=False,
        return_train_score=True,
        n_jobs=-1,
        cv=StratifiedKFold(n_splits=10)
    )
    t_start = time.time()
    logger.info(f'gscv is created for {name}.')
    gscv.fit(X, y)
    logger.info(f'gscv finished for {name}. It took {time.time() - t_start} seconds')

    folder_path, file_name = name.split('/')
    _save_model(gscv, f'{folder_path}/gscv_{file_name}')

    return gscv.best_estimator_


def get_document_sentiments(model, X, y):
    y_pred = model.predict(X)

    return y_pred
