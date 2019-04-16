import logging
import pickle
import statistics
import pandas as pd
import numpy as np

from gensim.corpora import Dictionary
from gensim.models import LsiModel
from gensim.models.ldamulticore import LdaModel
from gensim.models.wrappers import LdaMallet
from gensim.models import HdpModel

from gensim.models import RpModel
from gensim.models import TfidfModel
from gensim.models import LogEntropyModel

from gensim.models import CoherenceModel

logger = logging.getLogger(__name__)
params = {"num_topics": 5, "chunksize": 500, "training": True}


def _load_model(model_type, fname):
    logger.info(f'{model_type} type of {fname} is loading..')
    try:
        if model_type == 'lsi':
            return LsiModel.load(f'../model/lsi_model/{fname}')
        elif model_type == 'lda':
            return LdaModel.load(f'../model/lda_model/{fname}')
        elif model_type == 'mallet':
            return LdaMallet.load(f'../model/mallet_model/{fname}')
        elif model_type == 'hdp':
            return HdpModel.load(f'../model/mallet_model/{fname}')
    except Exception as ex:
        logger.warning(
            f'{model_type} type of {fname} could not be loaded.', exc_info=ex)
        return None


def _save_model(model_type, model, fname):
    logger.info(f'{model_type} type of {fname} is saved.')
    try:
        if model_type == 'lsi':
            return model.save(fname=f'../model/lsi_model/{fname}')
        elif model_type == 'lda':
            return model.save(fname=f'../model/lda_model/{fname}')
        elif model_type == 'mallet':
            return model.save(fname=f'../model/mallet_model/{fname}')
        elif model_type == 'hdp':
            return model.save(fname=f'../model/mallet_model/{fname}')
    except Exception as ex:
        logger.warning(f'{fname} could not be saved.', exc_info=ex)
        return None


def _save_model2(model, name):
    try:
        with open(f'../model/dict_and_matrix/{name}.pkl', 'wb') as f:
            pickle.dump(model, f)
            logger.info(f'{name} is saved.')
    except Exception as ex:
        logger.warning(f"{name} could not be saved.", exc_info=ex)


def create_dictionary(docs):
    id2word = Dictionary(documents=docs)
    id2word.filter_extremes(no_below=10, no_above=0.3)
    _save_model2(id2word, 'id2word')
    return id2word


def create_doc_term_matrix(docs, id2word, tfidf=False, logentropy=False, random_projections=False):
    doc_term_matrix = [id2word.doc2bow(doc) for doc in docs]
    _save_model2(doc_term_matrix, 'doc_term_matrix')

    if random_projections:
        rp_model = RpModel(corpus=doc_term_matrix,
                           id2word=id2word, num_topics=params['num_topics'])
        doc_term_matrix = rp_model[doc_term_matrix]
        _save_model2(doc_term_matrix, 'doc_term_matrix_random_projections')

    if tfidf:
        tfidf_model = TfidfModel(
            id2word=id2word, corpus=doc_term_matrix, normalize=True)
        doc_term_matrix = tfidf_model[doc_term_matrix]
        _save_model2(doc_term_matrix, 'doc_term_matrix_tfidf')

    if logentropy:
        log_model = LogEntropyModel(corpus=doc_term_matrix, normalize=True)
        doc_term_matrix = log_model[doc_term_matrix]
        _save_model2(doc_term_matrix, 'doc_term_matrix_logentropy')

    return doc_term_matrix


def get_lsi_model(doc_term_matrix, id2word, fname, num_topics=None):

    if params['training']:
        lsi_model = LsiModel(
            corpus=doc_term_matrix,
            id2word=id2word,
            num_topics=params['num_topics'] if num_topics is None else num_topics
        )
        _save_model('lsi', lsi_model, fname)
    else:
        lsi_model = _load_model('lsi', fname)

    return lsi_model


def get_lsi_model2(doc_term_matrix, id2word, fname, num_topics=None):

    if params['training']:
        lsi_model = LsiModel(
            corpus=doc_term_matrix,
            id2word=id2word,
            chunksize=5000,
            decay=0.7,
            onepass=False,
            power_iters=10,
            extra_samples=1000,
            num_topics=params['num_topics'] if num_topics is None else num_topics
        )
        _save_model('lsi', lsi_model, fname)
    else:
        lsi_model = _load_model('lsi', fname)

    return lsi_model


def get_lda_model(doc_term_matrix, id2word, fname, num_topics=None):

    if params['training']:
        lda_model = LdaModel(
            corpus=doc_term_matrix,
            id2word=id2word,
            num_topics=params['num_topics'] if num_topics is None else num_topics,
            passes=5,
            per_word_topics=True
        )
        _save_model('lda', lda_model, fname=fname)
    else:
        lda_model = _load_model('lda', fname)

    return lda_model


def get_lda_model2(doc_term_matrix, id2word, fname, num_topics=None):

    if params['training']:
        lda_model = LdaModel(
            corpus=doc_term_matrix,
            id2word=id2word,
            num_topics=params['num_topics'] if num_topics is None else num_topics,
            chunksize=3000,
            passes=20,
            alpha='auto',
            # eta='auto',
            iterations=100,
            per_word_topics=True
        )
        _save_model('lda', lda_model, fname=fname)
    else:
        lda_model = _load_model('lda', fname)

    return lda_model


def get_lda_mallet_model(doc_term_matrix, id2word, fname):
    mallet_path = '../model/mallet/bin/mallet'

    if params['training']:
        lda_mallet = LdaMallet(
            mallet_path=mallet_path,
            corpus=doc_term_matrix,
            id2word=id2word,
            workers=6,
            num_topics=params['num_topics']
        )
        _save_model('mallet', lda_mallet, fname=fname)
    else:
        lda_mallet = _load_model('mallet', fname)

    return lda_mallet


def get_hdp_model(doc_term_matrix, id2word, fname):
    if params['training']:
        hdp_model = HdpModel(
            corpus=doc_term_matrix,
            id2word=id2word,
            max_chunks=10000,
            chunksize=2000,
            kappa=0.6,
            tau=32.0,
            eta=0.05,

        )
        _save_model('hdp', hdp_model, fname=fname)
    else:
        hdp_model = _load_model('hdp', fname)

    return hdp_model


def get_coherence_model(model, doc_term_matrix, id2word, revs):

    coh_model = CoherenceModel(
        model=model,
        corpus=doc_term_matrix,
        dictionary=id2word,
        coherence='c_v'
    )

    return coh_model


def get_document_topics(model, doc_term_matrix, revs, fname):
    """ scores topics to sentences first, then picks the mode for the doc and creates mode, sentence topics, and sentence topics with probabilities. """
    results = []

    try:
        i, j = 0, 0
        for index, rev in enumerate(revs):
            # get document count on each review
            j = j + len(rev)
            logger.debug(
                f"model: {fname} doc => i: {i} j: {j} rev => {index + 1} of {len(revs)}")
            doc_model_list = model[doc_term_matrix[i:j]]
            i = j

            # get topic prob distribution for each document in review, based on max probability
            in_doc_topic_prob_list = []

            if type(model) == LdaModel:
                in_doc_topic_prob_list = [max(sent_topic_list, key=lambda x: x[1]) if len(
                    sent_topic_list) > 0 else (np.nan, 0.0) for sent_topic_list, y, z in doc_model_list]
            else:
                in_doc_topic_prob_list = [max(sent_topic_list, key=lambda x: x[1]) if len(
                    sent_topic_list) > 0 else (np.nan, 0.0) for sent_topic_list in doc_model_list]

            # topics
            doc_topic_list = [topic for topic, prob in in_doc_topic_prob_list]
            doc_topics = ",".join([str(topic) for topic in doc_topic_list])
            doc_topic_mode = max(doc_topic_list, key=doc_topic_list.count)

            results.append(
                (doc_topic_mode, doc_topics, in_doc_topic_prob_list))
    except Exception as ex:
        logger.warning(
            f"scoring problem for {fname}: => i: {i} j: {j}", exc_info=ex)

    return results


def get_document_topics2(model, doc_term_matrix, revs, fname):
    """ scores topics to sentences first, then picks the mode for the doc and creates mode, sentence topics, and sentence topics with probabilities. """
    results = []

    try:
        i, j = 0, 0
        for index, rev in enumerate(revs):
            # get document count on each review
            j = j + len(rev)
            logger.debug(
                f"model: {fname} doc => i: {i} j: {j} rev => {index + 1} of {len(revs)}")
            doc_model_list = model[doc_term_matrix[i:j]]
            # logger.info(f"model: {fname} doc => i: {i} j: {j} rev => {len(doc_model_list)} rev => {len(rev)}")
            i = j

            # get topic prob distribution for each document in review, based on max probability
            in_doc_topic_prob_list = []

            if type(model) == LdaModel:
                for sent_topic_list, y, z in doc_model_list:
                    in_doc_topic_prob = max(sent_topic_list, key=lambda x: x[1]) if len(
                        sent_topic_list) > 0 else (np.nan, 0.0)
                    in_doc_topic_prob_list.append(in_doc_topic_prob)

                logger.info(
                    f"rev: {len(rev)} in_doc: {len(in_doc_topic_prob)} rev_sent: {len(sent_topic_list)}")
                # logger.info(f"{sent_topic_list}, {y}, {z}")
            else:
                in_doc_topic_prob_list = [max(sent_topic_list, key=lambda x: x[1]) if len(
                    sent_topic_list) > 0 else (np.nan, 0.0) for sent_topic_list in doc_model_list]

            # topics
            doc_topic_list = [topic for topic, prob in in_doc_topic_prob_list]
            doc_topics = ",".join([str(topic) for topic in doc_topic_list])
            doc_topic_mode = max(doc_topic_list, key=doc_topic_list.count)

            results.append(
                (doc_topic_mode, doc_topics, in_doc_topic_prob_list))
    except Exception as ex:
        logger.warning(
            f"scoring problem for {fname}: => i: {i} j: {j}", exc_info=ex)

    return results
