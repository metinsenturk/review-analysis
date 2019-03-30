import logging
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
params = {"num_topics": 5, "chunksize": 500}


def _load_model(type, fname='../../model/'):
    try:
        if type == 'lsi':
            return LsiModel.load(fname)
        elif type == 'lda':
            return LdaModel.load(fname)
        elif type == 'mallet':
            return LdaMallet.load(fname)
    except:
        return None


def _save_model(model, fname):
    model.save(fname)


def create_doc_term_matrix(docs, tfidf=False, logentropy=False, random_projections=False):
    id2word = Dictionary(documents=docs)
    doc_term_matrix = [id2word.doc2bow(doc) for doc in docs]

    if random_projections:
        rp_model = RpModel(corpus=doc_term_matrix, id2word=id2word, num_topics=params['num_topics'])
        doc_term_matrix = rp_model[doc_term_matrix]

    if tfidf:
        tfidf_model = TfidfModel(id2word=id2word, corpus=doc_term_matrix, normalize=True)
        doc_term_matrix = tfidf_model[doc_term_matrix]
    
    if logentropy:
        log_model = LogEntropyModel(corpus=doc_term_matrix, normalize=True)
        doc_term_matrix = log_model[doc_term_matrix]

    return doc_term_matrix, id2word


def get_lsi_model(doc_term_matrix, id2word, fname, num_topics=None):
    """
    if fname is not None:
        try:
            return LsiModel.load(fname)
        except:
            pass
    """

    lsi_model = LsiModel(
        corpus=doc_term_matrix,
        id2word=id2word,
        num_topics=params['num_topics'] if num_topics is None else num_topics
    )

    _save_model(lsi_model, fname)

    return lsi_model


def get_lda_model(doc_term_matrix, id2word, fname, num_topics=None):
    """
    try:
        lda_model = LdaModel.load(fname)
    except:
        pass
    """
    
    lda_model = LdaModel(
            corpus=doc_term_matrix,
            id2word=id2word,
            num_topics=params['num_topics'] if num_topics is None else num_topics,
            passes=5,            
            per_word_topics=True
        )

    _save_model(lda_model, fname=fname)

    return lda_model


def get_lda_mallet_model(doc_term_matrix, id2word, fname):
    mallet_path = '../model/mallet/bin/mallet'
    """
    if fname is not None:
        try:
            LdaMallet(fname)
        except:
            pass
    """

    lda_mallet = LdaMallet(
        mallet_path=mallet_path,
        corpus=doc_term_matrix,
        id2word=id2word,
        workers=6,
        num_topics=params['num_topics']
    )

    _save_model(lda_mallet, fname=fname)

    return lda_mallet


def get_hdp_model(doc_term_matrix, id2word, fname):
    hdp_model = HdpModel(
        corpus=doc_term_matrix, 
        id2word=id2word
    )

    _save_model(hdp_model, fname=fname)

    return hdp_model


def get_coherence_model(model, doc_term_matrix, id2word, revs):

    coh_model = CoherenceModel(
        model=model, 
        corpus=doc_term_matrix, 
        dictionary=id2word,
        coherence='c_v'
    )

    return coh_model


def get_document_topics(model, doc_term_matrix, revs):
    """ scores topics to sentences first, then picks the mode for the doc and creates mode, sentence topics, and sentence topics with probabilities. """
    results = []

    try:
        i, j = 0, 0
        for index, rev in enumerate(revs):
            # get document count on each review
            j = j + len(rev)
            logger.info(f"model: {type(model)} doc => i: {i} j: {j} rev => {index + 1} of {len(revs)}")
            doc_model_list = model[doc_term_matrix[i:j]]
            i = j

            # get topic prob distribution for each document in review, based on max probability
            in_doc_topic_prob_list = []

            if type(model) == LdaModel:
                in_doc_topic_prob_list = [max(sent_topic_list, key=lambda x: x[1]) if len(sent_topic_list) > 0 else (np.nan, 0.0) for sent_topic_list, y, z in doc_model_list]
            else:
                in_doc_topic_prob_list = [max(sent_topic_list, key=lambda x: x[1]) if len(sent_topic_list) > 0 else (np.nan, 0.0) for sent_topic_list in doc_model_list]                

            # topics
            doc_topic_list = [topic for topic, prob in in_doc_topic_prob_list]
            doc_topics = ",".join([str(topic) for topic in doc_topic_list])
            doc_topic_mode = max(doc_topic_list, key=doc_topic_list.count)

            results.append((doc_topic_mode, doc_topics, in_doc_topic_prob_list))
    except Exception as ex:
        logger.warning("scoring problem.", exc_info=ex)

    return results
