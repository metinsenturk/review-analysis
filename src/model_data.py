import itertools
import logging
import time
import multiprocessing
from multiprocessing import Queue
from multiprocessing.pool import Pool
from multiprocessing.context import Process

import pandas as pd
import numpy as np

from models import topic_analysis

# log configuration
logger = logging.getLogger(__name__)
logging.getLogger("gensim").setLevel(logging.WARNING)
logger_multi = multiprocessing.log_to_stderr()
logger_multi.setLevel(logging.INFO)


def get_lsi_results(doc_term_matrix, id2word, revs, fname, output=None):
    logger_multi.info(f"{fname} started")
    time_start = time.time()
    lsi_model = topic_analysis.get_lsi_model(
        doc_term_matrix, id2word, f'../model/lsi_model/{fname}.model')
    logger_multi.info(f"lsi took {time.time() - time_start} seconds to complete the model.")
    doc_topic_tuples = topic_analysis.get_document_topics(
        lsi_model, doc_term_matrix, revs)

    logger_multi.info('results completed. putting in queue..')
    output.put((fname, doc_topic_tuples))
    logger_multi.info('results sent.')
    
    return doc_topic_tuples


def get_lda_results(doc_term_matrix, id2word, revs, fname, output=None):
    logger_multi.info(f"{fname} started")
    time_start = time.time()
    lda_model = topic_analysis.get_lda_model(
        doc_term_matrix, id2word, f'../model/lda_model/{fname}.model')
    logger_multi.info(f"lda took {time.time() - time_start} seconds to complete the model.")
    doc_topic_tuples = topic_analysis.get_document_topics(
        lda_model, doc_term_matrix, revs)

    logger_multi.info('results completed. putting in queue..')
    output.put((fname, doc_topic_tuples))
    logger_multi.info('results sent.')

    return doc_topic_tuples


def get_mallet_results(doc_term_matrix, id2word, revs, fname, output=None):
    logger_multi.info(f"{fname} started")
    time_start = time.time()
    mallet_model = topic_analysis.get_lda_mallet_model(
        doc_term_matrix, id2word, f'../model/mallet_model/{fname}.model')
    logger_multi.info(f"mallet took {time.time() - time_start} seconds to complete the model.")
    doc_topic_tuples = topic_analysis.get_document_topics(
        mallet_model, doc_term_matrix, revs)

    logger_multi.info('results completed. putting in queue..')
    output.put((fname, doc_topic_tuples))
    logger_multi.info('results sent.')

    return doc_topic_tuples


def run_topic_models(tokens_list, to_file=None, lsi=True, lda=True, mallet=True):
    """ gets normalized tokens and returns topics for all algortithms. """
    
    # sentences
    revs = tokens_list
    docs = list(itertools.chain(*revs))
    logger.info(f"number of reviews and documents => revs: {len(revs)} docs: {len(docs)}")

    # topic modeling
    doc_term_matrix, id2word = topic_analysis.create_doc_term_matrix(docs)
    doc_term_matrix_tfidf, id2word = topic_analysis.create_doc_term_matrix(docs, tfidf=True)
    
    # topic models in multiprocessing
    output = Queue()
    processes = []
    results = []

    if lsi:
        p_lsi = Process(name='lsi', target=get_lsi_results, args=(
            doc_term_matrix, id2word, revs, 'lsi', output))
        processes.append(p_lsi)
    if lda:    
        p_lda = Process(name='lda', target=get_lda_results, args=(
            doc_term_matrix, id2word, revs, 'lda', output))
        processes.append(p_lda)
        #p_lda_tfidf = Process(name='lda_tfidf', target=get_lda_results, args=(
            doc_term_matrix_tfidf, id2word, revs, 'lda_tfidf', output))
        #processes.append(p_lda_tfidf)
    if mallet:    
        p_mallet = Process(name='mallet', target=get_mallet_results, args=(
            doc_term_matrix, id2word, revs, 'mallet', output))
        processes.append(p_mallet)

    for process in processes:
        logger.info("{} process is started.".format(process.name))
        process.start()

    for process in processes:
        process.join(10)
        logger.info("{} process is completed".format(process.name))
        results.append(output.get())
        logger.info("output received.")

    output.close()
    output.join_thread()

    for process in processes:
        logger.info("process status for {}: {}".format(process.name, process.is_alive()))

    # merging results
    logger.info("saving results in file")
    pd_topics=[]
    for name, tuples_list in results:
        column_names=['topic_mode', 'topic_mode_prob', 'topic_list']
        pd_topic=pd.DataFrame(tuples_list, columns=[
                              name + '_' + col for col in column_names])
        logger.info(f"topics for {name} has shape of : {pd_topic.shape}")
        pd_topics.append(pd_topic)

    # saving results
    df_topics=pd.concat(pd_topics, axis=1)
    logger.info(f"topics has shape of : {df_topics.shape}")
    if to_file is not None:
        df_topics.to_csv(f'../data/processed/{to_file}', index=False)
        logger.info("file saved.")
    
    return df_topics
