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
from models import sentiment_analysis

# log configuration
logger = logging.getLogger(__name__)
logging.getLogger("gensim").setLevel(logging.WARNING)
logger_multi = multiprocessing.log_to_stderr()
logger_multi.setLevel(logging.INFO)


def get_lsi_results(doc_term_matrix, id2word, revs, fname, num_topics= None,output=None):
    logger_multi.info(f"{fname} started")
    time_start = time.time()
    lsi_model = topic_analysis.get_lsi_model(
        doc_term_matrix, id2word, f'../model/lsi_model/{fname}.model', num_topics)
    logger_multi.info(f"{fname} took {time.time() - time_start} seconds to complete the model.")
    doc_topic_tuples = topic_analysis.get_document_topics(
        lsi_model, doc_term_matrix, revs)

    logger_multi.info('results completed. putting in queue..')
    output.put((fname, doc_topic_tuples))
    logger_multi.info('results sent.')
    
    return doc_topic_tuples


def get_lda_results(doc_term_matrix, id2word, revs, fname, num_topics=None, output=None):
    logger_multi.info(f"{fname} started")
    time_start = time.time()
    lda_model = topic_analysis.get_lda_model(
        doc_term_matrix, id2word, f'../model/lda_model/{fname}.model', num_topics)
    logger_multi.info(f"{fname} took {time.time() - time_start} seconds to complete the model.")
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
    logger_multi.info(f"{fname} took {time.time() - time_start} seconds to complete the model.")
    doc_topic_tuples = topic_analysis.get_document_topics(
        mallet_model, doc_term_matrix, revs)

    logger_multi.info('results completed. putting in queue..')
    output.put((fname, doc_topic_tuples))
    logger_multi.info('results sent.')

    return doc_topic_tuples

def get_hdp_results(doc_term_matrix, id2word, revs, fname, output=None):
    logger_multi.info(f"{fname} started")
    time_start = time.time()
    mallet_model = topic_analysis.get_hdp_model(
        doc_term_matrix, id2word, f'../model/mallet_model/{fname}.model')
    logger_multi.info(f"{fname} took {time.time() - time_start} seconds to complete the model.")
    doc_topic_tuples = topic_analysis.get_document_topics(
        mallet_model, doc_term_matrix, revs)

    logger_multi.info('results completed. putting in queue..')
    output.put((fname, doc_topic_tuples))
    logger_multi.info('results sent.')

    return doc_topic_tuples


def run_topic_models(tokens_list, to_file=None, transformations=False, find_optimal_num_topics=False, lsi=True, lda=True, mallet=True, hdp=True):
    """ gets normalized tokens and returns topics for all algortithms. """
    
    # sentences
    revs = tokens_list
    docs = list(itertools.chain(*revs))
    logger.info(f"number of reviews and documents => revs: {len(revs)} docs: {len(docs)}")

    # topic modeling
    doc_term_matrix, id2word = topic_analysis.create_doc_term_matrix(docs)
    doc_term_matrix_tfidf, id2word = topic_analysis.create_doc_term_matrix(docs, tfidf=True)
    doc_term_matrix_random_projections, id2word = topic_analysis.create_doc_term_matrix(docs, random_projections=True)
    doc_term_matrix_logentropy, id2word = topic_analysis.create_doc_term_matrix(docs, logentropy=True)
    
    # topic models in multiprocessing
    output = Queue()
    processes = []
    results = []

    if lsi:
        p_lsi = Process(name='lsi', target=get_lsi_results, args=(
                doc_term_matrix, id2word, revs, 'lsi', None, output))
        processes.append(p_lsi)

        if find_optimal_num_topics:
            for num_topics in range(2, 21, 3):
                p_lsi = Process(name=f'lsi_{num_topics}', target=get_lsi_results, args=(
                    doc_term_matrix, id2word, revs, f'lsi_{num_topics}', num_topics, output))
                processes.append(p_lsi)
        
        if transformations:
            p_lsi = Process(name='lsi_tfidf', target=get_lsi_results, args=(
                doc_term_matrix_tfidf, id2word, revs, 'lsi_tfidf', None, output))
            processes.append(p_lsi)

            p_lsi = Process(name='lsi_logentropy', target=get_lsi_results, args=(
                doc_term_matrix_logentropy, id2word, revs, 'lsi_logentropy', None, output))
            processes.append(p_lsi)

            p_lsi = Process(name='lsi_random_projections', target=get_lsi_results, args=(
                doc_term_matrix_random_projections, id2word, revs, 'lsi_random_projections', None, output))
            processes.append(p_lsi)
    if lda:
        p_lda = Process(name='lda', target=get_lda_results, args=(
                    doc_term_matrix, id2word, revs, 'lda', None, output))
        processes.append(p_lda)

        if find_optimal_num_topics:
            for num_topics in range(2, 21, 3):
                p_lda = Process(name=f'lda_{num_topics}', target=get_lda_results, args=(
                    doc_term_matrix, id2word, revs, f'lda_{num_topics}', num_topics, output))
                processes.append(p_lda)
        
        if transformations:
            p_lda = Process(name='lda_tfidf', target=get_lda_results, args=(
                doc_term_matrix_tfidf, id2word, revs, 'lda_tfidf', None, output))
            processes.append(p_lda)

            p_lda = Process(name='lda_logentropy', target=get_lda_results, args=(
                doc_term_matrix_logentropy, id2word, revs, 'lda_logentropy', None, output))
            processes.append(p_lda)

            p_lda = Process(name='lda_random_projections', target=get_lda_results, args=(
                doc_term_matrix_random_projections, id2word, revs, 'lda_random_projections', None, output))
            processes.append(p_lda)
    if mallet:    
        p_mallet = Process(name='mallet', target=get_mallet_results, args=(
            doc_term_matrix, id2word, revs, 'mallet', output))
        processes.append(p_mallet)

    if hdp:
        p_hdp = Process(name='hdp', target=get_mallet_results, args=(
            doc_term_matrix, id2word, revs, 'hdp', output))
        
        if transformations:
            p_hdp = Process(name='hdp_tfidf', target=get_hdp_results, args=(
                doc_term_matrix_tfidf, id2word, revs, 'hdp_tfidf', output))
            processes.append(p_hdp)

            p_hdp = Process(name='hdp_logentropy', target=get_hdp_results, args=(
                doc_term_matrix_logentropy, id2word, revs, 'hdp_logentropy', output))
            processes.append(p_hdp)

            p_hdp = Process(name='hdp_random_projections', target=get_hdp_results, args=(
                doc_term_matrix_random_projections, id2word, revs, 'hdp_random_projections', output))
            processes.append(p_hdp)

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


def run_sentiment_models(revs_list, sentiment_list, to_file, sgd=True, log=True, mnb=True, rdg=True):
    
    X = revs_list
    y = sentiment_list

    logger.info(f"number of items in reviews and sentiments: revs => {len(X)} sentiments: {len(y)}")

    models = [
        ('log', sentiment_analysis.log_model),
        ('mnb', sentiment_analysis.mnb_model),
        ('rdg', sentiment_analysis.rdg_model),
        ('sgd', sentiment_analysis.sgd_model),
    ]

    results = []

    for model_name, fn_model in models:
        clf = fn_model(X, y, model_name)
        y_pred = sentiment_analysis.get_document_sentiments(clf, X, y)
        results.append(pd.Series(y_pred, name=model_name))

    # saving into file
    df_sentiments = pd.concat(results, axis=1)
    logger.info(f"sentiments has shape of : {df_sentiments.shape}")
    if to_file is not None:
        df_sentiments.to_csv(f'../data/processed/{to_file}', index=False)
        logger.info("file saved.")
    
    return df_sentiments
    