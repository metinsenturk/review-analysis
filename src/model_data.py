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
logging.getLogger("sklearn").setLevel(logging.FATAL)
logger_multi = multiprocessing.log_to_stderr()
logger_multi.setLevel(logging.INFO)


def get_lsi_results(doc_term_matrix, id2word, revs, fname, num_topics= None, output=None):
    try:
        logger_multi.info(f"{fname} started")
        time_start = time.time()
        lsi_model = topic_analysis.get_lsi_model2(
            doc_term_matrix, id2word, f'{fname}.model', num_topics)
        logger_multi.info(f"{fname} took {time.time() - time_start} seconds to complete the model.")
        doc_topic_tuples = topic_analysis.get_document_topics(
            lsi_model, doc_term_matrix, revs, fname)

        logger_multi.info('results completed. putting in queue..')
        output.put((fname, doc_topic_tuples))
        logger_multi.info('results sent.')
        
        return doc_topic_tuples
    except Exception as ex:
        logger.warning("get_lsi_results failed.", exc_info=ex)


def get_lda_results(doc_term_matrix, id2word, revs, fname, num_topics=None, output=None):
    try:
        logger_multi.info(f"{fname} started")
        time_start = time.time()
        lda_model = topic_analysis.get_lda_model2(
            doc_term_matrix, id2word, f'{fname}.model', num_topics)
        logger_multi.info(f"{fname} took {time.time() - time_start} seconds to complete the model.")
        doc_topic_tuples = topic_analysis.get_document_topics(
            lda_model, doc_term_matrix, revs, fname)

        logger_multi.info('results completed. putting in queue..')
        output.put((fname, doc_topic_tuples))
        logger_multi.info('results sent.')
        
        return doc_topic_tuples
    except Exception as ex:
        logger.warning("get_lda_results failed.", exc_info=ex)


def get_mallet_results(doc_term_matrix, id2word, revs, fname, output=None):
    logger_multi.info(f"{fname} started")
    time_start = time.time()
    mallet_model = topic_analysis.get_lda_mallet_model(
        doc_term_matrix, id2word, f'{fname}.model')
    logger_multi.info(f"{fname} took {time.time() - time_start} seconds to complete the model.")
    doc_topic_tuples = topic_analysis.get_document_topics(
        mallet_model, doc_term_matrix, revs, fname)

    logger_multi.info('results completed. putting in queue..')
    output.put((fname, doc_topic_tuples))
    logger_multi.info('results sent.')

    return doc_topic_tuples

def get_hdp_results(doc_term_matrix, id2word, revs, fname, output=None):
    """ This is buggy. When it runs, it says likelihood is decreasing! """
    logger_multi.info(f"{fname} started")
    time_start = time.time()
    mallet_model = topic_analysis.get_hdp_model(
        doc_term_matrix, id2word, f'{fname}.model')
    logger_multi.info(f"{fname} took {time.time() - time_start} seconds to complete the model.")
    doc_topic_tuples = topic_analysis.get_document_topics(
        mallet_model, doc_term_matrix, revs, fname)

    logger_multi.info('results completed. putting in queue..')
    output.put((fname, doc_topic_tuples))
    logger_multi.info('results sent.')

    return doc_topic_tuples


def run_topic_models(revs_list, docs_list, to_file=None, transformations=False, find_optimal_num_topics=False, training=False, lsi=True, lda=True, mallet=True, hdp=False):
    """ gets normalized tokens and returns topics for all algortithms. """
    
    logger.info("topic algorithms starting..")

    # sentences
    revs = revs_list
    docs = docs_list
    logger.info(f"number of reviews and documents => revs: {len(revs)} docs: {len(docs)}")

    # dictionary
    id2word = topic_analysis.create_dictionary(docs)
    
    # doc_term_matrix
    doc_term_matrix = topic_analysis.create_doc_term_matrix(docs, id2word)
    doc_term_matrix_tfidf = topic_analysis.create_doc_term_matrix(docs, id2word, tfidf=True)
    doc_term_matrix_random_projections = topic_analysis.create_doc_term_matrix(docs, id2word, random_projections=True)
    doc_term_matrix_logentropy = topic_analysis.create_doc_term_matrix(docs, id2word, logentropy=True)
    logger.info("doc_term_matrixes and dictionaries created.")
    
    # choise of training
    topic_analysis.params['training'] = training

    # topic models in multiprocessing
    try:        
        output = Queue()
        processes = []
        results = []
        
        # debug
        # get_lda_results(doc_term_matrix_tfidf, id2word, revs, 'lda_tfidf', None, output)
        
        if lsi:
            if find_optimal_num_topics:
                for num_topics in range(2, 15, 3):
                    p_lsi = Process(name=f'lsi_{num_topics}', target=get_lsi_results, args=(
                        doc_term_matrix, id2word, revs, f'lsi_{num_topics}', num_topics, output))
                    processes.append(p_lsi)
            else:
                p_lsi = Process(name='lsi_5', target=get_lsi_results, args=(
                    doc_term_matrix, id2word, revs, 'lsi_5', None, output))
                processes.append(p_lsi)
            
            if transformations:
                p_lsi = Process(name='lsi_tfidf', target=get_lsi_results, args=(
                    doc_term_matrix_tfidf, id2word, revs, 'lsi_tfidf', None, output))
                # processes.append(p_lsi)

                p_lsi = Process(name='lsi_logentropy', target=get_lsi_results, args=(
                    doc_term_matrix_logentropy, id2word, revs, 'lsi_logentropy', None, output))
                processes.append(p_lsi)

                p_lsi = Process(name='lsi_random_projections', target=get_lsi_results, args=(
                    doc_term_matrix_random_projections, id2word, revs, 'lsi_random_projections', None, output))
                processes.append(p_lsi)
        if lda:
            if find_optimal_num_topics:
                for num_topics in range(2, 15, 3):
                    p_lda = Process(name=f'lda_{num_topics}', target=get_lda_results, args=(
                        doc_term_matrix, id2word, revs, f'lda_{num_topics}', num_topics, output))
                    processes.append(p_lda)
            else:
                p_lda = Process(name='lda_5', target=get_lda_results, args=(
                            doc_term_matrix, id2word, revs, 'lda_5', None, output))
                processes.append(p_lda)
            
            if transformations:
                p_lda = Process(name='lda_tfidf', target=get_lda_results, args=(
                    doc_term_matrix_tfidf, id2word, revs, 'lda_tfidf', None, output))
                # processes.append(p_lda) # TODO: BUG - Does not complete, app frozen.

                p_lda = Process(name='lda_logentropy', target=get_lda_results, args=(
                    doc_term_matrix_logentropy, id2word, revs, 'lda_logentropy', None, output))
                # processes.append(p_lda) # TODO: BUG in here. Does not complete, app frozen. 

                p_lda = Process(name='lda_random_projections', target=get_lda_results, args=(
                    doc_term_matrix_random_projections, id2word, revs, 'lda_random_projections', None, output))
                # processes.append(p_lda) # TODO: BUG in here: Message: RuntimeWarning: invalid value encountered in multiply gammad = self.alpha + expElogthetad * np.dot(cts / phinorm, expElogbetad.T)
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
        
        while len(results) < len(processes):
        # logger.info('i m here...')
            for process in processes:                
                process.join(5)
                # logger.info("{} process is joining..., its status is {}".format(process.name, process.is_alive()))
                # if process.is_alive():
                # logger.info(f"{process.name} is getting results...")
                # alive_cnt.append(process.name)
                result = output.get()
                results.append(result)
                logger.info(f"output received for {result[0]}.")
                logger.info(f"results: {len(results)} procs: {len(processes)} => {len(results) < len(processes)}")
                # process.terminate()
                # logger.info(f"{process.name} is terminating.")

        # output.close()
        # output.join_thread()
        logger.info("processeses finished.")

        for process in processes:
            logger.info("process status for {}: {}".format(process.name, process.is_alive()))
            process.kill()
            logger.info(f"{process.name} is terminated.")

    except Exception as ex:
        logger.warning("run_topic_models failed.", exc_info=ex)

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


def run_sentiment_models(revs_list, sentiment_list, to_file, optimum=False, sgd=True, log=True, mnb=True, rdg=False):
    
    logger.info("sentiment algorithms starting..")

    X = revs_list
    y = sentiment_list
    logger.info(f"number of items in reviews and sentiments: revs => {len(X)} sentiments: {len(y)}")

    models = []
    if sgd:
        models.append(('sgd', sentiment_analysis.sgd_model))
    if log:
        models.append(('log', sentiment_analysis.log_model))
    if mnb:
        models.append(('mnb', sentiment_analysis.mnb_model))
    if rdg:
        models.append(('rdg', sentiment_analysis.rdg_model))

    pipes = [
        sentiment_analysis.get_pipe(),
        sentiment_analysis.get_pipe(tfidf=False),
    ]

    model_pipe_list = list(itertools.product(pipes, models))

    sentiment_analysis.params['optimum'] = optimum

    results = []

    for pipe, (model_name, fn_model) in model_pipe_list:
        clf = fn_model(X, y, pipe, model_name)
        logger.info(f"model built and saved for {model_name}.")
        y_pred = sentiment_analysis.get_document_sentiments(clf, X, y)
        results.append(pd.Series(y_pred, name=model_name))
        pipe.steps.pop()
        logger.info(f"predictions calculated for {model_name}. accuracy is {np.mean(y == y_pred)}.")

    # saving into file
    df_sentiments = pd.concat(results, axis=1)
    logger.info(f"sentiments has shape of : {df_sentiments.shape}")
    if to_file is not None:
        df_sentiments.to_csv(f'../data/processed/{to_file}', index=False)
        logger.info("file saved.")
    
    return df_sentiments
    