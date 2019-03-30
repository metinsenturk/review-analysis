import itertools
import logging
import time
from multiprocessing import Process, Queue

import pandas as pd
import numpy as np

import utilities
from models import topic_analysis

logger = logging.getLogger(__name__)
logging.getLogger("gensim").setLevel(logging.WARNING)


def get_lsi_results(doc_term_matrix, id2word, revs, fname, output):
    logger.info(f"{fname} started")
    lsi_model = topic_analysis.get_lsi_model(
        doc_term_matrix, id2word, f'../model/lsi_model/{fname}.model')
    doc_topic_tuples = topic_analysis.get_document_topics(
        lsi_model, doc_term_matrix, revs)

    time.sleep(5)
    output.put(('lsi', doc_topic_tuples))

    return doc_topic_tuples


def get_lda_results(doc_term_matrix, id2word, revs, fname, output):
    logger.info(f"{fname} started")
    lda_model = topic_analysis.get_lda_model(
        doc_term_matrix, id2word, f'../model/lda_model/{fname}.model')
    doc_topic_tuples = topic_analysis.get_document_topics(
        lda_model, doc_term_matrix, revs)

    time.sleep(10)
    output.put(('lda', doc_topic_tuples))

    return doc_topic_tuples


def get_mallet_results(doc_term_matrix, id2word, revs, fname, output):
    logger.info(f"{fname} started")
    mallet_model = topic_analysis.get_lda_mallet_model(
        doc_term_matrix, id2word, f'../model/mallet_model/{fname}.model')
    doc_topic_tuples = topic_analysis.get_document_topics(
        mallet_model, doc_term_matrix, revs)

    output.put(('mallet', doc_topic_tuples))


def test_topic_models(from_filepath, to_filepath, lsi=True, lda=True, mallet=True):
    # read data
    df = pd.read_csv(from_filepath)
    logger.info("file read")
    df = utilities.fix_token_columns(df.iloc[:95, :])
    logger.info("file fix completed.")

    # sentences
    revs = df.norm_tokens_doc[:95]
    docs = list(itertools.chain(*revs))
    logger.info(f"number of reviews and documents => revs: {len(revs)} docs: {len(docs)}")

    # topic modeling
    doc_term_matrix, id2word = topic_analysis.create_doc_term_matrix(docs)
    
    # topic models in multiprocessing
    output = Queue()
    processes = []
    results = []

    #results.append(('lsi', get_lsi_results(doc_term_matrix, id2word, revs, 'lsi', output)))
    #results.append(('lda', get_lda_results(doc_term_matrix, id2word, revs, 'lda', output)))

    if lsi:
        p_lsi = Process(name='lsi', target=get_lsi_results, args=(
            doc_term_matrix, id2word, revs, 'lsi', output))
        processes.append(p_lsi)
    if lda:    
        p_lda = Process(name='lda', target=get_lda_results, args=(
            doc_term_matrix, id2word, revs, 'lda', output))
        processes.append(p_lda)
    if mallet:    
        p_mallet = Process(name='mallet', target=get_mallet_results, args=(
            doc_term_matrix, id2word, revs, 'mallet', output))
        processes.append(p_mallet)

    for process in processes:
        logger.info("{} process is started.".format(process.name))
        process.start()

    for process in processes:
        process.join()
        logger.info("{} process is completed".format(process.name))
        results.append(output.get())
        logger.info("output is appended to the list.")

    for process in processes:
        logger.info("process status for {}: {}".format(process.name, process.is_alive()))

    # saving results
    logger.info("saving results in file")
    pd_topics=[]
    for name, tuples_list in results:
        column_names=['topic_mode', 'topic_mode_prob', 'topic_list']
        pd_topic=pd.DataFrame(tuples_list, columns=[
                              name + '_' + col for col in column_names])
        logger.info(f"topics for {name} has shape of : {pd_topic.shape}")
        pd_topics.append(pd_topic)

    df_topics=pd.concat(pd_topics, axis=1)
    logger.info(f"topics has shape of : {df_topics.shape}")
    df_topics.to_csv(to_filepath, index=False)
    logger.info("file saved.")
