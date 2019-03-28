import itertools
import logging

import pandas as pd
import numpy as np

import utilities
from models import topic_analysis

logger = logging.getLogger()

def test_topic_models(filepath):
    # read data
    df = pd.read_csv(filepath)
    logger.info("file read")
    df = utilities.fix_token_columns(df)
    logger.info("file fix completed.")
    
    # sentences
    revs = df.norm_tokens_doc[:100]
    docs = list(itertools.chain(*revs))
    
    # topic modeling
    doc_term_matrix, id2word = topic_analysis.create_doc_term_matrix(docs)
    
    lda_model = topic_analysis.get_lda_model(doc_term_matrix, id2word, '../model/lda_test.model')
    doc_topic_tuples = topic_analysis.get_document_topics(lda_model, doc_term_matrix, revs)

    lda_mallet = topic_analysis.get_lda_mallet_model(doc_term_matrix, id2word, '../model/lda_test.model')
    doc_topic_tuples = topic_analysis.get_document_topics(lda_mallet, doc_term_matrix, revs)

    doc_topic_tuples
    pd.DataFrame(doc_topic_tuples, columns=['topic_mode', 'topic_mode_prob', 'topic_list'])