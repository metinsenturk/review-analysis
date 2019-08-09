import logging
import pandas as pd
from . import utilities
from .features.text_preprocessing import SpaCyProcessing, NLTKProcessing

#logger
logger = logging.getLogger(__name__)
# init library
cleanup = SpaCyProcessing()


def textprocessing(test_reviews):
    p = NLTKProcessing(test_reviews)

    reviews1 = []
    for review in test_reviews:
        review = p.lower(review)
        review = p.punctuation(review)  # todo: bug
        review = p.stopwords(review)
        review = p.freqwords(review)
        review = p.shortwords(review)
        review = p.rarewords(review)
        # review = p.spelling(review)
        review = p.tokenize(review)
        review = p.stemming(review)
        review = p.lemnatize(review)
        reviews1.append(review)

    return(reviews1)


def apply_text_processing(revs_list, to_file=None, read_from_file=False, nrows=None, remove_stopwords=True, remove_alpha=True, remove_punct=True, remove_pos=True, lemmatize=True, remove_short_words=True):
    
    df_series = revs_list.description

    # read or create file after preprocessing
    if read_from_file is False:
        df_dump = df_series.apply(lambda x: [tuple(i) for i in cleanup.doc_sent_clean_up(x)])
        if to_file is not None:
            df_dump.to_csv(f'../data/processed/{to_file}', index=False)
    else:
        df_dump = pd.read_csv(f'../data/processed/{to_file}', nrows=nrows, header=None, index_col=False, names=['norm_tokens_doc'])
        df_dump = df_dump.iloc[:, 0]
        df_dump = utilities.fix_token_columns2(df_dump)
        logger.info("file fix is completed.")

    # some reviews are empty sents, remove them
    df_to_be_dropped = df_dump[df_dump.apply(lambda x: len(x) == 0)]
    df_dump = df_dump.drop(df_to_be_dropped.index, axis=0)
    logger.info(f"file empty rows removed. the shape of the file is {df_dump.shape}.")

    # temp: remove all bad indexes from df.
    revs_list = revs_list.drop(df_to_be_dropped.index, axis=0)
    revs_list['norm_tokens_doc'] = df_dump
    logger.info(f"return dataframe constructed. the shape is {revs_list.shape}.")

    return revs_list
