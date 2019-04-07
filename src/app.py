# standard imports
import logging
import argparse
import itertools

# third party imports
import pandas as pd
import numpy as np

# local imports
import utilities
from process_data import apply_text_processing
from model_data import run_topic_models, run_sentiment_models

# logging init
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)-25s - %(levelname)-8s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
    handlers=[
        logging.FileHandler('../data/logs/logs_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("app")
logger.info("app started")

# init main
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--starti",
        type=int,
        default=0,
        help="start index of the competitors data."
    )
    parser.add_argument(
        "--endi",
        type=int,
        default=1,
        help="end index of the competitors data."
    )

    flags, unparsed = parser.parse_known_args()

    if input("Press any key to start..") is not None:
        # get_competitor_reviews(flags.starti, flags.endi)
        
        # read data
        df = pd.read_csv('../data/processed/hi_rws_0001_0256_complete.csv', nrows=1500, memory_map=True)
        logger.info("file is read.")

        # cleanup & fixing
        df['norm_tokens_doc'] = df.description.apply(lambda x: apply_text_processing(x))
        logger.info("file cleaning is completed.")
        # df = utilities.fix_token_columns(df) # if list is dumped to a file
        # logger.info("file fix is completed.")
        
        revs_list = df.norm_tokens_doc
        docs_list = list(itertools.chain(*revs_list))

        run_topic_models(
            revs_list=revs_list,
            docs_list=docs_list, 
            to_file='hi_rws_0001_0256_topics.csv', 
            transformations=True, 
            find_optimal_num_topics=True, 
            training=True,
            lsi=True,
            lda=True,            
            mallet=False,
            hdp=False
        )

        # run_sentiment_models(
        #     revs_list=revs_list.apply(lambda x: ' '.join(itertools.chain(*x))),
        #     sentiment_list=df.sentiment,
        #     to_file='hi_rws_0001_0256_sentiments.csv',
        #     optimum=True,
        #     sgd=True,
        #     log=True,
        #     mnb=True,
        #     rdg=False # does not have predict_proba
        # )