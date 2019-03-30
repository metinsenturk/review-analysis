# standard imports
import logging
import argparse

# third party imports
import pandas as pd
import numpy as np

# local imports
import utilities
from model_data import run_topic_models

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
        df = pd.read_csv('../data/processed/yp_competitors_rws_0001_0050_complete.csv')
        logger.info("file read.")
        df = utilities.fix_token_columns(df.copy().loc[:1000,:])
        logger.info("file fix completed.")
        run_topic_models(
            tokens_list=df.norm_tokens_doc[:50], 
            to_file='yp_competitors_rws_0001_0050_topics.csv', transformations=True, find_optimal_num_topics=True, mallet=False)