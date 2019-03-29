# standard imports
import logging
import argparse

# third party imports
import pandas as pd
import numpy as np

# local imports
from model_data import test_topic_models

# logging init
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)-8s - %(name)-12s - %(message)s',
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
        test_topic_models(
            from_filepath='../data/processed/yp_competitors_rws_0001_0050_complete.csv', 
            to_filepath='../data/processed/yp_competitors_rws_0001_0050_topics.csv')