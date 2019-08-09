#!/usr/bin/python3
"""
Document of app.
"""

# standard imports
import sys
import logging
import argparse
import itertools

# third party imports
import pandas as pd
import numpy as np

# local imports
import config
import utilities
from build_data import get_competitor_reviews, view_current_businesses
from process_data import apply_text_processing
from model_data import run_topic_models, run_sentiment_models

# logging init
logger = logging.getLogger(__name__)
logger.warning("app started")


def train_data(args):
    # read data
    df = pd.read_csv(
        f'../data/processed/{args.file_name}', nrows=None, memory_map=True)
    logger.info(f"file is read. the shape of the file is {df.shape}.")

    # cleanup & fixing
    df = apply_text_processing(
        revs_list=df,
        to_file=f'{args.file_name}_processed.csv',
        read_from_file=True,
        nrows=None
    )
    revs_list = df.norm_tokens_doc
    docs_list = list(itertools.chain(*revs_list))
    logger.info("file text processing is completed.")

    # topic modeling
    run_topic_models(
        revs_list=revs_list,
        docs_list=docs_list,
        to_file=f'{args.file_name}_topics.csv',
        transformations=True,
        find_optimal_num_topics=False,
        training=True,
        lsi=False,
        lda=True,
        mallet=False
    )

    # sentiment modeling
    run_sentiment_models(
        revs_list=revs_list.apply(lambda x: ' '.join(itertools.chain(*x))),
        sentiment_list=df.sentiment,
        to_file=f'{args.file_name}_sentiments.csv',
        optimum=True,
        sgd=True,
        log=True,
        mnb=True,
    )


def download_data(args):
    get_competitor_reviews(
        args.start_index, args.end_index, download_competitors=False)


# init main
if __name__ == '__main__':

    # config
    config.get_config('./config.ini')

    # arguments
    parser = argparse.ArgumentParser(
        description='NLP trainer and new review classifier.')
    subparsers = parser.add_subparsers()

    # train data
    parser_train = subparsers.add_parser(
        'train', help='train app by providing data.')
    parser_train.set_defaults(func=train_data)
    parser_train.add_argument("--file_name",
                                type=int,
                                default=0,
                                help="provide a file name to train a nlp model for review analysis.")

    # view data
    parser_view = subparsers.add_parser(
        'view', help='view current businesses of Yelp.')
    parser_view.set_defaults(func=view_current_businesses)

    # download data
    parser_download = subparsers.add_parser(
        'download', help='download data from Yelp.')
    parser_download.set_defaults(func=download_data)
    parser_download.add_argument("--start_index",
                                 type=int,
                                 default=0,
                                 help="start index of the competitors data.")
    parser_download.add_argument("--end_index",
                                 type=int,
                                 default=1,
                                 help="end index of the competitors data.")

    # default print help
    if len(sys.argv) <= 1:
        sys.argv.append('--help')

    # parse args and supply into functions
    args = parser.parse_args()
    args.func(args)
