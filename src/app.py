# standard imports
import os
import sys
import logging
import argparse

# relative import hack which I don't like :/
# os.chdir(sys.path[0])

# third party imports
import pandas as pd
import numpy as np

# local imports
from data import folder_paths as fp
from data.data_builder import create_dataset
from data.data_scrapper import yelp_branches, scrappers
from features.text_preprocessing import preprocessing, nlp_preprocessing
from features.feature_extraction import model

# logging init
logging.basicConfig(
    filename='logs_app.log',
    level=logging.INFO, 
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)
logger.info("app started")

def get_competitor_reviews(start_index, end_index, dw_competitors=False):
    sc = scrappers()

    if dw_competitors:
        sc.yp_get_competitors(yelp_branches)
    
    try:
        sc.yp_get_competitor_reviews(start_index=start_index, end_index=end_index)
    except Exception as ex:
        logger.warning("error: " + ex)
    print("helo")

def view_current_businesses():
    message = ' Current yelp branches that is \n' \
              ' followed is listed in below.\n\n' + "\n".join(yelp_branches)
    return print(message)

def convert_to_csv():
    for branch_name in yelp_branches:
        from_path = fp.yp_raw_folder_path(branch_name)
        to_path = fp.yp_processed_folder_path(branch_name)

        create_dataset(from_path, to_path)

def textprocessing2(dataframe):
    df = pd.read_csv('././data/processed/dataset1.csv')
    df = df.reviews.apply(str.lower)
    return(df.head())

def textprocessing(test_reviews):
    p = preprocessing(test_reviews)

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

def training():
    dataset = pd.read_csv('././data/processed/dataset1.csv')
    reviews = dataset.iloc[:, 1]

    reviews1 = textprocessing(reviews)

    # nlp ops
    nlp = nlp_preprocessing(reviews1)
    tokenized_reviews = pd.Series(reviews1).apply(lambda x: x.split())
    reviews2 = nlp.lemmatization(tokenized_reviews)

    dataset['reviews'] = pd.Series(reviews2).apply(lambda x: ' '.join(x))
    dataset.to_csv('././data/processed/dataset2.csv', index=False)


def topic_from_file(filepath):
    dataset1 = pd.read_csv('././data/processed/dataset2.csv')
    train_reviews = dataset1.iloc[:, 1]

    dataset2 = pd.read_csv(filepath)
    test_reviews = dataset2.iloc[:, 1]

    # text processing
    reviews1 = textprocessing(test_reviews)

    # nlp ops
    nlp = nlp_preprocessing(reviews1)
    tokenized_reviews = pd.Series(reviews1).apply(lambda x: x.split())
    reviews2 = nlp.lemmatization(tokenized_reviews)

    train_reviews1 = pd.Series(train_reviews.tolist()).apply(
        lambda x: x.split()).tolist()

    # model
    m = model()
    lda_model = m.model_load()
    topic_probabilities = m.predict(lda_model, train_reviews1, reviews2)

    [print(i) for i in topic_probabilities]


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
        get_competitor_reviews(flags.starti, flags.endi)