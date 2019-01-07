# standard imports
import os
import sys
import logging

logging.getLogger(name="review-analytics")

## relative import hack which I don't like :/
logging.info(f'module name: {__name__},path: {os.getcwd()}')
os.chdir(sys.path[0])
logging.info(f'module name: {__name__},path: {os.getcwd()}')

# third party imports
import click
import pandas as pd
import numpy as np

# local imports
from data.data_builder import create_dataset
from data.data_scrapper import yelp_branches, scrappers
from features.text_preprocessing import preprocessing, nlp_preprocessing
from features.feature_extraction import model

def textprocessing(test_reviews):
    p = preprocessing(test_reviews)

    reviews1 = []
    for review in test_reviews:
        # lower
        review = p.lower(review)
        # punctuation
        review = p.punctuation(review)  # todo: bug
        # stopwords
        review = p.stopwords(review)
        # freq
        review = p.freqwords(review)
        # shortwords
        review = p.shortwords(review)
        # rare
        review = p.rarewords(review)
        # spelling
        # review = p.spelling(review)
        # tokenize
        review = p.tokenize(review)
        # stemming
        review = p.stemming(review)
        # lemmatization
        review = p.lemnatize(review)
        reviews1.append(review)
    
    return(reviews1)

@click.group()
def cli():
    pass

@click.command()
def view_current_businesses():
    message = '\
    Current yelp branches that is\n \
    followed is listed in below.\n\n' + "\n".join(yelp_branches)
    return print(message)

@click.command()
@click.option('--filepath', help='prepare yelp json data for analysis.')
def convert_to_csv(filepath):
    print(filepath)
    print(os.getcwd())
    create_dataset(filepath, '././data/processed/dataset1.csv')


    # text cleaning

@click.command()
def training():
    dataset = pd.read_csv('././data/processed/dataset1.csv')
    reviews = dataset.iloc[:, 1]

    reviews1 = textrocessing(reviews)

    # nlp ops
    nlp = nlp_preprocessing(reviews1)
    tokenized_reviews = pd.Series(reviews1).apply(lambda x: x.split())
    reviews2 = nlp.lemmatization(tokenized_reviews)

    dataset['reviews'] = pd.Series(reviews2).apply(lambda x: ' '.join(x))
    dataset.to_csv('././data/processed/dataset2.csv', index=False)


@click.command()
@click.option('--filepath', help='evalutate file based on previous model.')
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
    
    train_reviews1 = pd.Series(train_reviews.tolist()).apply(lambda x: x.split()).tolist()

    # model
    m = model()
    lda_model = m.model_load()
    topic_probabilities = m.predict(lda_model, train_reviews1, reviews2)

    [print(i) for i in topic_probabilities]

cli.add_command(convert_to_csv)
cli.add_command(training)
cli.add_command(topic_from_file)
cli.add_command(view_current_businesses)

if __name__ == '__main__':
    cli()
