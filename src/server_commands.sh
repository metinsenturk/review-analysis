#!/bin/sh

# get model directory -- method 1
# scp -r msenturk@dsl.saintpeters.edu:/home/msenturk/review-analysis/model/ ../model_server/
rsync -auv --exclude mallet/ msenturk@dsl.saintpeters.edu:/home/msenturk/review-analysis/model/ ../model-server/

rsync -av msenturk@dsl.saintpeters.edu:/home/msenturk/review-analysis/data/ ../data-server/

# copy final models into main model directory for analysis
rsync -r ../model-final/ ../model/
rsync -r ../data-final/ ../data/

# get topic results file
scp msenturk@dsl.saintpeters.edu:/home/msenturk/review-analysis/data/processed/hi_rws_0001_0256_topics.csv ../data-server/processed/hi_rws_0001_0256_topics.csv
scp msenturk@dsl.saintpeters.edu:/home/msenturk/review-analysis/data/processed/hi_rws_0001_0256_processed.csv ../data-server/processed/hi_rws_0001_0256_processed.csv
scp msenturk@dsl.saintpeters.edu:/home/msenturk/review-analysis/data/processed/out_of_vocabulary_words.csv ../data-server/processed/out_of_vocabulary_words.csv

# get sentiments results file
scp msenturk@dsl.saintpeters.edu:/home/msenturk/review-analysis/data/processed/hi_rws_0001_0256_sentiments_backup.csv ../data/processed/hi_rws_0001_0256_sentiments_backup.csv


# get logs file
scp msenturk@dsl.saintpeters.edu:/home/msenturk/review-analysis/data/logs/logs_app.log ../data/logs/logs_app.log