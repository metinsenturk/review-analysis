#!/bin/sh

# get model directory -- method 1
# scp -r msenturk@dsl.saintpeters.edu:/home/msenturk/review-analysis/model/ ../model_server/
rsync -av --exclude mallet/ msenturk@dsl.saintpeters.edu:/home/msenturk/review-analysis/model/ ../model_server/

# get topic results file
scp msenturk@dsl.saintpeters.edu:/home/msenturk/review-analysis/data/processed/hi_rws_0001_0256_topics.csv ../data/processed/hi_rws_0001_0256_topics_server.csv

scp msenturk@dsl.saintpeters.edu:/home/msenturk/review-analysis/data/processed/hi_rws_0001_0256_topics.csv ../data/processed/hi_rws_0001_0256_topics_server.cs

# get sentiments results file
scp msenturk@dsl.saintpeters.edu:/home/msenturk/review-analysis/data/processed/hi_rws_0001_0256_sentiments_backup.csv ../data/processed/hi_rws_0001_0256_sentiments_backup.csv


# get logs file
scp msenturk@dsl.saintpeters.edu:/home/msenturk/review-analysis/data/logs/logs_app.log ../data/logs/logs_app.log