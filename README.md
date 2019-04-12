:ok_hand: # Review Analysis Exploration :heart_eyes::fire:

Reviews are important aspect of a business. Customers leave a review to comment and share their point of view of the business in terms of their observiations on business's channels such as staffing, quality of service, ambience, customer service, and so on.

In this project, we are observing the business in terms of reviews. We use natural language processing techniques (NLP) to analyze reviews in order to extract topics and the sentiment of the topics. This release focuses on evaluation of multiple algorithms for both sentiment and topic algorithms. Results of algorithms gathered in files for analysis of best algorithm for each topic modeling and sentiment modeling. Following is the flow of the process.

## Process Flow

### Data Preprocessing

Data processing is essential part of the further analysis. In this project, the processing is implemented by using two main libraries with similar approaches. SpaCy and NLTK's sentnce classifiers are used along with other features.

For NLTK;

- Word and sentences are tokenized.
- Stopwords, punctuations removed.
- Words are first lemmatized, then stemming applied.

For SpaCy, SpaCy's english mult-task CNN trained model is used. More information on the model is available at [here.](https://spacy.io/models/en#en_core_web_lg)

- Word, and sentences are tokenized.
- Stopwords, punctuations, out of vocabulary words, words shorter than 3 characters, and words that are in the list of part-of-speech (POS) tags are removed.
- Lemmatized words are extracted.

### Modeling

We trained a topic and a sentiment classifier by using most popular algorithms. For topic modeling, sentences are considered to be the documents. Therefore, each review is divided into seperate sentences during training process, and after training they reconstructed into the reviews back. This way, we aimed to capture multiple topics that a reviewer might be covering in the review. 

The following algorithms are used for topic modeling.

- Latent Semantic Indexing (LSI)
- Latent Dirichlet Allocation (LDA)
- MALLET
- Hierarchical Dirichlet Process (HDP)

For sentiment modeling, we followed a supervised approach since we used the review as the document and the reviewers' star rating as the ground truth. Following algorithms has been compared and GridSearchCV is used in order to find best models.

- Naive Bayes Classifier for Multinomial Models
- Support Vector Machine (SVM)
- Logistic Regression
- Ridge Classifier

## Why Yelp?

Current version of the application considers only Yelp as the resource for reviews. Yelp is a good resource for opinion mining because of [Yelp's Official Review Policy](https://www.yelp-support.com/article/Don-t-Ask-for-Reviews) since they don't ask for people to write reviews on the places they visit. This is very important because the reviewers are sharing their ideas without a notification from Yelp, which means that the review can be geniune.

## How it Works?

### Download the dataset
To replicate the analysis, one must first download the reviews for a desired business in Yelp. To download the dataset, ```build_data.py``` can be used. I should warn you that downloading the dataset may not be right and you are at your own risk to download data from Yelp. However if you do, the way it works is in the following code.

```python
from build_data import yelp_branches, get_competitor_reviews

# alias of the business that is interested
yelp_branches = ['yelp_business_alias1','yelp_business_alias2']

# this will download the data to .data/raw/file_name.json
get_competitor_reviews(0, len(yelp_branches))
```
### Preprocessing the dataset

Processing part follows two steps. First, the json file has to be converted into csv file. You can do that using the following.

```python
from data.data_builder import create_dataset

create_dataset('json_file_path', 'csv_file_path')
```

Second, using ```process_data.py```, you can use either NLTK or SpaCy to process the data. I recommend SpaCy since it considers POS tag removal in this version. POS tags' can be removed by using NLTK but that hasn't been implemented yet with this version.

### Modeling

```model_data.py``` is the file to run all algorithms for sentiment and topic modeling. Both algorithms have options to run multiple algorithms all at once. Feel free to choose the algorithms desired and run. Topic modeling function implements multiprocessing since training of LDA and scoring of documents requires a lot of time. It runs all models at the same time and collects the results all in one file. I don't recommend running MALLET, since it takes much longer time to complete than others, if your dataset is large (over 100K documents).

```python
from model_data import run_topic_models, run_sentiment_models

# topic modeling
run_topic_models(
    revs_list=revs_list,
    docs_list=docs_list, 
    to_file='file_name_topics.csv', 
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
    to_file='file_name_sentiments.csv',
    optimum=True,
    sgd=True,
    log=True,
    mnb=True,
)
```

### Model Evaluation and Insights

This version requires manual analysis of algorithms using Jupyter Notebooks. Under the notebooks/ directory, 1.3-multi-topic-results.ipynb and 1.0-archive-models.ipynb notebooks can be used for evaluation of models using topic-document frequency distribution, T-SNE plots, topic coherence and model log perplexity scores.

For sentiment analysis, since the ground truths is obtained from star ratings, *sklearn*'s classification_report can be used by examining F1 Score, Recall and Precision. Notebook 1.4-sentiment-results.ipynb can be used for this purpose.

To view insights about the businesses and reviews, use 1.0-archive-models.ipynb to create a final file in which you will merge the selected topic and sentiment model's scores. Then, you can use tableau file under the tableau/ directory to view the complete results.

## Future Implementations and TODO

### Future Implementations

The goal of this project was to explore and examine available models for topic and sentiment algorithms. In addition to existing algorithms, doc2vec model can be used for feature extraction from documents to create word embeddings. After that, sklearn's clustering algorithms like Birch, Kmeans, etc. can be used to find topics in the dataset. 

For sentiment modeling, instead of classification of reviews, a clustering approach for sentences can be considered using various machine learning algorihtms. CNN can be trained as well as clustering algorithms that is mentioned for topic modeling. 

### Todo Lists

### 0.0.9 Todo List

- [x] Gathering in ```app.py``` from strach to final dataset.
- [x] Improvements on Gensim dictionary and LDA model parameters
- [x] Fixes on SpaCy processing.

### 0.0.7 Todo List

- [x] SpaCy processing with POS tagging
- [x] Model evaluation notebooks for sentiment and topic modeling.
- [x] Visualizing with tableau on final dataset.

### 0.0.5 Todo List

- [x] Apply multiple sentiment modeling algorithms
- [x] Use CV and GridSearch algorithm to improve sentiment modeling.
- [x] Multiprocessing in topic modeling.

### 0.0.5 Todo List

- [x] Apply multiple topic modeling algorithms
- [x] Apply sentiment models

### 0.0.1 Todo List

- [x] Determine a group of restaurants for analysis
- [x] Applying topic modeling to the group

## Releases

- [x] 0.0.9 Exploration of various algorihtms
- [ ] 0.1.0 Python application to download, analyze topic and find sentiment for reviews for a given yelp alias.

## Bugs

- [ ] HDP for topic modeling throws convergence warning.
- [ ] Ridge regression for sentiment does not finish.

## Licence

This project is licenced open source with MIT. Use it at your own will, I had my fun.