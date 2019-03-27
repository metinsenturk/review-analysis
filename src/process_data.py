from features.text_preprocessing import preprocessing

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