import gensim
from gensim import corpora
from gensim.test.utils import common_texts
from gensim.corpora import Dictionary
from gensim.corpora.dictionary import Dictionary
from gensim.test.utils import datapath

class model:
    def __init__(self): 
        super()
    
    def model_save(self, lda_model):
        temp_file = datapath("model")
        lda_model.save(temp_file)

    def model_load(self):
        temp_file = datapath("model")
        LDA = gensim.models.ldamodel.LdaModel
        lda_model = LDA.load(temp_file)

        return lda_model

    def model(self, texts_list):
        dictionary = corpora.Dictionary(texts_list)
        doc_term_matrix = [dictionary.doc2bow(rev) for rev in texts_list] 
        LDA = gensim.models.ldamodel.LdaModel
        lda_model = LDA(
            corpus=doc_term_matrix, 
            id2word=dictionary, 
            num_topics=4, 
            random_state=0,
            chunksize=1000, 
            passes=50
        )
        return lda_model

    def predict(self, lda_model, train_list, test_list):
        dictionary = corpora.Dictionary(train_list)

        corpus_list = [dictionary.doc2bow(rev) for rev in test_list]        
        doc_lda_list = [lda_model[corpus] for corpus in corpus_list]

        return(doc_lda_list)