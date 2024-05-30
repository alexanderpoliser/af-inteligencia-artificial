import sklearn as skl
from sklearn import feature_extraction
import numpy as np
import gensim
from gensim.models import Word2Vec

def term_frequency_train(df_train):
    df_train_tf = df_train.copy()
   
    vectorizer = skl.feature_extraction.text.CountVectorizer(
    analyzer = "word", tokenizer = None, preprocessor = None,
    stop_words = None, lowercase = True, binary=False,
    dtype=np.int32)

    vectorizer.fit_transform(df_train_tf)

    return df_train_tf, vectorizer

def term_frequency_test(df_test, vectorizer):
     df_test_tf = df_test.copy()

     vectorizer.transform(df_test_tf)

     return df_test_tf

def convert_to_binarie(df_train_tf, df_test_tf):
    df_train_bin = df_train_tf.copy()
    df_test_bin = df_test_tf.copy()

    df_train_bin[df_train_bin>0] = 1
    df_test_bin[df_test_bin>0] = 1

    return df_train_bin, df_test_bin

def tf_idf(df_train_tf, df_test_tf):
    tfidf_model = skl.feature_extraction.text.TfidfTransformer(
    norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=True)

    df_train_tfidf = tfidf_model.fit_transform(df_train_tf)
    df_test_tfidf = tfidf_model.transform(df_test_tf)

    return df_train_tfidf, df_test_tfidf

def generate_embededdings(df_train, df_test):
    df_train_embedded = []
    for i, msg in enumerate(df_train):
        df_train_embedded.append(msg.split())

    df_test_embedded = []
    for i, msg in enumerate(df_test):
        df_test_embedded.append(msg.split())

    return df_train_embedded, df_test_embedded

def word2vec(df_train_embedded, vector_size, window, min_count):
    embedding_model =  Word2Vec(sentences = df_train_embedded,
                          vector_size = vector_size,
                          window = window,
                          min_count = min_count)
    return embedding_model
    
def get_document_vector_by_mean(embedding_model, document):

    word_list = []
    for word in document:
        try:
            vec = embedding_model.wv[word]
            word_list.append(vec)
        except:
            pass

    if len(word_list)>0:
        vector_mean = np.mean( word_list, axis=0 )
    else:
        vector_mean = np.zeros( embedding_model.wv.vector_size )

    return vector_mean

def get_document_vector_with_zeros_vector(embedding_model, document, max_length):

    first_word = embedding_model.wv.index_to_key[0]
    dimEmbedding = embedding_model.wv[first_word].shape[0]

    word_vectors = []

    for i in range(max_length):

      zeros_vector = np.zeros(dimEmbedding)
      word_vectors.append(zeros_vector)

      if i<len(document):
          try:
            word_vectors[i] = embedding_model.wv[document[i]]
          except:
            pass
    word_vectors = np.array(word_vectors)

    return word_vectors

def dataframe_to_matrix_by_mean(dataframe, embedding_model):
    dataframe_embedding = []
    
    for document in dataframe:
        vec = get_document_vector_by_mean(embedding_model, document)
        dataframe_embedding.append(vec)

    dataframe_embedding = np.array(dataframe_embedding)

    return dataframe_embedding