# Importa as bibliotecas
import sklearn as skl
import numpy as np
from gensim.models import Word2Vec

# Função que recebe uma lista de documentos de treino e retorna uma representação TF (Term-frequency) desses documentos
def term_frequency_train(document_list):
    # Cria uma cópia da lista de documentos
    new_document_list = document_list.copy()
   
    # Cria uma instância do CountVectorizer do scikit-learn, para "contar" a frequência de cada termo
    vectorizer = skl.feature_extraction.text.CountVectorizer(
        analyzer = "word",
        tokenizer = None, 
        preprocessor = None,
        stop_words = None, 
        lowercase = True,
        binary=False,
        dtype=np.int32
    )

    # Executa a contagem de frequência dos termos
    document_list_tf = vectorizer.fit_transform(new_document_list)

    # Retorna a representação TF e a instância do CountVectorizer (para reutilizarmos a mesma instância)
    return document_list_tf, vectorizer

# Função que recebe uma lista de documentos de teste e uma instância de CountVectorizer para retornar uma representação TF (Term-frequency) desses documentos
def term_frequency_test(document_list, vectorizer):
    # Cria uma cópia da lista de documentos
    new_document_list = document_list.copy()

    # Executa a contagem de frequência dos termos
    document_list_tf = vectorizer.transform(new_document_list)

    # Retorna a representação TF
    return document_list_tf

# Função que recebe a representação TF dos documentos de treino e teste para retornar a representação em binário
def convert_to_binary(train_tf, test_tf):
    # Copia a representação TF
    train_binary = train_tf.copy()
    test_binary = test_tf.copy()

    # Transforma a representação TF para binário
    # Caso a palavra tenha mais de uma aparição, setaremos o valor como 1, caso contrário, setaremos como 0
    train_binary[train_binary>0] = 1
    test_binary[test_binary>0] = 1
    train_binary[train_binary<0] = 0
    test_binary[test_binary<0] = 0

    # Retorna a representação em binário
    return train_binary, test_binary

# Função que recebe a representação TF dos documentos de treino e teste para retornar a representação em TF-IDF (Term Frequency - Inverse Data Frequency)
def tf_idf(train_tf, test_tf):
    # Cria uma instância do transformador de TF-IDF do scikit-learn
    tfidf_model = skl.feature_extraction.text.TfidfTransformer(
    norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=True)

    # Executa a transformação de TF para TF-IDF
    train_tfidf = tfidf_model.fit_transform(train_tf)
    test_tfidf = tfidf_model.transform(test_tf)

    # Retorna a representação TF-IDF
    return train_tfidf, test_tfidf

# Função que recebe a lista de documentos de treino e teste e retorna a lista de palavras 
def generate_embededdings(train_documents, test_documents):
    # Cria a lista de palavras de treino
    train_embedded = []

    # Preenche a lista de palavras de treino
    for msg in train_documents:
        train_embedded.append(msg.split())

    # Cria a lista de palavras de teste
    test_embedded = []

    # Preenche a lista de palavras de teste
    for msg in test_documents:
        test_embedded.append(msg.split())

    # Retorna as listas
    return train_embedded, test_embedded

# Função que recebe a lista de palavras, o tamanho de cada palavra, o tamanho da janela e a contagem minima para as palavras 
# para retornar um modelo Word2Vec
def word2vec(words, vector_size, window, min_count):
    # Cria o modelo utilizando os parâmetros especificados
    embedding_model =  Word2Vec(sentences = words,
                          vector_size = vector_size,
                          window = window,
                          min_count = min_count)
    
    # Retorna o modelo
    return embedding_model
    
# Função para criar o vetor para um documento utilizando média através de um modelo Word2Vec
def get_document_vector_by_mean(embedding_model, document):
    # Cria a lista de vetores 
    word_list = []
    for word in document:
        # Caso a palavra esteja no modelo, adiciona o vetor dela na lista
        try:
            vec = embedding_model.wv[word]
            word_list.append(vec)
        except:
            pass

    # Se a lista de vetores conter algum vetor de palavra, gera a média dessa lista
    # Caso contrário, gera uma lista de zeros do tamanho necessário
    if len(word_list)>0:
        vector_mean = np.mean( word_list, axis=0 )
    else:
        vector_mean = np.zeros( embedding_model.wv.vector_size )

    # Retorna o vetor do documento
    return vector_mean

# Função para criar o vetor para um documento utilizando média através de um modelo Word2Vec (para o modelo pré-treinado)
def get_document_vector_by_mean_pre_trained_model(embedding_model, document):
    # Cria a lista de vetores 
    word_list = []
    for word in document:
        # Caso a palavra esteja no modelo, adiciona o vetor dela na lista
        try:
            vec = embedding_model[word]
            word_list.append(vec)
        except KeyError:
            pass

    # Se a lista de vetores conter algum vetor de palavra, gera a média dessa lista
    # Caso contrário, gera uma lista de zeros do tamanho necessário
    if len(word_list) > 0:
        vector_mean = np.mean(word_list, axis=0)
    else:
        vector_mean = np.zeros(embedding_model.vector_size)

    # Retorna o vetor do documento
    return vector_mean

# Função para criar uma lista com os vetores da lista de documentos indicada
def dataframe_to_matrix_by_mean(document_list, embedding_model):
    # Cria a lista de vetores de documento
    document_vector_list = []
    
    # Preenche a lista com o vetor de cada documento
    for document in document_list:
        vec = get_document_vector_by_mean(embedding_model, document)
        document_vector_list.append(vec)

    # Transforma a lista em um array do numpy
    document_vector_list = np.array(document_vector_list)

    # Retorna a lista com os vetores dos documentos
    return document_vector_list

# Função para criar uma lista com os vetores da lista de documentos indicada (para o modelo Word2Vec pré-treinado)
def dataframe_to_matrix_by_mean_pre_trained_model(document_list, embedding_model):
    # Cria a lista de vetores de documento  
    document_vector_list = []

    # Preenche a lista com o vetor de cada documento
    for document in document_list:
        vec = get_document_vector_by_mean_pre_trained_model(embedding_model, document)
        document_vector_list.append(vec)

    # Transforma a lista em um array do numpy
    document_vector_list = np.array(document_vector_list)

    # Retorna a lista com os vetores dos documentos
    return document_vector_list

# Função para criar o vetor para um documento sem utilizar média
def get_document_vector_with_zeros_vector(embedding_model, document, max_length):
    # Seleciona a primeira palavra do modelo Word2Vec
    first_word = embedding_model.wv.index_to_key[0]

    # Armazena a dimensão da embedding da primeira palavra
    embedding_dimension = embedding_model.wv[first_word].shape[0]

    # Cria a lista de vetores das palavras
    word_vectors = []

    # Itera sobre cada palavra, respeitando o limite de palavras indicado
    for i in range(max_length):
        # Cria um vetor com a dimensão da embedding, contendo apenas zeros
        zeros_vector = np.zeros(embedding_dimension)

        # Adiciona o vetor de zeros na lista de vetores de palavras
        word_vectors.append(zeros_vector)

        # Para cada palavra do documento, busca o vetor da mesma
        # Caso a palavra não exista no modelo Word2Vec, mantém o vetor da palavra como zeros
        if i < len(document):
            try:
                word_vectors[i] = embedding_model.wv[document[i]]
            except:
                pass

    # Retorna o vetor do documento
    return word_vectors

# Função para criar o vetor para um documento sem utilizar média (para modelo Word2Vec pré-treinado)
def get_document_vector_with_zeros_vector_pre_trained_model(embedding_model, document, max_length):
    # Seleciona a primeira palavra do modelo Word2Vec
    first_word = embedding_model.index_to_key[0]

    # Armazena a dimensão da embedding da primeira palavra
    embedding_dimension = len(embedding_model[first_word])

    # Cria a lista de vetores das palavras
    word_vectors = []

    # Itera sobre cada palavra, respeitando o limite de palavras indicado
    for i in range(max_length):
        # Cria um vetor com a dimensão da embedding, contendo apenas zeros
        zeros_vector = np.zeros(embedding_dimension)

        # Adiciona o vetor de zeros na lista de vetores de palavras
        word_vectors.append(zeros_vector)

        # Para cada palavra do documento, busca o vetor da mesma
        # Caso a palavra não exista no modelo Word2Vec, mantém o vetor da palavra como zeros
        if i<len(document):
            try:
                word_vectors[i] = embedding_model[document[i]]
            except:
                pass

    # Retorna o vetor do documento
    return word_vectors

# Função para criar uma lista com os vetores da lista de documentos indicada, sem utilizar média
def dataframe_to_matrix_by_zeros(document_list, embedding_model):
    # Cria a lista de vetores de documento  
    document_vector_list = []
    
    # Preenche a lista com o vetor de cada documento
    for document in document_list:
        vec = get_document_vector_with_zeros_vector(embedding_model, document)
        document_vector_list.append(vec)

    # Transforma a lista em um array do numpy
    document_vector_list = np.array(document_vector_list)

    # Retorna a lista com os vetores dos documentos
    return document_vector_list

# Função para criar uma lista com os vetores da lista de documentos indicada, sem utilizar média
def dataframe_to_matrix_by_zeros_pre_trained_model(document_list, embedding_model):
    # Cria a lista de vetores de documento  
    document_vector_list = []
    
    # Preenche a lista com o vetor de cada documento
    for document in document_list:
        vec = get_document_vector_with_zeros_vector_pre_trained_model(embedding_model, document)
        document_vector_list.append(vec)

    # Transforma a lista em um array do numpy
    document_vector_list = np.array(document_vector_list)

    # Retorna a lista com os vetores dos documentos
    return document_vector_list