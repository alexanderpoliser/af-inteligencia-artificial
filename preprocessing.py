import nltk
import unicodedata # sera usada para remover acentos dos documentos em lingua portuguesa
import re

# Download the stopwords corpus
nltk.download('stopwords')

# Download the RSLPStemmer
nltk.download('rslp')

from nltk.stem import RSLPStemmer # para fazer a estemização em documentos da lingua portuguesa

def preprocess_text(text):
    # Lower case
    text = text.lower()

    # remove os acentos das palavras
    nfkd_form = unicodedata.normalize('NFKD', text)
    text = u"".join([c for c in nfkd_form if not unicodedata.combining(c)])

    # remove tags HTML
    regex = re.compile('<[^<>]+>')
    text = re.sub(regex, " ", text)

    # normaliza as URLs
    regex = re.compile('(http|https)://[^\s]*')
    text = re.sub(regex, "<URL>", text)

    # normaliza emails
    regex = re.compile('[^\s]+@[^\s]+')
    text = re.sub(regex, "<EMAIL>", text)

    # converte todos os caracteres não-alfanuméricos em espaço
    regex = re.compile('[^A-Za-z0-9]+')
    text = re.sub(regex, " ", text)

    # normaliza os numeros com ponto
    regex = re.compile('[0-9]+.[0-9]+')
    text = re.sub(regex, "NUMERO", text)

    # normaliza os numeros com virgula
    regex = re.compile('[0-9]+,[0-9]+')
    text = re.sub(regex, "NUMERO", text)

    # normaliza os numeros
    regex = re.compile('[0-9]+')
    text = re.sub(regex, "NUMERO", text)

    # substitui varios espaçamentos seguidos em um só
    text = ' '.join(text.split())

    # separa o texto em palavras
    words = text.split()

    # trunca o texto para apenas 200 termos
    # words = words[0:200]

    # remove stopwords
    words = text.split() # separa o texto em palavras
    words = [w for w in words if not w in nltk.corpus.stopwords.words('portuguese')]
    text = " ".join( words )

    # aplica estemização
    stemmer_method = RSLPStemmer()
    words = [ stemmer_method.stem(w) for w in words ]
    text = " ".join( words )

    # remove palavras compostas por apenas um caracter
    words = text.split() # separa o texto em palavras
    words = [ w for w in words if len(w)>1 ]
    text = " ".join( words )

    return text

def preprocess(df_train, df_test):
    for i, row in df_train.iterrows():
        df_train.at[i, "Text"] = preprocess_text(row.Text)


    for i, row in df_test.iterrows():
        df_test.at[i, "Text"] = preprocess_text(row.Text)

    return df_train, df_test