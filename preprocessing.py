# Importa as bibliotecas
import nltk
import unicodedata
import re

# Realiza o download das stopwords
nltk.download('stopwords')

# Realiza o download do RSLPStemmer (para estemização)
nltk.download('rslp')

# Importa o RSLPStemmer que foi baixado
from nltk.stem import RSLPStemmer

# Função para realizar o pré-processamento de um texto
def preprocess_text(text):
    # Transforma o texto em caixa baixa (lowercase)
    text = text.lower()

    # Remove os acentos das palavras utilizando o unicodedata
    nfkd_form = unicodedata.normalize('NFKD', text)
    text = u"".join([c for c in nfkd_form if not unicodedata.combining(c)])

    # Remove possíveis tags HTML
    regex = re.compile('<[^<>]+>')
    text = re.sub(regex, " ", text)

    # Substitui as URLs por <URL>
    regex = re.compile('(http|https)://[^\s]*')
    text = re.sub(regex, "<URL>", text)

    # Substitui os e-mails por <EMAIL>
    regex = re.compile('[^\s]+@[^\s]+')
    text = re.sub(regex, "<EMAIL>", text)

    # Remove todos os caracteres não alfanuméricos
    regex = re.compile('[^A-Za-z0-9]+')
    text = re.sub(regex, " ", text)

    # Substitui os números com ponto por "NUMERO"
    regex = re.compile('[0-9]+.[0-9]+')
    text = re.sub(regex, "NUMERO", text)

    # Substitui os números com vírgula por "NUMERO"
    regex = re.compile('[0-9]+,[0-9]+')
    text = re.sub(regex, "NUMERO", text)

    # Substitui os números por "NUMERO"
    regex = re.compile('[0-9]+')
    text = re.sub(regex, "NUMERO", text)

    # Normaliza os espaçamentos
    text = ' '.join(text.split())

    # Separa o texto em palavras
    words = text.split()

    # Trunca o texto para apenas 200 palavras
    words = words[0:200]

    # Remove as stopwords
    words = [w for w in words if not w in nltk.corpus.stopwords.words('portuguese')]

    # Aplica estemização nas palavras
    stemmer_method = RSLPStemmer()
    words = [ stemmer_method.stem(w) for w in words ]

    # Remove as palavras com apenas um caractere
    words = [ w for w in words if len(w)>1 ]

    # Junta o texto novamente
    text = " ".join( words )

    # Retorna o texto pré-processado
    return text

# Função para aplicar o pré-processamento nos dataframes de treino e teste
def preprocess(df_train, df_test):
    # Aplica o pré-processamento para todas as linhas do dataframe de treino
    for i, row in df_train.iterrows():
        df_train.at[i, "Text"] = preprocess_text(row.Text)

    # Aplica o pré-processamento para todas as linhas do dataframe de teste
    for i, row in df_test.iterrows():
        df_test.at[i, "Text"] = preprocess_text(row.Text)

    # Retorna os dataframes pré-processados
    return df_train, df_test