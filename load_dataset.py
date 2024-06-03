# Importa as bibliotecas
import subprocess
import os
from zipfile import ZipFile

# Função para realizar o download de arquivos
# É necessário indicar a URL e o diretório para o download
def _download_files(url, path):
    # Cria o comando CMD utilizando wget
    command = ['wget', '-q', url, '-O', '%s' %(path)]

    # Cria um novo subprocesso indicando o comando a ser executado 
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Aguarda o comando do subprocesso ser executado e recebe a saída do comando
    _, stderr = process.communicate()

    # Verifica se o comando foi finalizado com sucesso e exibe uma mensagem de sucesso ou de erro
    if process.returncode == 0:
        print(f'Processo finalizado.')
    else:
        print(f'Houve um erro: {stderr.decode("utf-8")}')

# Função para descompactar um arquivo .zip em uma determinada pasta
def _unzip(path, path_folder):
    try:
        # Cria um objeto ZipFile com o arquivo especificado
        z = ZipFile(path, 'r')

        # Extrai todos os arquivos do zip para a pasta especificada
        z.extractall(path_folder)

        # "Fecha" o objeto ZipFile 
        z.close()

        # Exibe a mensagem de sucesso
        print("Arquivo descompactado com sucesso!")
    except:
        # Exibe a mensagem de erro
        print("Houve um erro ao tentar descompactar o arquivo")

# Função para realizar o download e descompactação dos arquivos do dataset da competição 
# e também do modelo word2vec pré-treinado
def load_dataset():
    # Váriaveis armazenando o nome das pastas desejadas
    dataset_folder = 'data/'
    pre_trained_model_folder = 'pre-trained-models/'

    # Váriaveis armazenando o caminho final dos arquivos zip
    dataset_zip_file_path = dataset_folder + 'arquivos_competicao.zip'
    pre_trained_model_zip_file_path = pre_trained_model_folder + 'skip_s50.zip'

    # Váriaveis com as URLs de download do dataset da competição e do modelo pré-treinado
    dataset_download_url = 'https://www.dropbox.com/scl/fo/2vh6qw9x2ae8zoma7md98/ALGVx_ju4WiPjneRZ68crs8?rlkey=s919cfytsov4bafkvnufmpgwg&e=1&st=qjynn11z&dl=0'
    pre_trained_model_download_url = 'http://143.107.183.175:22980/download.php?file=embeddings/word2vec/skip_s50.zip'

    # Cria as pastas caso ainda não existam
    if not os.path.isdir(dataset_folder and pre_trained_model_folder):
        os.mkdir(dataset_folder)
        os.mkdir(pre_trained_model_folder)

    # Realiza o download e descompactação do dataset da competição
    _download_files(dataset_download_url, dataset_zip_file_path)
    _unzip(dataset_zip_file_path, dataset_folder)

    # Realiza o download e descompactação do modelo pré-treinado (word2vec)
    _download_files(pre_trained_model_download_url, pre_trained_model_zip_file_path)
    _unzip(pre_trained_model_zip_file_path, pre_trained_model_folder)