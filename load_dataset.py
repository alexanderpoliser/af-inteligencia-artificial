import subprocess
import os
from zipfile import ZipFile

def download_files(url, path):

    cmd = ['wget', '-q', url, '-O', '%s' %(path)]

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    _, stderr = process.communicate()

    if process.returncode == 0:
        print(f'Processo finalizado.')
    else:
        print(f'Houve um erro: {stderr.decode("utf-8")}')

def unzip(path, pathFolder):

    try:
        z = ZipFile(path, 'r')
        z.extractall(pathFolder)
        z.close()

        print("Arquivo descompactado com sucesso!")
    except:
        print("Houve um erro ao tentar descompactar o arquivo")

pathFiles1 = 'data/'
pathFiles2 = 'pre-trained-models/'

if not os.path.isdir(pathFiles1 and pathFiles2):
    os.mkdir(pathFiles1)
    os.mkdir(pathFiles2)

url1 = 'https://www.dropbox.com/scl/fo/2vh6qw9x2ae8zoma7md98/ALGVx_ju4WiPjneRZ68crs8?rlkey=s919cfytsov4bafkvnufmpgwg&e=1&st=qjynn11z&dl=0'
datasetPath1 = pathFiles1 + 'arquivos_competicao.zip'

url2 = 'http://143.107.183.175:22980/download.php?file=embeddings/word2vec/skip_s50.zip'
datasetPath2 = pathFiles2 + 'skip_s50.zip'


def load_dataset():
    download_files(url1, datasetPath1)
    unzip(datasetPath1, pathFiles1)

    download_files(url2, datasetPath2)
    unzip(datasetPath2, pathFiles2)