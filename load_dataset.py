import subprocess
import os
from zipfile import ZipFile

def downloadFiles(url, path):

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

pathFiles = 'data/'

if not os.path.isdir(pathFiles):
    os.mkdir(pathFiles)

url = 'https://www.dropbox.com/scl/fo/2vh6qw9x2ae8zoma7md98/ALGVx_ju4WiPjneRZ68crs8?rlkey=s919cfytsov4bafkvnufmpgwg&e=1&st=qjynn11z&dl=0'
datasetPath = pathFiles + 'arquivos_competicao.zip'

downloadFiles(url, datasetPath)
unzip(datasetPath, pathFiles)