# Importa as bibliotecas
import xml.etree.ElementTree as ET

# Função para extrair o texto a partir de um arquivo XML (das tags "<p>")
def _extract_text_from_xml(xml_file):
    # Lê o arquivo XML e cria uma árvore de elementos
    tree = ET.parse(xml_file)

    # Obtém o elemento raiz da árvore
    root = tree.getroot()

    # Inicializa uma variável para armazenar o texto extraído
    text = ""

    # Itera sobre todos os elementos "p" (parágrafos) na árvore
    for elem in root.iter("p"):
        # Verifica se o elemento possui texto (não vazio)
        if elem.text:
            # Adiciona o texto do elemento (removendo espaços em branco) à variável "text"
            text += elem.text.strip() + " "

    # Retorna o texto extraído (de todos os paragrafos do XML), removendo espaços em branco adicionais
    return text.strip()

# Função para transformar arquivos XML em texto
# substituindo a coluna "ID" (nomes dos arquivos XML) dos dataframes 
# em uma coluna "Text", contendo o texto extraído para cada XML
def transform_xml_to_text(df_train, df_test):
    # Cria uma cópia dos dataframes
    new_df_train = df_train.copy()
    new_df_test = df_test.copy()

    # Inicializa a coluna de texto do dataframe de treino
    new_df_train["Text"] = ""

    # Itera cada linha do dataframe de treino
    for idx, row in new_df_train.iterrows():
        # Cria uma váriavel contendo o nome do arquivo XML
        xml_file = row["ID"]
        try:
            # Extrai o texto do arquivo XML
            file_text = _extract_text_from_xml("data/news/"+xml_file)
            # Insere o texto extraido na linha atual
            new_df_train.at[idx, "Text"] = file_text
        except FileNotFoundError:
            pass
    
    # Inicializa a coluna de texto do dataframe de teste
    new_df_test["Text"] = ""

    # Itera cada linha do dataframe de treino
    for idx, row in new_df_test.iterrows():
        # Cria uma váriavel contendo o nome do arquivo XML
        xml_file = row["ID"]
        try:
            # Extrai o texto do arquivo XML
            file_text = _extract_text_from_xml("data/news/"+xml_file)
            # Insere o texto extraido na linha atual
            new_df_test.at[idx, "Text"] = file_text
        except FileNotFoundError:
            pass
    
    # Remove a coluna "ID" (que contém os nomes dos arquivos XML) para os dois dataframes
    new_df_train.drop(columns=["ID"], inplace=True)
    new_df_test.drop(columns=["ID"], inplace=True)

    # Retorna os dois dataframes"
    return new_df_train, new_df_test