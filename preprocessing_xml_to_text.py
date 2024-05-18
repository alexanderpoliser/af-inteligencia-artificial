import xml.etree.ElementTree as ET

def extract_text_from_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    text = ""
    for elem in root.iter("p"):
        if elem.text:
            text += elem.text.strip() + " "
    return text.strip()

def transform_xml_to_text(df_train, df_test):
    new_df_train = df_train.copy()
    new_df_test = df_test.copy()

    new_df_train["Text"] = ""

    for idx, row in new_df_train.iterrows():
        xml_file = row["ID"]
        try:
            file_text = extract_text_from_xml("data/news/"+xml_file)
            new_df_train.at[idx, "Text"] = file_text
        except FileNotFoundError:
            pass
    
    new_df_test["Text"] = ""

    for idx, row in new_df_test.iterrows():
        xml_file = row["ID"]
        try:
            file_text = extract_text_from_xml("data/news/"+xml_file)
            new_df_test.at[idx, "Text"] = file_text
        except FileNotFoundError:
            pass
    
    new_df_train.drop(columns=["ID"], inplace=True)
    new_df_test.drop(columns=["ID"], inplace=True)

    return new_df_train, new_df_test