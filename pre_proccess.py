import os
import nltk
import re
import pandas as pd
import unicodedata
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
#nltk.download('stopwords')
#nltk.download('punkt_tab')

def carregaDados():
    caminho = "data_set"
    caminho_completo = os.path.join(caminho, "SMSSpamCollection")
    return pd.read_csv(caminho_completo, sep="\t", header=None, names=['label', 'text'])


stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def remove_accent(text):
    return ''.join(
        c for c in unicodedata.normalize('NFD', text) 
        if unicodedata.category(c) != 'Mn' 
    )


def clean_text(text, tokenize=False):
    text = remove_accent(text.lower())  #remove os acentos enquanto deixa tudo minusculo
    text = re.sub(r'\d+', 'numtoken', text)  #substitui valores numericos por 'numtoken'
    text = re.sub(r'[^a-z\s]', '', text) #remove caracteres especiais
    text = re.sub(r"\s+", " ", text) #remove espacos desnecessarios

    words = word_tokenize(text)

    filtered = []
    for w in words:  #para cada palavra
        if w not in stop_words and len(w) > 2:  #testa se é uma stopword
            raiz = stemmer.stem(w) #reduz a palavra ao radical
            filtered.append(raiz)  # guarda


    if tokenize:
        return filtered
    return ' '.join(filtered)

def preProccess():
    df = carregaDados()

    textos_limpos = []
    for texto in df['text']:
        textos_limpos.append(clean_text(texto)) #limpa texto por texto e insere na lista
    df['text_clean'] = textos_limpos # coluna nova no dataframe com os textos limpos

    return df



def testa():
    df_processado = preProccess()
    print(df_processado[['text', 'text_clean']].head())  #mostra 5 exemplos comparando 'text' com 'text_clea'


testa()