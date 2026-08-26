import pandas as pd
from pre_proccess import preProccess
from sklearn.feature_extraction.text import TfidfVectorizer

def cria_tfidf(df):
    vetorizador = TfidfVectorizer()
    x_tfidf = vetorizador.fit_transform(df['text_clean'])
    return x_tfidf, vetorizador


def testa():
    df = preProccess() #carrega e pre processa os dados

    x_tfidf, vetorizador = cria_tfidf(df)

    print(f'Formato da matriz gerada: {x_tfidf.shape}')   
    print(
        f'Vocabulário total: {len(vetorizador.get_feature_names_out())} palavras'
    )

testa()
     