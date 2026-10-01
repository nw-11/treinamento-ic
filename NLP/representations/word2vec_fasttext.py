import pandas as pd
from pre_proccess_word2vec import preProccess
from gensim.models import Word2Vec, FastText

import numpy as np

def frase_para_vetor(palavras, ft_model):

    if not palavras:
        return np.zeros(ft_model.get_dimension())
    # Obtém o vetor FastText para cada palavra da frase
    vetores = [ft_model.get_word_vector(palavra) for palavra in palavras]
    
    # Retorna o vetor médio da frase (1D)
    return np.mean(vetores, axis=0)


def cria_fasttext(df, ft_model):
    """
    Gera a matriz X (2D) pronta para o Scikit-Learn.
    """
    X = np.array([frase_para_vetor(texto, ft_model) for texto in df['text_clean']])
    return X