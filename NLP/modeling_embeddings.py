from gensim.models import Word2Vec, FastText
import numpy as np
from pre_proccess import preProccess
from sklearn.feature_extraction.text import TfidfVectorizer

def get_embeddings(tipo):
    df_clean = preProccess(tipo)
    if tipo == 'tfidf':
        vetorizador = TfidfVectorizer()
        # Converte o texto processado na matriz densa (2D array)
        X = vetorizador.fit_transform(df_clean['text_clean']).toarray()
        return X, vetorizador, df_clean
    
    tokenized_sentences = df_clean['text_clean']
    if tipo == 'w2v':
        model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=2, workers=4)
    elif tipo == 'fasttext':
        model = FastText(sentences=tokenized_sentences, vector_size=100, window=5, min_count=2, workers=4)
    else:
        raise ValueError(
            f"tipo '{tipo}' invalido. Escolha entre 'tfidf', 'w2v' ou 'fasttext'."
        )

    #transformar uma frase em um unico vetor

    def get_mean_vector(sentence_tokens):
        #pega o vetor de cada palavra, se a palavra existir no modelo
        vectors = [model.wv[word] for word in sentence_tokens if word in model.wv]
        if len(vectors) > 0:
            return np.mean(vectors, axis=0) #media aritmetica dos vetores
        else:
            return np.zeros(100) #vetor de zeros se nao encontrar nenhuma palavra
        
        #criar a matriz final
    X = np.array([get_mean_vector(tokens) for tokens in tokenized_sentences])

    return X, model, df_clean #retorna o X, nosso modelo usado,