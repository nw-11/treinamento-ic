import os
import pandas as pd
import matplotlib.pyplot as plt

def carregaDados():
    caminho = "data_set"
    caminho_completo = os.path.join(caminho, "SMSSpamCollection")
    return pd.read_csv(caminho_completo, sep="\t", header=None, names=['label', 'text'])

def plotaGrafico(df):
    #criando dois graficos com subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # --- Gráfico 1: Quantidade total de mensagens ---
    df['label'].value_counts().plot(kind='bar', color=['blue', 'red'], ax=axes[0])
    axes[0].set_title("Quantidade de Mensagens")
    axes[0].set_xlabel('Classe')
    axes[0].set_ylabel('Quantidade')
    axes[0].tick_params(axis='x', rotation=0)

    # --- Gráfico 2: Média de palavras por classe ---
    medias = df.groupby('label')['num_palavras'].mean()
    medias.plot(kind='bar', color=['skyblue', 'salmon'], ax=axes[1])
    axes[1].set_title("Média de Palavras por Classe")
    axes[1].set_xlabel('Classe')
    axes[1].set_ylabel('Média de Palavras')
    axes[1].tick_params(axis='x', rotation=0)

    plt.tight_layout() # Ajusta os espaços para os textos não se sobreporem
    plt.show() # Abre uma única janela com os dois gráficos


def analisePalavras(df):
    resultado = []
    for frase in df['text']:
        quantidade = len(str(frase).split())
        resultado.append(quantidade)
    df['num_palavras'] = resultado
    return df
    

def main():
    df = carregaDados()
    df = analisePalavras(df)
    plotaGrafico(df)
    print("Total de documentos lidos: {df['label'].count()}")

main()

