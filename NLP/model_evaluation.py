import pandas as pd
from pre_proccess import preProccess
from scipy import stats
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tfidf import cria_tfidf

def model_evaluation():

    df = preProccess()
    X, vetorizador = cria_tfidf(df)
    y = df['label']

    modelos = {
        'KNN': KNeighborsClassifier(n_neighbors=7, metric='cosine'),    #definindo os models(usando os melhores parametros encontrados no passo 5)
        'SVM': SVC(C=1, kernel='linear'),                              
        'Árvore': DecisionTreeClassifier(criterion='gini', max_depth=30, random_state=42)
    }

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    notas_dos_modelos = {}

    print('--- MÉDIAS E DESVIOS (DESEMPENHO GERAL) ---')
    for nome, modelo in modelos.items():
        notas = cross_val_score(modelo, X, y, cv=cv, scoring='f1_macro', n_jobs=-1)#testa para CADA fold, usando os outros 9 de treino, e retorna a lista com as 10 notas
        notas_dos_modelos[nome] = notas       #guarda a lista com as 10 notas para cada modelo
        
        print(f'{nome} | media: {notas.mean():.4f} | desvio padrao: {notas.std():.4f}')

    print('\n--- COMPARAÇÃO ESTATÍSTICA ---')
    
    pares = [('KNN', 'SVM'), ('KNN', 'Árvore'), ('SVM', 'Árvore')]
    
    limite_padrao = 0.05

    limite_bonferroni = limite_padrao/len(pares) # correção de bonferroni para fazer o teste T
    print(f'Nível de significância exigido (Bonferroni): {limite_bonferroni:.4f}\n')

    for m1, m2 in pares:
        # Compara as 10 notas do Modelo 1 com as 10 notas do Modelo 2 e retorna t(sinal/oscilacao) e p(ruido)

        estatistica, p = stats.ttest_rel(notas_dos_modelos[m1], notas_dos_modelos[m2])
        
        # Se o p for menor, pouco ruido, há diferenca real
        melhormodelo = None
        if p < limite_bonferroni: 
            if(estatistica > 0):        #t(sinal/oscilacao) = modelo1 - modelo2
                melhormodelo = m1           
            elif(estatistica < 0):
                melhormodelo = m2
            print(f'{m1} vs {m2}')
            print(f'   -> p-valor: {p:.5f} | Veredito: Diferença Significativa (Há um superior)')
            print(f'   -> melhor modelo: {melhormodelo}')
            
        else:        #ruido maior do que o limite, empate.
            print(f'{m1} vs {m2}')
            print(f'   -> p-valor: {p:.5f} | Veredito: Estatisticamente Equivalentes(empate)')          

model_evaluation()

#concluimos que na nossa base de dados, o melhor modelo é o SVM