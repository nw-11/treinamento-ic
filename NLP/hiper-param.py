import pandas as pd
from pre_proccess import preProccess
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tfidf import cria_tfidf

def hiperparam():
    df = preProccess()
    x, vetorizador = cria_tfidf(df)
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(    
    x, y, test_size=0.2, random_state=42, stratify=y        #preparamos os parametros de treino
    )
    grade_knn = {
        'n_neighbors': [3, 5, 7, 9],
        'metric': ['euclidean', 'cosine'],    #nas grades, podemos colocar os parametros que quisermos, com os valores que quisermos do dominio
                                                #por exemplo nessas grades, estamos usando apenas 2 parametros usando 4 valores no primeiro
                                                #e 2 valores no segundo, 4x2 é o numero de combinacoes = 8 que vai ser testada
    }
    grade_svm = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
    }

    grade_arvore = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30],
    }

    modelos_e_grades = [
        ('KNN', KNeighborsClassifier(), grade_knn),
        ('SVM', SVC(), grade_svm),
        ('Árvore de Decisão', DecisionTreeClassifier(random_state=42), grade_arvore),
    ]

    melhores_modelos = {}

    for nome, modelo, grade in modelos_e_grades:
        print(f'Otimizando hiperparâmetros para: {nome}...')

        grid = GridSearchCV(
            estimator=modelo,
            param_grid=grade,
            scoring='f1_macro',             #usa macrof1 pra analisar os resultados dos testes
            cv=5,                   
            n_jobs=-1,                          #percorre cada modelo
        )

        grid.fit(X_train, y_train)              #para treinar o modelo no nosso gerenciador de testes(cv = 5; logo sao 40 treinos, porque sao 8 combinacoes, 8x5)

        melhores_modelos[nome] = grid.best_estimator_    #guarda no dicionario o modelo final e treinado, com os melhores parametros encontrados

        print(f'-> Melhores parâmetros para {nome}: {grid.best_params_}')           #imprime
        print(f'-> Melhor Macro F1 médio no treino: {grid.best_score_:.4f}\n')   

    print('=' * 60)
    print('DESEMPENHO DOS MODELOS')
    print('=' * 60)

    for nome, modelo_otimizado in melhores_modelos.items():
        predicoes = modelo_otimizado.predict(X_test)            #percorre o dicionario pra mostrar os resultados dos modelos "melhorados"
        print(f'\nRelatório para {nome}:')                      
        print(classification_report(y_test, predicoes, digits=4))

#hiperparam() #teste