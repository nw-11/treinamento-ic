from modeling_embeddings import get_embeddings
from scipy import stats
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def model_evaluation(tipo_representacao='tfidf'):
    print(f'--- Avaliação Estatística usando: {tipo_representacao} ---')

    X, modelo_repr, df_clean = get_embeddings(tipo_representacao)
    y = df_clean['label']

    modelos = {
        'KNN': KNeighborsClassifier(n_neighbors=9, metric='cosine'),
        'SVM': SVC(C=10, kernel='rbf'),                             #usar os melhores parametros encontrados em hiper-param.py
        'Árvore': DecisionTreeClassifier(
            criterion='gini', max_depth=10, random_state=42
        ),
    }

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    notas_dos_modelos = {}

    for nome, modelo in modelos.items():
        notas = cross_val_score(
            modelo, X, y, cv=cv, scoring='f1_macro', n_jobs=-1
        )
        notas_dos_modelos[nome] = notas
        print(f'{nome} | media: {notas.mean():.4f} | desvio padrao: {notas.std():.4f}')

    print('\n--- COMPARAÇÃO ESTATÍSTICA ---')
    
    pares = [('KNN', 'SVM'), ('KNN', 'Árvore'), ('SVM', 'Árvore')]
    limite_padrao = 0.05
    limite_bonferroni = limite_padrao / len(pares) # correção de bonferroni para o teste T
    print(f'Nível de significância exigido (Bonferroni): {limite_bonferroni:.4f}\n')

    for m1, m2 in pares:
        estatistica, p = stats.ttest_rel(notas_dos_modelos[m1], notas_dos_modelos[m2])
        
        melhormodelo = None
        if p < limite_bonferroni: 
            if estatistica > 0:
                melhormodelo = m1           
            elif estatistica < 0:
                melhormodelo = m2
            print(f'{m1} vs {m2}')
            print(f'   -> p-valor: {p:.5f} | Veredito: Diferença Significativa (Há um superior)')
            print(f'   -> melhor modelo: {melhormodelo}')
        else:
            print(f'{m1} vs {m2}')
            print(f'   -> p-valor: {p:.5f} | Veredito: Estatisticamente Equivalentes (empate)')          


#model_evaluation('tfidf') #testar

# Concluimos que na nossa base de dados, o melhor modelo é o SVM usando tfidf