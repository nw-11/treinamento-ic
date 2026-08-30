import pandas as pd
from pre_proccess import preProccess
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tfidf import cria_tfidf



df = preProccess()

x, vetorizador = cria_tfidf(df)
y = df['label']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

#modelo knn
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)   #.fit -> treina o modelo com os textos e o label
pred_knn = knn.predict(x_test) #.predict -> faz o teste
acc_knn = accuracy_score(y_test, pred_knn) #recebe as notas com base no y_test(accuracy e macrof1)
f1_knn = f1_score(y_test, pred_knn, average='macro')

#modelo svm
svm = SVC()
svm.fit(x_train, y_train) #.fit -> treina o modelo com os textos e o label
pred_svm = svm.predict(x_test) #.predict -> faz o teste

acc_svm = accuracy_score(y_test, pred_svm) #recebe as notas com base no y_test(accuracy e macrof1)
f1_svm = f1_score(y_test, pred_svm, average='macro')

#modelo decision tree
arvore = DecisionTreeClassifier(random_state=42)
arvore.fit(x_train, y_train)    #.fit -> treina o modelo com os textos e o label
pred_arvore = arvore.predict(x_test)    #.predict -> faz o teste
acc_arvore = accuracy_score(y_test, pred_arvore)    #recebe as notas com base no y_test(accuracy e macrof1)
f1_arvore = f1_score(y_test, pred_arvore, average='macro') 



print('--- RESULTADOS ---')
print(f'KNN -> acurácia: {acc_knn:.4f} | Macro F1: {f1_knn:.4f}')
print(f'SVM -> acurácia: {acc_svm:.4f} | Macro F1: {f1_svm:.4f}')
print(f'decision tree -> acurácia: {acc_arvore:.4f} | Macro F1: {f1_arvore:.4f}')