# Exercicios

README.md - Wine Quality and Iris Classification.
#
Este é um repositório com códigos para classificação utilizando o conjunto de dados de qualidade de vinhos e o conjunto de dados Iris.

Requisitos
Para executar os códigos neste repositório, você precisa ter as seguintes bibliotecas instaladas:

- pandas
- matplotlib
- seaborn
- scikit-learn

# Você pode instalá-las usando o seguinte comando:

- pip install pandas 
- pip install matplotlib 
- pip install seaborn 
- pip install scikit-learn

# Wine Quality Dataset
O conjunto de dados Wine Quality (WineQT.csv) é um arquivo CSV contendo informações sobre a qualidade de diferentes vinhos. Ele contém as seguintes colunas:

- "fixed acidity"
- "volatile acidity"
- "citric acid"
- "residual sugar"
- "chlorides"
- "free sulfur dioxide"
- "total sulfur dioxide"
- "density"
- "pH"
- "sulphates"
- "alcohol"
- "quality" (variável alvo)

# Explorando o Dataset
Você pode visualizar as primeiras 10 linhas do conjunto de dados executando o seguinte código:

- import pandas as pd

- dataset = pd.read_csv("WineQT.csv")
- print(dataset.head(10))

Para obter informações sobre o conjunto de dados, como o número de linhas e colunas, tipos de dados e se há valores nulos, você pode executar:

- print(dataset.shape)

- dataset.info()

- dataset.isnull()

# Análise e Visualizações

Há várias análises e visualizações que podem ser realizadas neste conjunto de dados. Alguns exemplos incluem:

# Verificação da distribuição da qualidade do vinho:

- import matplotlib.pyplot as plt
- import seaborn as sns

sns.catplot(x='quality', data=dataset, kind='count', margin_titles=True)
plt.title("Distribuição da Qualidade do Vinho")
plt.show()

# Análise da correlação entre as variáveis:

- correlation = dataset.corr()
- plt.figure(figsize=(10, 9))
- sns.heatmap(correlation, cbar=True, square=True, fmt=".2f", annot=True, annot_kws={'size': 8})
- plt.title("Matriz de Correlação")
- plt.show()

# Visualização das relações entre qualidade do vinho e outras variáveis:

- plt.figure(figsize=(9, 8))
- sns.barplot(x="quality", y="volatile acidity", data=dataset)
- plt.title("Acidez Volátil vs. Qualidade")
- plt.show()

E assim por diante, você pode explorar outras visualizações relacionadas às demais variáveis do conjunto de dados.

# Modelagem

O código também inclui a modelagem utilizando o algoritmo Random Forest para classificar a qualidade do vinho:

- from sklearn.model_selection import train_test_split
- from sklearn.ensemble import RandomForestClassifier
- from sklearn.metrics import accuracy_score

# Separar as colunas de features e a variável alvo

- X = dataset.drop(['quality', 'Id'], axis=1)
- y = dataset['quality']

# Dividir os dados em conjuntos de treinamento e teste

- X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar e treinar o modelo RandomForestClassifier

- model = RandomForestClassifier()
- model.fit(X_train, y_train)

# Fazer previsões usando os dados de teste

- y_pred = model.predict(X_test)

# Calcular a acurácia do modelo

- accuracy = accuracy_score(y_test, y_pred)
- print(f'Acurácia do modelo: {accuracy}')

# Iris Dataset

O conjunto de dados Iris é um conjunto de dados clássico que contém informações sobre três espécies de íris: Iris-setosa, Iris-versicolor e Iris-virginica. Ele é carregado a partir da biblioteca scikit-learn.

# Modelagem

O código inclui a modelagem utilizando o algoritmo K-Nearest Neighbors (KNN) para classificar as espécies de íris:

- from sklearn.datasets import load_iris
- from sklearn.model_selection import train_test_split
- from sklearn.neighbors import KNeighborsClassifier
- from sklearn.metrics import accuracy_score

# Carregar o conjunto de dados Iris

- data = load_iris()
- X = data.data
- y = data.target

# Dividir os dados em conjuntos de treinamento e teste

- X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar o modelo KNeighborsClassifier

- model = KNeighborsClassifier(n_neighbors=3)  # Número de vizinhos é definido como 3 neste exemplo

# Treinar o modelo com os dados de treinamento

- model.fit(X_train, y_train)

# Fazer previsões usando os dados de teste

- y_pred = model.predict(X_test)

# Calcular a acurácia do modelo

- accuracy = accuracy_score(y_test, y_pred)
- print(f'Acurácia do modelo KNN para classificação: {accuracy}')

# Validação Cruzada e Métricas de Avaliação

O código também inclui exemplos de como realizar a validação cruzada e calcular diversas métricas de avaliação:

# Validação cruzada com o algoritmo RandomForestClassifier:

- from sklearn.model_selection import cross_val_score

# Inicializar o modelo

- model = RandomForestClassifier()

# Realizar a validação cruzada e obter as acurácias em cada fold

- accuracies = cross_val_score(model, X, y, cv=5)

# Calcular a média e o desvio padrão das acurácias
- mean_accuracy = accuracies.mean()
- std_accuracy = accuracies.std()

- print(f'Acurácia média da validação cruzada: {mean_accuracy}')
- print(f'Desvio padrão da acurácia da validação cruzada: {std_accuracy}')

# Métricas de avaliação para validação cruzada:

- from sklearn.metrics import precision_score, recall_score, f1_score
- from sklearn.model_selection import cross_val_predict

# Obter as previsões do modelo usando validação cruzada

- y_pred_cv = cross_val_predict(model, X, y, cv=5)

# Calcular as métricas

- precision = precision_score(y, y_pred_cv, average='weighted')
- recall = recall_score(y, y_pred_cv, average='weighted')
- f1 = f1_score(y, y_pred_cv, average='weighted')

- print(f'Precisão da validação cruzada: {precision}')
- print(f'Recall da validação cruzada: {recall}')
- print(f'F1-score da validação cruzada: {f1}')

# Métricas de avaliação para a classificação do conjunto de teste usando o modelo KNN:

- from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Calcular a acurácia do modelo

- accuracy = accuracy_score(y_test, y_pred)

# Calcular precisão, recall e F1-score

- precision = precision_score(y_test, y_pred, average='weighted')
- recall = recall_score(y_test, y_pred, average='weighted')
- f1 = f1_score(y_test, y_pred, average='weighted')

# Calcular a matriz de confusão

- confusion_mat = confusion_matrix(y_test, y_pred)

- print(f'Acurácia: {accuracy}')
- print(f'Precisão: {precision}')
- print(f'Recall: {recall}')
- print(f'F1-score: {f1}')
- print('Matriz de Confusão:')
- print(confusion_mat)

# Como usar

- Para executar o código e obter os resultados, basta copiar os trechos relevantes e colá-los em um ambiente Python com as bibliotecas necessárias instaladas. Certifique-se de ter o arquivo WineQT.csv no mesmo diretório do arquivo de script que está utilizando para ler o conjunto de dados.

- Este é apenas um exemplo de um README.md e você pode adicionar mais detalhes conforme necessário para o seu projeto. Certifique-se de incluir informações relevantes sobre o contexto do projeto, como instalar e executar o código, e uma descrição mais detalhada das análises e modelos implementados.


