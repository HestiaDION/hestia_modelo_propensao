
## **Pipeline de Classificação de Modelos de Machine Learning**

Este projeto é uma pipeline de machine learning desenvolvida em Python com o objetivo de realizar a classificação de dados usando algoritmos como KNN, Decision Tree e Naive Bayes. O código foi projetado para ser flexível, permitindo a busca pelos melhores hiperparâmetros, o pré-processamento automatizado de dados e a seleção do melhor modelo baseado na acurácia.

----------

## Estrutura do Projeto

### 1. **Importações de Bibliotecas**

O projeto utiliza bibliotecas populares para manipulação de dados, visualização, construção de modelos e avaliação de desempenho. Algumas das bibliotecas usadas incluem:

- **Pandas** e **NumPy** para manipulação de dados;
- **Matplotlib**, **Seaborn** e **Plotly** para visualização de dados;
- **Scikit-learn** para pré-processamento, construção e validação de modelos;
- **Imbalanced-learn** para balanceamento de dados com técnicas como SMOTE;
- **Joblib** para salvamento do modelo final.

### 2. **Carregamento e Análise Inicial dos Dados**

O dataset é lido a partir de um arquivo Excel. Em seguida, são geradas estatísticas descritivas para as variáveis numéricas e categóricas, além da visualização da distribuição das variáveis categóricas por meio de gráficos de barras.

### 3. **Separação de Atributos e Variáveis Alvo**

Os dados são divididos em variáveis de entrada (`X`) e a variável de saída (`y`), que é a variável que o modelo deverá prever. A variável alvo neste caso é a coluna `'Usaria o App'`.

### 4. **Pré-processamento de Dados**

É criado um pipeline de pré-processamento utilizando `ColumnTransformer` para realizar a codificação de variáveis categóricas com `OrdinalEncoder` e a padronização de variáveis numéricas com `StandardScaler`. O pipeline também inclui o uso de PCA para redução de dimensionalidade.

### 5. **Divisão de Dados em Treino e Teste**

Os dados são divididos em treino (70%) e teste (30%) utilizando `train_test_split`.

### 6. **Construção e Otimização de Modelos**

- **Decision Tree**: Usa-se `GridSearchCV` para encontrar os melhores hiperparâmetros para a árvore de decisão.
- **KNN**: Realiza-se a busca do melhor valor de `k` com `GridSearchCV`.
- **Naive Bayes**: Utiliza-se a implementação padrão de Naive Bayes Gaussiano.

### 7. **Treinamento dos Modelos**

Todos os modelos ajustados são treinados nos dados de treino, e suas previsões são feitas nos dados de teste.

### 8. **Avaliação de Desempenho**

Utiliza-se `classification_report` para gerar métricas como precisão, recall e F1-score para cada modelo. Além disso, é realizada uma validação cruzada (`cross_val_score`) para avaliar a performance média dos modelos em várias divisões dos dados de treino.

### 9. **Seleção do Melhor Modelo**

Com base na acurácia média da validação cruzada, o melhor modelo é selecionado. Este modelo é treinado novamente em todo o conjunto de dados de treino.

### 10. **Salvamento do Modelo**

O modelo final é serializado e salvo em um arquivo `.pkl` usando `Joblib`, permitindo que ele seja carregado e reutilizado posteriormente.
