import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score

# Título da aplicação
st.title('Previsão de Portabilidade')

# Função para carregar e processar dados
def load_and_process_data(file_path):
    # Carregar dados
    data = pd.read_excel(file_path)

    # Remover espaços em branco dos nomes de colunas
    data.columns = data.columns.str.replace(' ', '_')

    # Separar features (X) e target (y)
    X = data.drop(columns=['fez_portabilidade', 'nm_participante'], errors='ignore')
    y = data['fez_portabilidade']

    # Identificar colunas numéricas e categóricas
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

    # Pipeline de pré-processamento
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Dividir dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Pré-processar dados de treino
    X_train_preprocessed = preprocessor.fit_transform(X_train)

    # Treinar modelo
    model = LGBMClassifier(random_state=42)
    model.fit(X_train_preprocessed, y_train)

    # Avaliar desempenho do modelo nos dados de teste
    X_test_preprocessed = preprocessor.transform(X_test)
    y_pred = model.predict(X_test_preprocessed)
    accuracy = accuracy_score(y_test, y_pred)

    st.write(f'### Desempenho do Modelo')
    st.write(f'Acurácia: {accuracy:.2f}')

    return X, y, model, preprocessor, categorical_cols

# Função para realizar a previsão
def predict(model, original_data, preprocessor, input_data, categorical_cols):
    # Transformar os dados de entrada usando o preprocessor
    input_data_transformed = preprocessor.transform(input_data)

    # Fazer a previsão usando o modelo treinado
    prediction = model.predict(input_data_transformed)
    prediction_proba = model.predict_proba(input_data_transformed)

    # Exibir a previsão
    st.write('### Resultado da Previsão:')
    if prediction[0] == 1:
        st.write('O participante fez portabilidade.')
    else:
        st.write('O participante não fez portabilidade.')

    st.write('### Probabilidades:')
    st.write(f'Probabilidade de não fazer portabilidade: {prediction_proba[0][0]:.2f}')
    st.write(f'Probabilidade de fazer portabilidade: {prediction_proba[0][1]:.2f}')

# Componente para upload de arquivo
uploaded_file = st.file_uploader("Carregue o arquivo Excel", type=['xlsx'])

# Verificar se um arquivo foi carregado
if uploaded_file is not None:
    # Carregar os dados
    X, y, model, preprocessor, categorical_cols = load_and_process_data(uploaded_file)

    # Mostrar os dados carregados
    st.write('### Dados Carregados:')
    st.write(X.head())

    # Interface para inserir dados e fazer a previsão
    st.write('### Faça uma previsão com base nos dados carregados:')
    input_data = {}
    for col in X.columns:
        if col in categorical_cols:
            input_data[col] = st.selectbox(f'Selecione {col}', [''] + X[col].unique().tolist())
        else:
            input_data[col] = st.number_input(f'Insira {col}', value=0.0)

    if st.button('Fazer Previsão'):
        input_df = pd.DataFrame([input_data])
        predict(model, X, preprocessor, input_df, categorical_cols)

else:
    st.write('Aguardando upload do arquivo...')
