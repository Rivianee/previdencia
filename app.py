{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyOGoj3RBkAEg32gF7/LIjcm",
      "include_colab_link": True
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Rivianee/previdencia/blob/main/app.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import streamlit as st\n",
        "import pandas as pd\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.preprocessing import OneHotEncoder\n",
        "from sklearn.compose import ColumnTransformer\n",
        "from sklearn.pipeline import Pipeline\n",
        "from sklearn.impute import SimpleImputer\n",
        "from lightgbm import LGBMClassifier\n",
        "from sklearn.metrics import accuracy_score\n",
        "\n",
        "# Título da aplicação\n",
        "st.title('Previsão de Portabilidade')\n",
        "\n",
        "# Função para carregar e processar dados\n",
        "def load_and_process_data(file_path):\n",
        "    # Carregar dados\n",
        "    data = pd.read_excel(file_path)\n",
        "\n",
        "    # Remover espaços em branco dos nomes de colunas\n",
        "    data.columns = data.columns.str.replace(' ', '_')\n",
        "\n",
        "    # Separar features (X) e target (y)\n",
        "    X = data.drop(columns=['fez_portabilidade', 'nm_participante'], errors='ignore')\n",
        "    y = data['fez_portabilidade']\n",
        "\n",
        "    # Identificar colunas numéricas e categóricas\n",
        "    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()\n",
        "    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()\n",
        "\n",
        "    # Pipeline de pré-processamento\n",
        "    numeric_transformer = Pipeline(steps=[\n",
        "        ('imputer', SimpleImputer(strategy='median'))\n",
        "    ])\n",
        "\n",
        "    categorical_transformer = Pipeline(steps=[\n",
        "        ('imputer', SimpleImputer(strategy='most_frequent')),\n",
        "        ('onehot', OneHotEncoder(handle_unknown='ignore'))\n",
        "    ])\n",
        "\n",
        "    preprocessor = ColumnTransformer(\n",
        "        transformers=[\n",
        "            ('num', numeric_transformer, numeric_cols),\n",
        "            ('cat', categorical_transformer, categorical_cols)\n",
        "        ])\n",
        "\n",
        "    # Dividir dados em treino e teste\n",
        "    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
        "\n",
        "    # Pré-processar dados de treino\n",
        "    X_train_preprocessed = preprocessor.fit_transform(X_train)\n",
        "\n",
        "    # Treinar modelo\n",
        "    model = LGBMClassifier(random_state=42)\n",
        "    model.fit(X_train_preprocessed, y_train)\n",
        "\n",
        "    # Avaliar desempenho do modelo nos dados de teste\n",
        "    X_test_preprocessed = preprocessor.transform(X_test)\n",
        "    y_pred = model.predict(X_test_preprocessed)\n",
        "    accuracy = accuracy_score(y_test, y_pred)\n",
        "\n",
        "    st.write(f'### Desempenho do Modelo')\n",
        "    st.write(f'Acurácia: {accuracy:.2f}')\n",
        "\n",
        "    return X, y, model, preprocessor, categorical_cols\n",
        "\n",
        "# Função para realizar a previsão\n",
        "def predict(model, original_data, preprocessor, input_data, categorical_cols):\n",
        "    # Transformar os dados de entrada usando o preprocessor\n",
        "    input_data_transformed = preprocessor.transform(input_data)\n",
        "\n",
        "    # Fazer a previsão usando o modelo treinado\n",
        "    prediction = model.predict(input_data_transformed)\n",
        "    prediction_proba = model.predict_proba(input_data_transformed)\n",
        "\n",
        "    # Exibir a previsão\n",
        "    st.write('### Resultado da Previsão:')\n",
        "    if prediction[0] == 1:\n",
        "        st.write('O participante fez portabilidade.')\n",
        "    else:\n",
        "        st.write('O participante não fez portabilidade.')\n",
        "\n",
        "    st.write('### Probabilidades:')\n",
        "    st.write(f'Probabilidade de não fazer portabilidade: {prediction_proba[0][0]:.2f}')\n",
        "    st.write(f'Probabilidade de fazer portabilidade: {prediction_proba[0][1]:.2f}')\n",
        "\n",
        "# Componente para upload de arquivo\n",
        "uploaded_file = st.file_uploader(\"Carregue o arquivo Excel\", type=['xlsx'])\n",
        "\n",
        "# Verificar se um arquivo foi carregado\n",
        "if uploaded_file is not None:\n",
        "    # Carregar os dados\n",
        "    X, y, model, preprocessor, categorical_cols = load_and_process_data(uploaded_file)\n",
        "\n",
        "    # Mostrar os dados carregados\n",
        "    st.write('### Dados Carregados:')\n",
        "    st.write(X.head())\n",
        "\n",
        "    # Interface para inserir dados e fazer a previsão\n",
        "    st.write('### Faça uma previsão com base nos dados carregados:')\n",
        "    input_data = {}\n",
        "    for col in X.columns:\n",
        "        if col in categorical_cols:\n",
        "            input_data[col] = st.selectbox(f'Selecione {col}', [''] + X[col].unique().tolist())\n",
        "        else:\n",
        "            input_data[col] = st.number_input(f'Insira {col}', value=0.0)\n",
        "\n",
        "    if st.button('Fazer Previsão'):\n",
        "        input_df = pd.DataFrame([input_data])\n",
        "        predict(model, X, preprocessor, input_df, categorical_cols)\n",
        "\n",
        "else:\n",
        "    st.write('Aguardando upload do arquivo...')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "VaSJEdjbRkdS",
        "outputId": "8a046e82-9f69-4694-8852-74fc2dd91bf4"
      },
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "2024-07-23 13:29:18.072 \n",
            "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
            "  command:\n",
            "\n",
            "    streamlit run /usr/local/lib/python3.10/dist-packages/colab_kernel_launcher.py [ARGUMENTS]\n"
          ]
        }
      ]
    }
  ]
}
