{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "include_colab_link": true
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
        "<a href=\"https://colab.research.google.com/github/Rivianee/previdencia/blob/main/app_py.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install streamlit pandas lightgbm scikit-learn openpyxl"
      ],
      "metadata": {
        "id": "-6NG6DVDVZoz",
        "outputId": "c3200ba9-f81f-492f-af4c-4bedec0f7398",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: streamlit in /usr/local/lib/python3.10/dist-packages (1.36.0)\n",
            "Requirement already satisfied: pandas in /usr/local/lib/python3.10/dist-packages (2.0.3)\n",
            "Requirement already satisfied: lightgbm in /usr/local/lib/python3.10/dist-packages (4.1.0)\n",
            "Requirement already satisfied: scikit-learn in /usr/local/lib/python3.10/dist-packages (1.2.2)\n",
            "Requirement already satisfied: openpyxl in /usr/local/lib/python3.10/dist-packages (3.1.5)\n",
            "Requirement already satisfied: altair<6,>=4.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (4.2.2)\n",
            "Requirement already satisfied: blinker<2,>=1.0.0 in /usr/lib/python3/dist-packages (from streamlit) (1.4)\n",
            "Requirement already satisfied: cachetools<6,>=4.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (5.4.0)\n",
            "Requirement already satisfied: click<9,>=7.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (8.1.7)\n",
            "Requirement already satisfied: numpy<3,>=1.20 in /usr/local/lib/python3.10/dist-packages (from streamlit) (1.25.2)\n",
            "Requirement already satisfied: packaging<25,>=20 in /usr/local/lib/python3.10/dist-packages (from streamlit) (24.1)\n",
            "Requirement already satisfied: pillow<11,>=7.1.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (9.4.0)\n",
            "Requirement already satisfied: protobuf<6,>=3.20 in /usr/local/lib/python3.10/dist-packages (from streamlit) (3.20.3)\n",
            "Requirement already satisfied: pyarrow>=7.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (14.0.2)\n",
            "Requirement already satisfied: requests<3,>=2.27 in /usr/local/lib/python3.10/dist-packages (from streamlit) (2.31.0)\n",
            "Requirement already satisfied: rich<14,>=10.14.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (13.7.1)\n",
            "Requirement already satisfied: tenacity<9,>=8.1.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (8.5.0)\n",
            "Requirement already satisfied: toml<2,>=0.10.1 in /usr/local/lib/python3.10/dist-packages (from streamlit) (0.10.2)\n",
            "Requirement already satisfied: typing-extensions<5,>=4.3.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (4.12.2)\n",
            "Requirement already satisfied: gitpython!=3.1.19,<4,>=3.0.7 in /usr/local/lib/python3.10/dist-packages (from streamlit) (3.1.43)\n",
            "Requirement already satisfied: pydeck<1,>=0.8.0b4 in /usr/local/lib/python3.10/dist-packages (from streamlit) (0.9.1)\n",
            "Requirement already satisfied: tornado<7,>=6.0.3 in /usr/local/lib/python3.10/dist-packages (from streamlit) (6.3.3)\n",
            "Requirement already satisfied: watchdog<5,>=2.1.5 in /usr/local/lib/python3.10/dist-packages (from streamlit) (4.0.1)\n",
            "Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.10/dist-packages (from pandas) (2.8.2)\n",
            "Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas) (2023.4)\n",
            "Requirement already satisfied: tzdata>=2022.1 in /usr/local/lib/python3.10/dist-packages (from pandas) (2024.1)\n",
            "Requirement already satisfied: scipy in /usr/local/lib/python3.10/dist-packages (from lightgbm) (1.11.4)\n",
            "Requirement already satisfied: joblib>=1.1.1 in /usr/local/lib/python3.10/dist-packages (from scikit-learn) (1.4.2)\n",
            "Requirement already satisfied: threadpoolctl>=2.0.0 in /usr/local/lib/python3.10/dist-packages (from scikit-learn) (3.5.0)\n",
            "Requirement already satisfied: et-xmlfile in /usr/local/lib/python3.10/dist-packages (from openpyxl) (1.1.0)\n",
            "Requirement already satisfied: entrypoints in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->streamlit) (0.4)\n",
            "Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->streamlit) (3.1.4)\n",
            "Requirement already satisfied: jsonschema>=3.0 in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->streamlit) (4.19.2)\n",
            "Requirement already satisfied: toolz in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->streamlit) (0.12.1)\n",
            "Requirement already satisfied: gitdb<5,>=4.0.1 in /usr/local/lib/python3.10/dist-packages (from gitpython!=3.1.19,<4,>=3.0.7->streamlit) (4.0.11)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.8.2->pandas) (1.16.0)\n",
            "Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.27->streamlit) (3.3.2)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.27->streamlit) (3.7)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.27->streamlit) (2.0.7)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.27->streamlit) (2024.7.4)\n",
            "Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.10/dist-packages (from rich<14,>=10.14.0->streamlit) (3.0.0)\n",
            "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.10/dist-packages (from rich<14,>=10.14.0->streamlit) (2.16.1)\n",
            "Requirement already satisfied: smmap<6,>=3.0.1 in /usr/local/lib/python3.10/dist-packages (from gitdb<5,>=4.0.1->gitpython!=3.1.19,<4,>=3.0.7->streamlit) (5.0.1)\n",
            "Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->altair<6,>=4.0->streamlit) (2.1.5)\n",
            "Requirement already satisfied: attrs>=22.2.0 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (23.2.0)\n",
            "Requirement already satisfied: jsonschema-specifications>=2023.03.6 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (2023.12.1)\n",
            "Requirement already satisfied: referencing>=0.28.4 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.35.1)\n",
            "Requirement already satisfied: rpds-py>=0.7.1 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.19.0)\n",
            "Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.10/dist-packages (from markdown-it-py>=2.2.0->rich<14,>=10.14.0->streamlit) (0.1.2)\n"
          ]
        }
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
        "outputId": "e1104923-2f4a-4765-857a-4f191bd76c1f"
      },
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "2024-07-24 01:16:16.012 \n",
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