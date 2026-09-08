import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Dados de treino
dados = {'Horas_de_estudo': [10, 2, 5, 8, 1],
         'Faltas': [2, 15, 6, 1, 20],
         'Nota': [8.5, 3.0, 6.5, 9.0, 2.5],
         'Situacao': ['Aprovado', 'Reprovado', 'Recuperação', 'Aprovado', 'Reprovado']
         }
df = pd.DataFrame(dados)

# Separação treino/teste
x = df[['Horas_de_estudo', 'Faltas', 'Nota']]
y = df[['Situacao']]
x_train, x_teste, y_train, y_teste = train_test_split(x, y, test_size=0.2, random_state=42)

# Treinamento do modelo
modelo = DecisionTreeClassifier(random_state=42)
modelo.fit(x_train, y_train)

# Interface Streamlit
st.title("Previsão de Situação do Aluno")

horas = st.number_input("Horas de estudo", min_value=0.0, step=1.0)
faltas = st.number_input("Faltas", min_value=0.0, step=1.0)
nota = st.number_input("Nota", min_value=0.0, max_value=10.0, step=0.1)

if st.button("Prever"):
    df_novo = pd.DataFrame([[horas, faltas, nota]], columns=['Horas_de_estudo', 'Faltas', 'Nota'])
    previsao = modelo.predict(df_novo)[0]
    st.success(f"O sistema previu: {previsao}")
