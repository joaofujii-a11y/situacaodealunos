import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# ---------- Configuração da página ----------
st.set_page_config(
    page_title="Previsão de Situação do Aluno",
    page_icon="🎓",
    layout="centered"
)

# ---------- CSS customizado ----------
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .title-container {
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }
    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    div.stButton > button {
        width: 100%;
        background-color: #4f46e5;
        color: white;
        font-weight: 600;
        padding: 0.6rem;
        border-radius: 10px;
        border: none;
        transition: background-color 0.2s ease;
    }
    div.stButton > button:hover {
        background-color: #4338ca;
        color: white;
    }
    .resultado-box {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 700;
        margin-top: 1.5rem;
    }
    .aprovado { background-color: #d1fae5; color: #065f46; }
    .recuperacao { background-color: #fef3c7; color: #92400e; }
    .reprovado { background-color: #fee2e2; color: #991b1b; }
    </style>
""", unsafe_allow_html=True)

# ---------- Dados de treino ----------
dados = {'Horas_de_estudo': [10, 2, 5, 8, 1],
         'Faltas': [2, 15, 6, 1, 20],
         'Nota': [8.5, 3.0, 6.5, 9.0, 2.5],
         'Situacao': ['Aprovado', 'Reprovado', 'Recuperação', 'Aprovado', 'Reprovado']
         }
df = pd.DataFrame(dados)

x = df[['Horas_de_estudo', 'Faltas', 'Nota']]
y = df[['Situacao']]
x_train, x_teste, y_train, y_teste = train_test_split(x, y, test_size=0.2, random_state=42)

modelo = DecisionTreeClassifier(random_state=42)
modelo.fit(x_train, y_train)

# ---------- Cabeçalho ----------
st.markdown('<div class="title-container"><h1>🎓 Previsão de Situação do Aluno</h1></div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Insira os dados do aluno para prever se ele será aprovado, ficará em recuperação ou será reprovado.</div>', unsafe_allow_html=True)

# ---------- Entradas em colunas ----------
col1, col2, col3 = st.columns(3)

with col1:
    horas = st.number_input("📚 Horas de estudo", min_value=0.0, step=1.0)
with col2:
    faltas = st.number_input("🚫 Faltas", min_value=0.0, step=1.0)
with col3:
    nota = st.number_input("📝 Nota", min_value=0.0, max_value=10.0, step=0.1)

st.write("")

# ---------- Botão e resultado ----------
if st.button("🔍 Prever situação"):
    df_novo = pd.DataFrame([[horas, faltas, nota]], columns=['Horas_de_estudo', 'Faltas', 'Nota'])
    previsao = modelo.predict(df_novo)[0]

    if previsao == "Aprovado":
        emoji, classe = "✅", "aprovado"
    elif previsao == "Recuperação":
        emoji, classe = "⚠️", "recuperacao"
    else:
        emoji, classe = "❌", "reprovado"

    st.markdown(
        f'<div class="resultado-box {classe}">{emoji} O sistema previu: {previsao}</div>',
        unsafe_allow_html=True
    )

    # Mini resumo dos dados inseridos
    st.write("")
    m1, m2, m3 = st.columns(3)
    m1.metric("Horas de estudo", f"{horas:.0f}h")
    m2.metric("Faltas", f"{faltas:.0f}")
    m3.metric("Nota", f"{nota:.1f}")

# ---------- Rodapé ----------
st.markdown("---")
st.caption("Modelo: Árvore de Decisão (scikit-learn) • Dados de treino ilustrativos")
