import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

# ---------- Configuração da página ----------
st.set_page_config(
    page_title="Previsão de Situação do Aluno",
    page_icon="🎓",
    layout="wide"
)

# ---------- CSS customizado ----------
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .header-gradient {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        padding: 2.2rem 1rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 14px rgba(79,70,229,0.25);
    }
    .header-gradient h1 {
        color: white;
        margin-bottom: 0.3rem;
    }
    .header-gradient p {
        color: #e0e7ff;
        font-size: 1rem;
        margin: 0;
    }
    .card {
        background-color: white;
        padding: 1.2rem;
        border-radius: 14px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        margin-bottom: 1rem;
    }
    div.stButton > button {
        width: 100%;
        background-color: #4f46e5;
        color: white;
        font-weight: 600;
        padding: 0.7rem;
        border-radius: 10px;
        border: none;
        transition: background-color 0.2s ease;
        font-size: 1rem;
    }
    div.stButton > button:hover {
        background-color: #4338ca;
        color: white;
    }
    .resultado-box {
        padding: 1.6rem;
        border-radius: 14px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    }
    .aprovado { background-color: #d1fae5; color: #065f46; }
    .recuperacao { background-color: #fef3c7; color: #92400e; }
    .reprovado { background-color: #fee2e2; color: #991b1b; }
    .history-item {
        padding: 0.6rem 1rem;
        border-radius: 8px;
        background-color: #f9fafb;
        border-left: 4px solid #4f46e5;
        margin-bottom: 0.4rem;
        font-size: 0.9rem;
    }
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

# ---------- Histórico na sessão ----------
if "historico" not in st.session_state:
    st.session_state.historico = []

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### ℹ️ Sobre o modelo")
    st.write("**Algoritmo:** Árvore de Decisão")
    st.write("**Biblioteca:** scikit-learn")
    st.write(f"**Amostras de treino:** {len(x_train)}")
    st.write(f"**Amostras de teste:** {len(x_teste)}")
    st.markdown("---")
    st.markdown("### 📊 Dados de treino")
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.markdown("---")
    if st.session_state.historico:
        st.markdown("### 🕓 Histórico de previsões")
        for item in reversed(st.session_state.historico[-5:]):
            st.markdown(f'<div class="history-item">{item}</div>', unsafe_allow_html=True)
        if st.button("🗑️ Limpar histórico"):
            st.session_state.historico = []
            st.rerun()

# ---------- Cabeçalho ----------
st.markdown("""
    <div class="header-gradient">
        <h1>🎓 Previsão de Situação do Aluno</h1>
        <p>Modelo de Machine Learning para prever aprovação com base em horas de estudo, faltas e nota</p>
    </div>
""", unsafe_allow_html=True)

# ---------- Abas ----------
aba_previsao, aba_arvore, aba_dados = st.tabs(["🔍 Previsão", "🌳 Árvore de Decisão", "📈 Análise dos Dados"])

# ---------- Aba de Previsão ----------
with aba_previsao:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Insira os dados do aluno")

    col1, col2, col3 = st.columns(3)
    with col1:
        horas = st.slider("📚 Horas de estudo por semana", 0.0, 20.0, 5.0, 0.5)
    with col2:
        faltas = st.slider("🚫 Número de faltas", 0.0, 30.0, 5.0, 1.0)
    with col3:
        nota = st.slider("📝 Nota atual", 0.0, 10.0, 6.0, 0.1)

    st.markdown('</div>', unsafe_allow_html=True)

    botao = st.button("🔍 Prever situação")

    if botao:
        df_novo = pd.DataFrame([[horas, faltas, nota]], columns=['Horas_de_estudo', 'Faltas', 'Nota'])
        previsao = modelo.predict(df_novo)[0]
        probabilidades = modelo.predict_proba(df_novo)[0]
        classes = modelo.classes_

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

        st.write("")
        st.markdown("**Confiança do modelo por classe:**")
        for classe_nome, prob in zip(classes, probabilidades):
            st.write(f"{classe_nome}")
            st.progress(float(prob))

        st.write("")
        m1, m2, m3 = st.columns(3)
        m1.metric("Horas de estudo", f"{horas:.1f}h")
        m2.metric("Faltas", f"{faltas:.0f}")
        m3.metric("Nota", f"{nota:.1f}")

        st.session_state.historico.append(
            f"{emoji} Horas: {horas:.0f} | Faltas: {faltas:.0f} | Nota: {nota:.1f} → {previsao}"
        )

# ---------- Aba da Árvore ----------
with aba_arvore:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Visualização da Árvore de Decisão")
    fig, ax = plt.subplots(figsize=(14, 8))
    plot_tree(
        modelo,
        feature_names=x.columns,
        class_names=modelo.classes_,
        filled=True,
        rounded=True,
        fontsize=10,
        ax=ax
    )
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Aba de Análise ----------
with aba_dados:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Distribuição das situações no dataset")
    contagem = df['Situacao'].value_counts()
    st.bar_chart(contagem)

    st.subheader("Relação Nota x Situação")
    st.scatter_chart(df, x="Nota", y="Faltas", color="Situacao")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Rodapé ----------
st.markdown("---")
st.caption("Modelo: Árvore de Decisão (scikit-learn) • Dados de treino ilustrativos • Feito com S
