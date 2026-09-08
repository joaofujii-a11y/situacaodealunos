import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from datetime import datetime

# ---------- Configuracao da pagina ----------
st.set_page_config(
    page_title="Previsao de Situacao do Aluno",
    page_icon="🎓",
    layout="wide"
)

# ---------- CSS customizado ----------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    .main {
        background-color: #f5f7fa;
    }

    .header-gradient {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #ec4899 100%);
        background-size: 200% 200%;
        animation: gradientMove 8s ease infinite;
        padding: 2.4rem 1rem;
        border-radius: 18px;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 6px 18px rgba(79,70,229,0.3);
    }

    @keyframes gradientMove {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .header-gradient h1 {
        color: white;
        margin-bottom: 0.3rem;
        font-weight: 700;
    }

    .header-gradient p {
        color: #ede9fe;
        font-size: 1.05rem;
        margin: 0;
    }

    .card {
        background-color: white;
        padding: 1.4rem;
        border-radius: 16px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        margin-bottom: 1.2rem;
        border: 1px solid #eef0f4;
    }

    .card h3, .card h4 {
        margin-top: 0;
    }

    div.stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        color: white;
        font-weight: 600;
        padding: 0.75rem;
        border-radius: 12px;
        border: none;
        font-size: 1rem;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }

    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 14px rgba(79,70,229,0.35);
        color: white;
    }

    .resultado-box {
        padding: 1.8rem;
        border-radius: 16px;
        text-align: center;
        font-size: 1.6rem;
        font-weight: 700;
        margin-top: 1rem;
        box-shadow: 0 4px 14px rgba(0,0,0,0.08);
        border: 2px solid rgba(0,0,0,0.03);
    }

    .aprovado { background-color: #d1fae5; color: #065f46; }
    .recuperacao { background-color: #fef3c7; color: #92400e; }
    .reprovado { background-color: #fee2e2; color: #991b1b; }

    .history-item {
        padding: 0.6rem 1rem;
        border-radius: 10px;
        background-color: #f9fafb;
        border-left: 4px solid #4f46e5;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
    }

    .badge {
        display: inline-block;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        background-color: #eef2ff;
        color: #4338ca;
        margin-right: 0.4rem;
    }

    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ---------- Dados de treino ----------
dados = {'Horas_de_estudo': [10, 2, 5, 8, 1],
         'Faltas': [2, 15, 6, 1, 20],
         'Nota': [8.5, 3.0, 6.5, 9.0, 2.5],
         'Situacao': ['Aprovado', 'Reprovado', 'Recuperacao', 'Aprovado', 'Reprovado']
         }
df = pd.DataFrame(dados)

x = df[['Horas_de_estudo', 'Faltas', 'Nota']]
y = df[['Situacao']]
x_train, x_teste, y_train, y_teste = train_test_split(x, y, test_size=0.2, random_state=42)

modelo = DecisionTreeClassifier(random_state=42)
modelo.fit(x_train, y_train)

# ---------- Estado da sessao ----------
if "historico" not in st.session_state:
    st.session_state.historico = []

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### ℹ️ Sobre o modelo")
    st.markdown(
        '<span class="badge">Arvore de Decisao</span>'
        '<span class="badge">scikit-learn</span>',
        unsafe_allow_html=True
    )
    st.write(f"**Amostras de treino:** {len(x_train)}")
    st.write(f"**Amostras de teste:** {len(x_teste)}")
    st.write(f"**Profundidade da arvore:** {modelo.get_depth()}")
    st.write(f"**Numero de folhas:** {modelo.get_n_leaves()}")

    st.markdown("---")
    st.markdown("### 📊 Dados de treino")
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("---")
    if st.session_state.historico:
        st.markdown("### 🕓 Historico de previsoes")
        for item in reversed(st.session_state.historico[-5:]):
            st.markdown(f'<div class="history-item">{item}</div>', unsafe_allow_html=True)

        historico_df = pd.DataFrame(st.session_state.historico, columns=["Registro"])
        csv = historico_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Baixar historico (CSV)",
            data=csv,
            file_name="historico_previsoes.csv",
            mime="text/csv"
        )

        if st.button("🗑️ Limpar historico"):
            st.session_state.historico = []
            st.rerun()
    else:
        st.info("Nenhuma previsao feita ainda.")

# ---------- Cabecalho ----------
st.markdown("""
    <div class="header-gradient">
        <h1>🎓 Previsao de Situacao do Aluno</h1>
        <p>Modelo de Machine Learning para prever aprovacao com base em horas de estudo, faltas e nota</p>
    </div>
""", unsafe_allow_html=True)

# ---------- Abas ----------
aba_previsao, aba_arvore, aba_dados = st.tabs(["🔍  Previsao", "🌳  Arvore de Decisao", "📈  Analise dos Dados"])

# ---------- Aba de Previsao ----------
with aba_previsao:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Insira os dados do aluno")

    col1, col2, col3 = st.columns(3)
    with col1:
        horas = st.slider("📚 Horas de estudo por semana", 0.0, 20.0, 5.0, 0.5)
    with col2:
        faltas = st.slider("🚫 Numero de faltas", 0.0, 30.0, 5.0, 1.0)
    with col3:
        nota = st.slider("📝 Nota atual", 0.0, 10.0, 6.0, 0.1)

    st.markdown("</div>", unsafe_allow_html=True)

    botao = st.button("🔍 Prever situacao")

    if botao:
        df_novo = pd.DataFrame([[horas, faltas, nota]], columns=['Horas_de_estudo', 'Faltas', 'Nota'])
        previsao = modelo.predict(df_novo)[0]
        probabilidades = modelo.predict_proba(df_novo)[0]
        classes = modelo.classes_

        if previsao == "Aprovado":
            emoji, classe = "✅", "aprovado"
        elif previsao == "Recuperacao":
            emoji, classe = "⚠️", "recuperacao"
        else:
            emoji, classe = "❌", "reprovado"

        st.markdown(
            f'<div class="resultado-box {classe}">{emoji} O sistema previu: {previsao}</div>',
            unsafe_allow_html=True
        )

        st.write("")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Confianca do modelo por classe**")
        for classe_nome, prob in zip(classes, probabilidades):
            st.write(f"{classe_nome} — {prob*100:.1f}%")
            st.progress(float(prob))
        st.markdown("</div>", unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Horas de estudo", f"{horas:.1f}h")
        m2.metric("Faltas", f"{faltas:.0f}")
        m3.metric("Nota", f"{nota:.1f}")

        registro = (
            f"{emoji} {datetime.now().strftime('%H:%M:%S')} - "
            f"Horas: {horas:.0f}, Faltas: {faltas:.0f}, Nota: {nota:.1f} -> {previsao}"
        )
        st.session_state.historico.append(registro)

# ---------- Aba da Arvore ----------
with aba_arvore:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Visualizacao da Arvore de Decisao")
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
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Aba de Analise ----------
with aba_dados:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Distribuicao das situacoes")
        contagem = df['Situacao'].value_counts()
        st.bar_chart(contagem)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Nota x Faltas por situacao")
        st.scatter_chart(df, x="Nota", y="Faltas", color="Situacao")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Tabela completa de dados")
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Rodape ----------
st.markdown("---")
f1, f2, f3 = st.columns(3)
f1.caption("Modelo: Arvore de Decisao (scikit-learn)")
f2.caption("Dados de treino ilustrativos")
f3.caption("Feito com Streamlit")
