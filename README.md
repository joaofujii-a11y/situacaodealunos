# 🎓 Previsor de Situação Escolar

Projeto de Machine Learning que utiliza uma **Árvore de Decisão** para prever a situação de um aluno (**Aprovado**, **Recuperação** ou **Reprovado**) com base em três variáveis: horas de estudo semanais, número de faltas e nota atual. Inclui uma interface interativa construída com **Gradio**.

## 📋 Sobre o projeto

O modelo é treinado com um dataset sintético balanceado (70 exemplos por classe) e combina:

- Um classificador de árvore de decisão (`scikit-learn`) treinado a partir dos padrões dos dados;
- Uma **regra de negócio explícita** que sobrepõe o modelo: se a frequência de faltas ultrapassar 75% do limite, o aluno é reprovado automaticamente, independentemente da nota — replicando a regra de frequência mínima comum em escolas brasileiras.

## ✨ Funcionalidades

- Geração de dataset sintético balanceado entre as três classes
- Treinamento de árvore de decisão com hiperparâmetros ajustados contra overfitting (`max_depth`, `min_samples_split`, `min_samples_leaf`)
- Avaliação do modelo: acurácia, relatório de classificação, matriz de confusão e validação cruzada (5-fold)
- Visualização gráfica da árvore de decisão (exportada em PNG)
- Interface web interativa (Gradio) com sliders, exemplos rápidos, barra de frequência e exibição da probabilidade por categoria

## 🛠️ Tecnologias

- Python 3
- pandas / numpy
- scikit-learn
- matplotlib / seaborn
- Gradio

## 📦 Instalação

```bash
pip install pandas numpy scikit-learn matplotlib seaborn gradio
```

## 🚀 Como executar

1. Abra o notebook (Google Colab ou Jupyter) ou o script `.py` do projeto.
2. Execute as células/seções na ordem:
   - Geração do dataset
   - Treinamento do modelo
   - Avaliação (métricas e matriz de confusão)
   - Visualização da árvore de decisão
   - Lançamento da interface Gradio
3. A interface abrirá em uma URL local (e, no Colab, também gera um link público temporário).

## 📊 Variáveis utilizadas

| Variável | Descrição | Faixa |
|---|---|---|
| `Horas_de_estudo` | Horas de estudo por semana | 0 – 12 |
| `Faltas` | Número de faltas no período | 0 – 25 |
| `Nota` | Nota atual do aluno | 0 – 10 |
| `Situacao` | Classe alvo prevista | Aprovado / Recuperação / Reprovado |

## ⚖️ Regra de reprovação por falta

Se `Faltas / 25 >= 75%`, o aluno é classificado como **Reprovado** automaticamente, sem passar pelo modelo — isso evita que uma nota alta "mascare" uma frequência insuficiente.

## 📈 Avaliação do modelo

O modelo é validado com:
- **Divisão treino/teste estratificada** (80/20), mantendo a proporção das classes
- **Validação cruzada** (5-fold) para uma estimativa mais robusta de desempenho
- **Matriz de confusão** e **classification report** (precisão, recall, F1-score) por classe

## 🖥️ Interface

A interface Gradio permite ajustar horas de estudo, faltas e nota por meio de sliders, exibindo:
- A situação prevista, com cor indicativa (verde/laranja/vermelho)
- O motivo da previsão (regra de frequência ou modelo)
- Uma barra visual de frequência
- A probabilidade estimada para cada uma das três categorias
- Exemplos prontos para teste rápido

## ⚠️ Observações

- O dataset é **sintético**, gerado artificialmente para fins didáticos — não reflete dados reais de alunos.
- Os limites (`FALTAS_MAXIMAS = 25`, `LIMITE_FREQUENCIA = 75%`, `NOTA_MAXIMA = 10`) são parametrizáveis no início do script.
