import pandas as p

dados = {'Horas_de_estudo': [10, 2, 5, 8, 1],
         'Faltas': [2, 15, 6, 1, 20],
         'Nota': [8.5, 3.0, 6.5, 9.0, 2.5],
         'Situacao': ['Aprovado','Reprovado','Recuperação','Aprovado','Reprovado']
         }
df = pd.DataFrame(dados)
df

from sklearn.model_selection import train_test_split
x = df[['Horas_de_estudo','Faltas','Nota']]
y = df[['Situacao']]
x_train, x_teste, y_train, y_teste = train_test_split(x, y, test_size=0.2, random_state=42)



from sklearn.tree import DecisionTreeClassifier
modelo = DecisionTreeClassifier(random_state=42)
modelo.fit(x_train, y_train)


novo_aluno = pd.DataFrame([[6,2,7.0]],columns=['Horas_de_estudo', 'Faltas', 'Nota'])
previsao = modelo.predict(novo_aluno)
print(f"O sistema previu: {previsao[0]}")



from sklearn.tree import plot_tree
plot_tree(modelo, feature_names=x.columns, class_names=modelo.classes_, filled=True)



import gradio as gr
def prever_situacao(horas, faltas, notas):
  df_novo = pd.DataFrame([[horas, faltas, notas]], columns=['Horas_de_estudo','Faltas','Nota'])
  return modelo.predict(df_novo)[0]

interface = gr.Interface(
    fn = prever_situacao, inputs = ["number","number","number"],outputs = "text"
)
interface.launch()
