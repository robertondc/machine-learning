from collections import Counter
import pandas as pd

df = pd.read_csv('busca.csv')

X_df = df[['home', 'busca', 'logado']]
Y_df = df['comprou']

Xdummies_df = pd.get_dummies(X_df)
Ydummies_df = Y_df

X = Xdummies_df.values
Y = Ydummies_df.values

porcentagem_de_treino = 0.8
porcetagem_de_teste = 0.1

tamanho_de_treino = porcentagem_de_treino * len(Y)
tamanho_de_teste = porcetagem_de_teste * len(Y)

treino_dados = X[0:tamanho_de_treino]
treino_marcadores = Y[0:tamanho_de_treino]

fim_do_teste = tamanho_de_treino + tamanho_de_teste
teste_dados = X[tamanho_de_treino:fim_do_teste]
teste_marcadores = Y[tamanho_de_treino:fim_do_teste]

validacao_dados = X[fim_do_teste:]
validacao_marcadores = Y[fim_do_teste:]


def taxa_de_acerto(marcadores_resultado, marcadores_esperados):
	acertos = (marcadores_resultado == marcadores_esperados)
	total_de_acertos = sum(acertos) #true e false somam como 0 e 1
	total_de_elementos = len(marcadores_esperados)
	taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos
	return taxa_de_acerto


def fit_and_predict(nome,modelo, treino_dados, treino_marcadores, teste_dados, teste_marcadores):
	modelo.fit(treino_dados,treino_marcadores)

	resultado = modelo.predict(teste_dados)

	msg = "Taxa de acerto do {0}: {1}".format(nome, taxa_de_acerto(resultado, teste_marcadores))
	print(msg)
	return taxa_de_acerto

from sklearn.naive_bayes import MultinomialNB
modelo_multinomial = MultinomialNB()
resultado_multinomial = fit_and_predict("MultinomialNB", modelo_multinomial, treino_dados, treino_marcadores, teste_dados, teste_marcadores)

from sklearn.ensemble import AdaBoostClassifier
modelo_adaboost = AdaBoostClassifier()
resultado_adaboost = fit_and_predict("AdaBoostClassifier", modelo_adaboost, treino_dados, treino_marcadores, teste_dados, teste_marcadores)

if resultado_multinomial > resultado_adaboost:
	vencedor = modelo_multinomial
else:
	vencedor = modelo_adaboost

resultado = vencedor.predict(validacao_dados)

msg = "Taxa de acerto do vencedor entre os dois algoritmos no mundo real: {0}".format(taxa_de_acerto(resultado, validacao_marcadores))
print(msg)
#eficacia do algoritimo via chute

acerto_base = max(Counter(teste_marcadores).itervalues())
taxa_de_acerto_base = 100.0 * acerto_base / len(teste_dados)
print("Taxa de acerto base: %f" % taxa_de_acerto_base)
print("Total de testes: %d" % len(teste_dados))

