from collections import Counter
import pandas as pd

df = pd.read_csv('busca2.csv')

X_df = df[['home', 'busca', 'logado']]
Y_df = df['comprou']

Xdummies_df = pd.get_dummies(X_df)
Ydummies_df = Y_df

X = Xdummies_df.values
Y = Ydummies_df.values

porcentagem_de_treino = 0.9

tamanho_de_treino = porcentagem_de_treino * len(Y)
tamanho_de_teste = len(Y) - tamanho_de_treino

treino_dados = X[:tamanho_de_treino]
treino_marcadores = Y[:tamanho_de_treino]

teste_dados = X[-tamanho_de_teste:]
teste_marcadores = Y[-tamanho_de_teste:]

def fit_and_predict(nome,modelo, treino_dados, treino_marcadores, teste_dados, teste_marcadores):
	modelo.fit(treino_dados,treino_marcadores)

	resultado = modelo.predict(teste_dados)
	acertos = (resultado == teste_marcadores)

	total_de_acertos = sum(acertos) #true e false somam como 0 e 1
	total_de_elementos = len(teste_dados)
	taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos
	
	msg = "Taxa de acerto do {0}: {1}".format(nome, taxa_de_acerto)	
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


#eficacia do algoritimo via chute

acerto_base = max(Counter(teste_marcadores).itervalues())
taxa_de_acerto_base = 100.0 * acerto_base / len(teste_dados)
print("Taxa de acerto base: %f" % taxa_de_acerto_base)
print("Total de testes: %d" % len(teste_dados))

