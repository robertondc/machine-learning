import csv

def carregar_acessos():
	X = []
	Y = []

	arquivo = open('acesso.csv','rb')
	leitor = csv.reader(arquivo)

	leitor.next() #pula header

	for home,funciona,contato,comprou in leitor:
		dado = [int(home),int(funciona),int(contato)]
		X.append(dado)
		Y.append(int(comprou))

	return X, Y