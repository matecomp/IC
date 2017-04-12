#!/usr/bin/env python
# -*- coding: utf-8 -*-

#Função que cria o vocabulário a partir do arquivo prolog
def create_vocab():
	print('Criando o vocabulario...')
	# 1º Passo: abrir o arquivo prolog com o BD
	archive = "JC_model/predicado-ent1-ent2.pl"
	with open(archive, 'r') as f:
	    read_data = f.read()

	# 2º Passo: extraia o vocabulário deste BD
	import re
	words = re.findall(u'"[a-zA-Z0-9]+"', read_data)
	#remova repetições: De 110861 para 2472 palavras
	vocabulary = sorted(list(set(words)))
	print("Numero de termos:")
	print("Total:",len(words))
	print("Unicos:",len(vocabulary))

	# 3º Passo: criar um arquivo com o vocabulário
	archive = "vocabulary.txt"
	with open(archive, 'r+') as f:
		for word in vocabulary:
			f.write(word+'\n')
	print('Vocabulario criado...')


#Função que lê o arquivo gerado na criação do vocabulário
#retornando uma listas com os termos existentes no arquivo prolog
def get_vocabulary(archive):
	# 1º Passo: abrir o arquivo
	with open(archive, 'r') as f:
	    read_data = f.read().split("\n")[:-1]
	return read_data

# OBS: Este vocabulário está salvo com as aspas no nome...
# exemplo: "nome"

# Para usar essa função num outro código, basta importar da seguinte forma:
# from [nome_do_arquivo] import [nome_da_função]

# Exemplo:
# from vocab import get_vocabulary
# variavel = get_vocabulary()

import numpy as np
def get_wordspace():
	return np.loadtxt('WE_model/WE.out')

from TSNE import *

def _start_shell(local_ns=None):
  # An interactive shell is useful for debugging/development.
  import IPython
  user_ns = {}
  if local_ns:
    user_ns.update(local_ns)
  user_ns.update(globals())
  IPython.start_ipython(argv=[], user_ns=user_ns)


if __name__ == "__main__":
	# Pegar a lista de termos do text8 e do arquivo Prolog
	JC_vocab = get_vocabulary("JC_model/vocabulary.txt")
	WE_vocab = get_vocabulary("WE_model/vocabulary.txt")

	# Pegar a matriz de pesos treinada com o text8
	WE = get_wordspace()
	
	# Reduzir a dimensão com o TSNE para 2D e criar uma imagem
	# com uma amostra de termos
	# saveTSNE(WE, WE_vocab)

	# Checar os tamanhos para ver se está correto
	print("Prolog termos: ", len(JC_vocab))
	print("Text8 termos: ", len(WE_vocab))
	print("Tamanho da matriz: ", WE.shape)

	# Termos de JC_vocab que não existem em WE_vocab
	setJC = set(JC_vocab)
	setWE = set(WE_vocab)
	uniqueJC = setJC.difference(setWE)
	print("Diff (JC - WE): ",len(uniqueJC))

	# Termos de JC_vocab que não existem em WE_vocab
	sharedT = setJC.intersection(setWE)
	print("Intersseccao JC ^ WE ",len(sharedT))

	_start_shell(locals())