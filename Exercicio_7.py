# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 17:39:14 2020

@author: Raul
"""
#===============================================================================
#   EEL817 (Redes Neurais)
#   LISTA DE EXERCÍCIOS - Exercício 7
#   ALUNO: RAUL BAPTISTA DE SOUZA
#   DRE: 115 110 845
#===============================================================================
#%%
# Imports
import random
import numpy as np
# Semente é o meu DRE: 115 110 845
seed = 115110845
# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.
np.random.seed(seed)
#%%
#-------------------------------------------------------------------------------
# Entradas (parâmeros): layer_sizes, activation_functions, model_question_1,
# model_question_2.
#-------------------------------------------------------------------------------
#       A lista 'layer_sizes' contém o número de neurônios nas respectivas 
#       camadas da rede. Por exemplo, se a lista  for [2, 2, 3, 2] então 
#       será uma rede de quatro camadas, com a primeira camada contendo 
#       2 neurônios, a segunda camada 2 neurônios, a terceira camada
#       1 neurônio e a última camada com 2 neurônios.
#           A lista 'activation_functions' contém o número que representa a
#       função de ativação nas respectivas camadas, onde 1 é o ReLU, 2 é
#       a LeakyReLU, 3 a tanh e 4 a sigmoid. Por exemplo, se a lista for
#       [0, 4, 3, 1], a camada inicial, não possui, a camada 2 terá função
#       de ativação sigmoid, a camada 3 função de ativação tanh e a camada
#       final terá função de ativação ReLU.
#       
#       Se model_question1 == True os parâmetros da questão 1 serão selecionados
#       Se model_question2 == True os parâmetros da questão 2 serão selecionados
#       Se ambos forem falsos, será selecionado pesos, bias e entrada aleatórios,
#       Se ambos forem True, será selecionado parâmetros da questão 2.
layer_sizes = [2,2,3,2]
activation_functions = [0,4,3,1]
model_question_1 = True
model_question_2 = False
#-------------------------------------------------------------------------------
#%%
input_layer = [layer_sizes[0]+1]
input_layer = [np.random.randn(y, 1) for y in input_layer[:]]
if(model_question_1 == True):
    input_layer[0][0][0] = 1
    input_layer[0][1][0] = -0.3
    input_layer[0][2][0] = 0.5
    activation_functions = [0,1,1,1]
elif(model_question_2 == True):
    input_layer[0][0][0] = 1
    input_layer[0][1][0] = -0.3
    input_layer[0][2][0] = 0.5
W = []
Z = []
Z.append(input_layer[0].T)
#-----------------------------------------------------------------------------
# FUNÇÕES DE ATIVAÇÃO E SUAS DERIVADAS
#-----------------------------------------------------------------------------
# Função de Ativação ReLU
def ReLU(x):
    return max(0,x)
# Função para retornar as derivadas da função ReLU
def ReLU_prime(x):
    if x > 0:
        return 1    
    else:
        return 0 
# Função de Ativação LeakyReLU
def LeakyReLU(x, a):
    return max(a*x,x)
# Função para retornar as derivadas da função LeakyReLU
def LeakyReLU_prime(x, a):
    if x > 0:
        return 1    
    else:
        return a
# Função de Ativação TanH
def tanH(x, a):
    return 2*sigmoid(2*x, a) - 1
# Função para retornar as derivadas da função tanH
def tanH_prime(x, a):
    return 1 - (tanH(x, a))**2
# Função de Ativação Sigmóide
def sigmoid(x, a):
    return 1.0/(1.0 + np.exp(-a*x))
# Função para retornar as derivadas da função Sigmóide
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))
def question_7(layer_sizes, activation_functions):
    # Os bias e pesos para a rede são inicializados aleatoriamente, usando 
    # uma distribuição Gaussiana com média 0 e variância 1.
    biases = [np.random.randn(y, 1) for y in layer_sizes[1:]]
    weights = [np.random.randn(y, x) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]
    # Montar as matrizes de peso de cada Camada
    for i in range(0,len(layer_sizes)-1):
        a = np.array(biases[i])
        b = np.array(weights[i])
        c = [np.concatenate((a,b), axis = 1)]
        W.append(c[0])
############################################################################
    # TESTE PROPAGAÇÃO DE ENTRADA QUESTÃO 1 / QUESTÃO 2 
    # neurônio 1
    if (model_question_1 or model_question_2 == True):
        W[0][0][0] = -0.3
        W[0][0][1] = -0.1
        W[0][0][2] = 0.0
        # neurônio 2
        W[0][1][0] = 0.2
        W[0][1][1] = 0.1
        W[0][1][2] = 0.4
        # neurônio 3
        W[1][0][0] = 0.3
        W[1][0][1] = 0.0
        W[1][0][2] = -0.1
        # neurônio 4
        W[1][1][0] = 0.4
        W[1][1][1] = 0.0
        W[1][1][2] = 0.4
        # neurônio 5
        W[1][2][0] = 0.1
        W[1][2][1] = 0.3
        W[1][2][2] = 0.0
        # neurônio 6
        W[2][0][0] = 0.3
        W[2][0][1] = -0.1
        W[2][0][2] = 0.5
        W[2][0][3] = -0.1
        # neurônio 7
        W[2][1][0] = 0.4
        W[2][1][1] = 0.3
        W[2][1][2] = -0.2
        W[2][1][3] = 0.4
##########################################################################
    # Propagação de Entrada:
    a = 0
    for i in range(len(layer_sizes)-1):
        if i != len(layer_sizes)-2:
            z = np.dot(Z[i],W[i].T)
            # Acrescentar função de ativação aqui
            for j in range(len(z[0])):
                a = a + 1 
                if (activation_functions[i+1]==1):
                    z[0][j] = ReLU(z[0][j]) 
                elif (activation_functions[i+1]==2):
                    z[0][j] = LeakyReLU(z[0][j], a) 
                elif (activation_functions[i+1]==3):
                    z[0][j] = tanH(z[0][j], a) 
                elif (activation_functions[i+1]==4):
                    z[0][j] = sigmoid(z[0][j], a) 
                else:
                    pass        
            z = np.append(1, np.array(z))
            z.shape = (z.size,1)
            z = z.T
            Z.append(np.array(z))
        else:
            z = np.dot(Z[i],W[i].T)
            for j in range(len(z[0])):
                a = a + 1 
                if (activation_functions[i+1]==1):
                    z[0][j] = ReLU(z[0][j]) 
                elif (activation_functions[i+1]==2):
                    z[0][j] = LeakyReLU(z[0][j], a) 
                elif (activation_functions[i+1]==3):
                    z[0][j] = tanH(z[0][j], a) 
                elif (activation_functions[i+1]==4):
                    z[0][j] = sigmoid(z[0][j], a) 
                else:
                    pass
            z.shape = (z.size,1)
            z = z.T
            Z.append(np.array(z))
    return Z


question_7(layer_sizes, activation_functions)
