# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 19:23:52 2020

@author: Raul
"""
#===============================================================================
#   EEL817 (Redes Neurais)
#   LISTA DE EXERCÍCIOS - Exercício 6 - MLP para a base de dados "Lista1.csv"
#   ALUNO: RAUL BAPTISTA DE SOUZA
#   DRE: 115 110 845
#   Script1: Exercicio_6_sem_cross_validation (Exercicio_6_script1.py) 
#       O primeiro script  dividi os dados em treino e teste (70%-30%)
#   Script2: Exercicio_6_com_cross_validation (Exercicio_6_script2.py)
#       Esse script efetua validação cruzada.
#===============================================================================

#-------------------------------------------------------------------------------
# Importar bibliotecas
#-------------------------------------------------------------------------------
#%%
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
tf.__version__
# MLP for with n-fold cross validation:
from sklearn.model_selection import StratifiedKFold
# Semente é o meu DRE: 115 110 845
seed = 115110845
# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.
np.random.seed(seed)
# The below set_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see:
# https://www.tensorflow.org/api_docs/python/tf/random/set_seed
tf.random.set_seed(seed)
#%%
#-------------------------------------------------------------------------------
# Ler o arquivo CSV com os dados do conjunto aprovação de crédito
#-------------------------------------------------------------------------------
dados_iniciais = pd.read_csv('Lista1.csv', delimiter = ';', decimal = '.') 
dados = dados_iniciais
#-------------------------------------------------------------------------------
# Verificar se há dados ausentes (null) em alguma coluna
#-------------------------------------------------------------------------------
print("-------------------------------------------------")
print("---Verificando se há dados ausentes nas colunas--")
print(dados_iniciais.isna().sum())
print("-------------------------------------------------")
#-------------------------------------------------------------------------------
# Executar o pré-processamento dos dados
#-------------------------------------------------------------------------------
# PreviousSurgery   ---> Descartar  (muitos nan / coluna inteira com 0s)
# Inheritance       ---> Descartar  (muitos nan)
print('Descartar atributos que possuam erros/incosistências/alta cardinalidade')
print("-------------------------------------------------")
dados = dados.drop(['PreviousSurgery', 'Inheritance'],axis=1)
#-------------------------------------------------------------------------------
# Selecionar os atributos que serão utilizados pelo classificador
#-------------------------------------------------------------------------------
print ( 'Verificar o valor médio de cada atributo em cada classe:')
print(dados_iniciais.groupby(['Outcome']).mean().T)
print("-------------------------------------------------")
atributos = [  
# 'Patient',
 'Age',
  'HealthFactor1',
# 'HealthFactor2',
 'BMI',
 'DiabetesPedigreeFunction',
# 'Pregnancies',
 'Glucose',
 'BloodPressure',
 'SkinThickness',
 'Insulin',
 'Outcome'
 ]
dados = dados[atributos]
#-------------------------------------------------------------------------------
# Criar os arrays X e Y separando atributos e alvo
#-------------------------------------------------------------------------------
X = dados.iloc[:, :-1].values
y = dados.iloc[:,-1].values  
#-------------------------------------------------------------------------------
# Tratar dados perdidos(nan): Apenas da coluna HealthFactor2 (possui poucos nan)
#-------------------------------------------------------------------------------
# imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
# imputer.fit(X[:,2:3])
# X[:,2:3] = imputer.transform(X[:,2:3])
#%%
#-------------------------------------------------------------------------------
# Feature Scaling
#-------------------------------------------------------------------------------
sc = StandardScaler()
X = sc.fit_transform(X)
#%%
#-------------------------------------------------------------------------------
# Building the ANN - Initializing the ANN
#-------------------------------------------------------------------------------
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []
for treino, teste in kfold.split(X, y):
  ann = tf.keras.models.Sequential()
  ann.add(tf.keras.layers.Dense(units=5, activation='relu'))
  ann.add(tf.keras.layers.Dense(units=5, activation='relu'))
  ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
  ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
  ann.fit(X, y, batch_size = 32, epochs = 100, verbose=0)
  scores = ann.evaluate(X, y, verbose=0)
  print("%s: %.2f%%" % (ann.metrics_names[1], scores[1]*100))
  cvscores.append(scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
