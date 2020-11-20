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
#       O primeiro script dividi os dados em treino e teste (70%-30%)
#   Script2: Exercicio_6_com_cross_validation (Exercicio_6_script2.py)
#       Esse script efetua validação cruzada.
#===============================================================================

#-------------------------------------------------------------------------------
# Importar bibliotecas
#-------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.metrics import confusion_matrix, accuracy_score
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
# Embaralhar o conjunto de dados para garantir que a divisão entre os dados de
# treino e os dados de teste esteja isenta de qualquer viés de seleção.
# E Separar X e Y em conjunto de treino e conjunto de teste
# Usar meu DRE = 115110845 como semente na geração de números aleatórios
# Usar 70% das amostras para treino e 30% para teste.
#-------------------------------------------------------------------------------
from sklearn.model_selection import train_test_split
x_treino, x_teste, y_treino, y_teste = train_test_split(X, y, test_size = 0.3, random_state = seed)
#%%
#-------------------------------------------------------------------------------
# Feature Scaling
#-------------------------------------------------------------------------------
sc = StandardScaler()
x_treino = sc.fit_transform(x_treino)
x_teste = sc.transform(x_teste)
#%%
#-------------------------------------------------------------------------------
# Building the ANN - Initializing the ANN
#-------------------------------------------------------------------------------
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []
for treino, teste in kfold.split(x_treino, y_treino):
  modelo = tf.keras.models.Sequential()
  modelo.add(tf.keras.layers.Dense(units=5, activation='relu'))
  modelo.add(tf.keras.layers.Dense(units=5, activation='relu'))
  modelo.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
  modelo.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
  modelo.fit(x_treino, y_treino, batch_size = 32, epochs = 100, verbose=0)
  scores = modelo.evaluate(x_teste, y_teste, verbose=0)
  print("%s: %.2f%%" % (modelo.metrics_names[1], scores[1]*100))
  cvscores.append(scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

#%%
# Making the predictions and evaluating the model
#-------------------------------------------------------------------------------
y_pred = modelo.predict(x_teste)
y_pred = (y_pred > 0.5)
# print(np.concatenate((y_pred.reshape(len(y_pred),1), y_teste.reshape(len(y_teste),1)),1))
#-------------------------------------------------------------------------------
# Making the Confusion Matrix
#-------------------------------------------------------------------------------
cm = confusion_matrix(y_teste, y_pred)
print('Matriz de confusão:')
print(cm)
print('Acurácia:')
print(accuracy_score(y_teste, y_pred))
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
  

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

plot_confusion_matrix(y_teste, y_pred, classes=[1, 0], title='Confusion matrix')
# plt.savefig("ConfusionMatrix.png")
#-------------------------------------------------------------------------------
# ROC curve
#-------------------------------------------------------------------------------
y_true = y_teste
y_probas = np.concatenate((1-y_pred,y_pred),axis=1)
skplt.metrics.plot_roc(y_true,y_probas)
plt.show()