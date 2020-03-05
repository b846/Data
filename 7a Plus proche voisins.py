#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 19:34:49 2020

@author: b
https://medium.com/@kenzaharifi/bien-comprendre-lalgorithme-des-k-plus-proches-voisins-fonctionnement-et-impl%C3%A9mentation-sur-r-et-a66d2d372679

#Algorithme des plus proches voisins
Méthode d'apprentissage supervisé utilisé pour les cas de régression et de classification
méthode non paramétrique dans laquelle le modèle mémorise les observations de l’ensemble d’apprentissage pour 
la classification des données de l’ensemble de test.

L'algorithme est qualifiée de paressuex (Lazy Learning), car il n'apprend rien pendant la phase d'entrainement.
Pour prédire la classe d’une nouvelle donnée d’entrée, il va chercher ses K voisins les plus proches 
(en utilisant la distance euclidienne, ou autres) et choisira la classe des voisins majoritaires.


"""

#Load of the data
import os
os.chdir ('/home/b/Documents/Python/Data/datasets_knn-master')

#Importer packages recquis 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns                # statistical data visualization
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix


#On lit les fichiers csv 
train =pd.read_csv("synth_train.txt", sep='\t')
test=pd.read_csv("synth_test.txt", sep='\t')
Xtrain=train[['x1','x2']].values
Ytrain=train.iloc[:,0].values
Xtest=test[['x1', 'x2']].values
Ytest=test.iloc[:,0].values

#On affiche les données
sns.scatterplot(x="x1", y="x2", data=train[train["y"] == 1], label=1)
sns.scatterplot(x="x1", y="x2", data=train[train["y"] == 2], label=2)


# On définit la méthode
knn=KNeighborsClassifier(n_neighbors=30)
#On entraîne le modèle :
knn.fit(Xtrain, Ytrain)
pred_test=knn.predict(Xtest)
#Evaluer le modèle en utilisant le taux d'erreur :
err_test30=np.sum(Ytest != pred_test)/len(Ytest)
print('err_test30: ', err_test30)
#Evaluer le modèle en utilisant le score :
knn.score(Xtest, Ytest)
#Matrice de confusion :
matrix_confusion = confusion_matrix(Ytest, pred_test)