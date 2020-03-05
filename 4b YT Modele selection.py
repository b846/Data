#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 19:17:44 2020

@author: b
https://www.youtube.com/watch?v=T4nZDuakYlU&list=PLO_fdPEVlfKoHQ3Ua2NtDL4nmynQC8YiS&index=9
Selection de variable

Dnas le module sklearn.module_selection,
on retrouve les transformers et les tests de dépendances
Selecteur variance: permet de sélectionner les variables selon leur variance ()
-VarianceThreshold: élimine les variables dont la variance est inférieure à un certain seuil
Selecteur : test statistique
test de dépendance, test ANOVA
-GenericUnivariateSelect
-SelectPercentile: sélecte toutes les variables qui sont au dessus d'un certain pourcentage de score
-SelectKBest: Sélectionne les K variables X dont le score du test de dépendance avec y est le plus élevé
-SelectFpr
-SelectFdr
-SelectFwe
Selecteur estimateur coefs, sélection des variables les plus importantes
-SelectFromModel: entraine un estimateur puis sélectionne les variables les plus importantes pour cet estimateur
Note: compatible avec les estimateurs qui développent une fonction paramétrée (attribut .coef_ ou .feature_importance_)
K-Nearest Neighbour incompatible
-RFE Recursif Feature Elimination: élimine les variables les moins importantes de façon récursive
un estimateur est entrainé plusieurs fois, après chaque entrainement, des features sont éliminées sur 
la base des coefficients les plus faibles de l'estimateur
-RFECV

Test de dépendance: utile pour les problèmes de classification, xhi², ANOVA
-chi2
-f_classif
-mutual_info_classif
Test utile pour la régression: Pearson Corr
-f_regression
-info_regression
"""

#Selecteur variance:
#VarianceThreshold
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.feature_selection import VarianceThreshold

iris = load_iris()
X = iris.data
y = iris.target

plt.plot(X)
plt.legend(iris.feature_names)

# Selection des variables
X.var(axis=0) # Donne la variance selon chaque variable

selector = VarianceThreshold(threshold=0.2)
selector.fit(X)
selector.get_support() # indique les variables qui ont été sélectionner
np.array(iris.feature_names)[selector.get_support()]


# Selecteur : test statistique
# Sélection de variable sur les test de dépendance, en générale, cette technique est plus puissante
# SelectKBest
from sklearn.feature_selection import SelectKBest, chi2
chi2(X, y) # tableau avec chi2 statistique et p-value

selector = SelectKBest(chi2, k=1) # Selecteur qui va retourner 1 variable parmis les 4, celle qui a le plsu d'impact
selector.fit_transform(X, y)
np.array(iris.feature_names)[selector.get_support()]




#Selecteur estimateur coefs
#SelectFromModel
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import SGDClassifier

selector = SelectFromModel(SGDClassifier(random_state=0),
                           threshold='mean')
selector.fit(X, y)
selector.get_support()
# Quelles sont les coefficient qui ont été trouvées ?
selector.estimator_.coef_
"""
Pour bien comprendre la matrice affichée
X.shape : (150,4)
y.shape : (150,1) avec 3 classes
On transforme la matrice X (150*4) en matrice y (150*3) en multipliant par une matrice (4*3)

le vecteur paramètre theta est donc une matrice de 4 lignes et de 3 colonnes
SelectFromModel va sélectionner la moyenne selon les colonnes et va sélectionner toutes les variables supérieure au seuil
"""


# Sélecteur récursif
# RFE Recursif Feature Elimination
from sklearn.feature_selection import RFE, RFECV

selector = RFECV(SGDClassifier(), step=1,  #step: nb de variable à élminer à chaque itération
                 min_features_to_select=2, #min_features_to_select: cb restera-t-il de variable à la fin
                 cv=5) 
selector.fit(X, y)
selector.ranking_ # permet de voir le classement finale des différentes variables
selector.grid_scores_ # score de SGDClassifier à chaque itération, cad à chaque enlèvement de variable



