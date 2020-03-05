#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 10:16:15 2020

@author: b
make_scorer() est utile pour créer ses propres métrics et les utiliser dans des algorithmes 
de Cross Validation, GridSearchCV, etc
Est utile dans le milieu professionnel, car le client n'en avoir rien à faire de notre
coeffient quadratique moyenne
Le client nous fournit un projet, avec un cahier des charges, dans lequel il y a des contraintes
Parmis ces contraintes, on va trouver des mesures de performances qui sont spécifiques au projet

Une metric est une fonction
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
m = 100
X = np.linspace(0, 4, m).reshape((m, 1))
y = 2 + X**1.3 + np.random.randn(m, 1)
y = y.ravel() # Return a contiguous flattened array.

plt.scatter(X, y)
plt.xlabel('X')
plt.ylabel('y')
plt.show()


# Régression linéaire
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X,y)
y_pred = model.predict(X)

plt.scatter(X, y)
plt.plot(X, y_pred, c='r', lw=3)

# Mesure de l'erreur
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y, y_pred)

plt.figure(figsize=(9,6))
plt.scatter(X, y)
plt.plot(X, y_pred, c='r', lw=3)
plt.plot(X, y + y*0.2, c='g', ls='--')
plt.plot(X, y - y*0.2, c='g', ls='--')


""" Notre client attends la contrainte suivante:
    toutes les valeurs doivent être dans une tolérance de +/- 20% par rappoirt à nos valeurs y"""
    
##########################################################
    # Création d'une métric
def custom_metric(y, y_pred):
    """ Metric permettant de mesurer une tolérance de +/- 20%"""
    return np.sum((y_pred < y + y*0.2) & (y_pred > y - y*0.2))/y.size
custom_metric(y, y_pred) # il y a 63% de predictions qui sont ok

from sklearn.metrics import make_scorer
# make_scorer(function, greater_is_better)
# greater_is_better=True signie que plus la valeur est grande, plus le modèle est performant^
custom_score = make_scorer(custom_metric, greater_is_better=True)

from sklearn.model_selection import cross_val_score, GridSearchCV
cross_val_score(LinearRegression(), X, y, cv=3, scoring=custom_score) # Le score est mauvais, car nous avons un modèle linéaire

# Création d'un modèle SVR
from sklearn.svm import SVR
model = SVR(kernel='rbf', degree=3)
params = {'gamma': np.arange(0.1, 1, 0.05)}

grid = GridSearchCV(model, param_grid=params, cv=3, scoring=custom_score)
grid.fit(X, y)
#estimator..get_params().keys(): check the list of available parameters
best_model = grid.best_estimator_

# Mesure de notre modèle
y_pred = best_model.predict(X)
custom_metric(y, y_pred)

# Graphique
plt.figure(figsize=(9,6))
plt.scatter(X, y)
plt.plot(X, y_pred, c='r', lw=3)
plt.plot(X, y + y*0.2, c='g', ls='--')
plt.plot(X, y - y*0.2, c='g', ls='--')


"""
Pour résumer, grâce à la fonction make_scorer, on a pu convertir la métric qui a été converti par notre client
en un scoreur que l'on peut utiliser pour entrainer plusieurs modèles
et sélectionner le modèle qui nous donnait la meilleure performance selon la métric attendue par notre client"""