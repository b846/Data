#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 09:03:30 2020

@author: b
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()
x = iris.data
y = iris.target

print(x.shape)
plt.scatter(x[:,0], x[:,1], alpha=0.8)

#On disivise le data test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5) #test_size avec 20%

#model
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=10)
model.fit(x_train, y_train)
print('test score: ', model.score(x_test, y_test))

# Cross validation
from sklearn.model_selection import cross_val_score
cross_val_score(KNeighborsClassifier(4), x_train, y_train, cv=5, scoring='accuracy').mean()

# Comparaison des modèles entre eux
val_score = []
for k in range(1, 50):
    val_score.append(cross_val_score(KNeighborsClassifier(n_neighbors=k), x_train, y_train, cv=5).mean())
plt.plot(val_score)
# On obtient les meilleurs per performanceds lorsque le nb de voisin est de 10


#Validation curve
# On affiche l'évolution d'un score lorsque l'on fait varier un hyperparamètre
from sklearn.model_selection import validation_curve
model = KNeighborsClassifier()
k = np.arange(1, 50)
train_score, val_score = validation_curve(model, x_train, y_train, param_name='n_neighbors', k, cv=5)
# Graphique
plt.plot(k, val_score.mean(axis=1), label='validation')
plt.plot(k, train_score.mean(axis=1), label='train')
plt.xlabel('score')
plt.ylabel('n_neighbors')
plt.legend()
plt.show()
# On peut ici repérer les cas d'overfitting: bon train_score, mauvais test_score

#####################################################################"
# GridSearchCV permet de tester toutes les combinaisons des hyperparmetres
from sklearn.model_selection import GridSearchCV
param_grid = {'n_neighbors': np.arange(1,20),
              'metric': ['euclidean', 'manhattan']}
grid = GridSearchCV(KNeighborsClassifier(), param_grid=param_grid, cv=5)
grid.fit(x_train, y_train)
grid.best_score_
grid.best_params_
model = grid.best_estimator_
model.score(x_test, y_test)



# Confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, model.predict(x_test))

############################################################
#Courbe d'apprentissage, learning scurve
# La courbe d'apprentissage permet de savoir si le modèle purrait avoir de meilleures performances avec plus de données
# learning curve: montre l'évolution des performances du modèle en fonction de la quantité de données
from sklearn.model_selection import learning_curve
N, train_score, val_score = learning_curve(model, x_train, y_train, train_sizes=np.linspace(0.2, 1.0, 5), cv=5)
print(N)
plt.plot(N, train_score.mean(axis=1), label='train')
plt.plot(N, val_score.mean(axis=1), label='validation')
plt.xlabel('train_sizes')
plt.legend()
plt.show()
# La performance n'évolue plus à partir de train_sizes = 40
