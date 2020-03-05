#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 15:05:17 2020

@author: b
"""


import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

#######################################""
# 1. Dataset
np.random.seed(0) # gère l'aélatoire
x, y = make_regression(n_samples=100, n_features=2, noise=10) #n_features est important
y = y + abs(y)
plt.scatter(x[:,0], y) # afficher les résultats. X en abscisse et y en ordonnée
plt.show()

# Dimension des données
print(x.shape)
print(y.shape)
# redimensionner y
y = y.reshape(y.shape[0], 1)
print(y.shape)

# Création de la matrice X contenant les biais
X = np.hstack((x, np.ones((x.shape[0], 1)))) # X contient x1, x2 et les bias
print(X.shape)
print(X[:10])

# Création du vecteur theta contenant les paramètres de la régression
np.random.seed(0) # pour produire toujours le meme vecteur theta aléatoire
theta = np.random.randn(3, 1)
theta


#########################################################
# 2. Création du modèle linéaire
def model(X, theta):
    #retourne le produits matricielle de X.theta
    return X.dot(theta)
model(X,theta)
plt.scatter(x[:,0],y)
plt.scatter(x[:,0],model(X, theta))
plt.show()


#########################################################
# 3. Fonction coût
def cost_function(X, y, theta):
    # retourne le coût entre X et y pour un theta choisit
    m = len(y)
    return 1/(2*m) * np.sum((model(X, theta) - y)**2)
cost_function(X, y, theta)

#########################################################
# 4. Descente de gradient
def grad(X, y, theta):
    # Renvoie le gradient du biais, afin de trouver la pente
    m = len(y)
    return 1/m * X.T.dot(model(X, theta) - y)
grad(X, y, theta).shape

########################################################
# 5. Algorithme descente de gradient
def gradient_descent(X, y, theta, learning_rate, n_iterations):
    # Algorithme itératif pour progresser vers le minimum
    cost_history = np.zeros(n_iterations) # création d'un tableau de stockage pour enregistrer l'évolution du Cout du modele
    
    for i in range(0, n_iterations):
        theta = theta - learning_rate * grad(X, y, theta) # mise a jour du parametre theta (formule du gradient descent)
        cost_history[i] = cost_function(X, y, theta) # on enregistre la valeur du Cout au tour i dans cost_history[i]
        
    return theta, cost_history


###########################################################
# 6. Phase d'entrainement
n_iterations = 1000
learning_rate = 0.01

theta_final, cost_history = gradient_descent(X, y, theta, learning_rate, n_iterations)
theta_final # voici les parametres du modele une fois que la machine a été entrainée

# création d'un vecteur prédictions qui contient les prédictions de notre modele final
predictions = model(X, theta_final)
predictions.shape

# Affiche les résultats de prédictions (en rouge) par rapport a notre Dataset (en bleu)
plt.scatter(x[:,0],y)
plt.scatter(x[:,1],predictions)
plt.show()

###########################################################
# 7. Courbe d'apprentissage
# la courbe d'apprentissage est sensée diminuée, il faut gérer le learning rate
plt.plot(range(n_iterations), cost_history)


###########################################################
# 8. Evaluation finale
# On calcule le correficient de corrélation pour déterminer la fiabilité de notre modèle
def coef_determination(y, pred):
    u = ((y - pred)**2).sum()
    v = ((y - y.mean())**2).sum()
    return 1 - u/v
coef_determination(y, predictions)
