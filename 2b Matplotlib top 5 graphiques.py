#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 19:28:06 2020

@author: b
"""

#############################
# CLASSIFICATION
############################

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()

x = iris.data
y = iris.target
names = list(iris.target_names)

print(f'x contient {x.shape[0]} exemples et {x.shape[1]} variables')
print(f'il y a {np.unique(y).size} classes')

########################################
# 5) GRAPHQIUE DE CLASSIFICATION AVEC plt.scatter
#######################################
plt.scatter(x[:, 0], x[:, 1], c=y, alpha=0.5) # on affiche les différentes classes, alpha permet de controler la transparence
plt.xlabel('longeur sépal')
plt.ylabel('largeur sépal')
plt.show()

plt.scatter(x[:, 0], x[:, 1], c=y, alpha=0.5, s=x[:,2]*100) # on controle la taille des points
plt.xlabel('longeur sépal')
plt.ylabel('largeur sépal')
plt.show()


########################################
# 4) GRAPHQIUE 3D
#######################################
from mpl_toolkits.mplot3d import Axes3D
ax = plt.axes(projection ='3d')
ax.scatter(x[:,0], x[:, 1], x[:,2])

f = lambda x, y: np.sin(x) + np.cos(y)
X = np.linspace(0, 5, 100)
Y = np.linspace(0, 5, 100)
X, Y = np.meshgrid(X, Y)
Z = f(X, Y)
Z.shape

ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, cmap='plasma')
plt.show()

########################################
# 3) HISTOGRAMME
#######################################
#bins est le nombre de sections
plt.hist(x[:,0], bins=30) # distribution pour la variable 0
plt.hist(x[:,0], bins=10) # distribution pour la variable 0

#Création de 2 histograme
plt.hist(x[:,0], bins=20)
plt.hist(x[:,1], bins=20)
plt.show()

# histogramme en 2D
plt.hist2d(x[:,0], x[:,1], cmap='Blues')
plt.xlabel('longueur sépal')
plt.xlabel('largeur sépal')
plt.colorbar()
plt.show()


###########################################
# 2) Countour plot
############################################
# Permet de visualiser les lignes de niveau, utile pour les algorithmes d'optimisation
plt.contour(X, Y, Z, 20)
plt.show()

plt.contour(X, Y, Z, 20, colors='black')
plt.show()

plt.contourf(X, Y, Z, 20, cmap='RdGy')
plt.show()


###########################################
# 1) Imshow
############################################
# permet d'afficher n'importe quel matrice numpy
#Permet de visualiser des masques par exemple
plt.imshow(np.corrcoef(x.T), cmap='Blues')
plt.colorbar()
plt.show()






