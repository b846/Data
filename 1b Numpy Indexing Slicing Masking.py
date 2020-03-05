#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 20:03:31 2020

@author: b
"""

import numpy as np
values = [1, 2.4, 234, 112, 345]
array = np.array(values)
A = np.arange(1, 10, 1).reshape(3,3)
b = np.ones((3,6))

# indexing
A[0, 1]

# Slicing
# A[debut:fin:pas, debut:fin:pas]
A[:, 0]

# Subsetting
B = A[0:2, 0:2]


C = np.zeros((4, 4))
C[1:3, 1:3] = 1
C

A = np.random.randint(0, 10, [5, 5])
A

# Booelan indexing
A[A < 5] = 10
A[(A<5) & (A>2)] = 12

# Masking
A[A>5]

#Tableau de fonctions
def f(i,j):
    return 10*i +j
np.fromfunction(f,(4,5),dtype =int)


###############################################
# NUMPY STATISTIQUE ET MATHEMATIQUE
###############################################
# Méthode appartenant à la classe ndarray
A = np.random.randint(0, 10, [10, 3])
np.random.rand(6)   # tableaux aléatoires de 6 nombres
A.sum(axis=0)
A.cumsum()
A.prod() # produit entre les coefficients
A.min(axis=0) #minimum selon l'axe zéro
A.argmin(axis=0) # Position du minimum
A.sort()
A.argsort() # retourne la façon dont les éléments doient être ordonnés pour trier les éléments
A.mean() #variance
A.std() # écart type
A.var() #variance
A


np.corrcoef(A) #permet de tracer une matrice de corrélation
values, counts = np.unique(A, return_counts=True) #renvoie les entités et leur répétition
counts.argsort()
values[counts.argsort()] #liste dans l'ordre les éléments les plus fréquents

# Affiche les valeurs des plus fréquentes au moins fréquentes
for i, j in zip(values[counts.argsort()], counts[counts.argsort()]):
    print(f'valeur {i} apparait {j}')

#NAN Corrections
# les fonctions suivantes permettent de calculer une moyennes même avec une valeur nan
A = np.random.randn(5, 5)
A[2, 2] = np.nan
A[4, 1] = np.nan
np.nanmean(A)
np.nanstd(A)
np.isnan(A) #Masque numpy avec la présence des nan
np.isnan(A).sum()
np.isnan(A).sum()/A.size
A[np.isnan(A)]=0


###############################################
# NUMPY ALGEBRE LINEAIRE
###############################################
A = np.ones((2, 3))
B = np.ones((3, 2))
A.T #Transposé

A.dot(B) #Produit matricielle


# Inversion de matrice carré de déterminant non null
A = np.random.randint(0, 10, [3, 3])
A
np.linalg.det(A)
np.linalg.inv(A)
np.linalg.pinv(A) #permet d'inverser une matrice avec qq différences
# Principale Component Analysis utilise les valeurs propres
np.linalg.eig(A)

# Exercice: standardiser une matrice A
np.random.seed(0)
A = np.random.randint(0, 100, [10, 5])
A
# Ma solution
moy = A.mean(axis=0)
ecart = A.std(axis=0)
for col in range(5):
    A[:,col] = (A[:,col] - moy[col])/ecart[col]
#La solution de Guillaume
D = (A - A.mean(axis=0)) / A.std(axis=0)
D.mean(axis=0)
D.std(axis=0)

###############################################
# NUMPY BROADCASTING
###############################################
    # Le broadcasting consiste à étendre les dimensions d'un tableau
    # Il faut faire attention aux dimensions de nos jeux de données !!!
np.random.seed(0)
A = np.random.randint(0, 10, [2, 3])
B = np.ones((2, 1))

A + 2
A + B # Les colonnes de B ont été étendues selon les colonnes
# pour faire du broadcasting, il y a des règles
# A et B dimensions égales ou égales à 1
A = np.random.randint(0, 10, [4, 1])
B = np.ones((1, 3))
A + B #les dimensions des vecteurs ont été étendues




