#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 21:46:50 2020

@author: b
But d'un graphique:
    - Voir, plutôt qu'imaginer
    - Mieux comprendre le problème
    - Mieux expliquer un phénomène

il existe 2 méthodes pour créer des graphiques sur matplotlib
- la fonction plot
- la méthode orienté objet
"""

import numpy as np

x =np.linspace(0, 2, 10)
y = x**2

# PYPLOT
import matplotlib.pyplot as plt
plt.figure(figsize=(8,6)) #Début de la figure, représente la feuille de travail
plt.plot(x, y, c='black', lw=5, ls='--', label='quadratique')
plt.plot(x, x**3, label='cubic')
plt.xlabel('axe x')
plt.ylabel('y label')
plt.title('title')
plt.legend()
plt.show()

plt.savefig('figure.png') # Sauvegarde de la figure


# SUBPLOT: permet de générer une grille de graphiques
plt.subplot(2, 1, 1) #2 ligne, 1 colonne
plt.plot(x, y, c='red')
plt.subplot(2,1, 2)
plt.plot(x, y, c='blue')



##########################################
# Méthode orienté objet
##########################################
fig, ax = plt.subplots() # on créer une figure et un axe, ils seront des objets, on peut leur appliquer des méthodes
ax.plot(x, y)
""" Avec cette méthode, on peut créer des graphes qui partagent la même abcisse
-fig est un objet
-ax est un tableau ndarray qui contient des objets
"""
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(x, y)
ax[1].plot(x, np.sin(x))
plt.show()


# Exercice
# créer un figure à n colonnes
dataset = {f"experience{i}": np.random.randn(100) for i in range(4)}
dataset = {f"experience{i}": np.random.randn(100, 3) for i in range(4)}

def graphique(data):
    n = len(data)
    plt.figure()
    
    for k, i in zip(data.keys(), range(1, n+1)):
        plt.subplot(n, 1, i)
        plt.plot(data[k])
        plt.title(k)
    plt.show()
graphique(dataset)