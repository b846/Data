#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 22:30:57 2020

@author: b

Comment faire du calcul scientique avec Scipy
"""

##############################################
# INTERPOLATION
#############################################
# on va combler les valeurs manquanttes avec interpolate
# Attention, Soyez sûr que votre interpolation ne cache pas la réalité

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 10)
y = x**2
plt.scatter(x, y)

from scipy.interpolate import interp1d
f = interp1d(x, y, kind='linear') # fonction d'interpolation

new_x = np.linspace(0, 10, 30)
result = f(new_x)

# interpolation cubique
y = np.sin(x)
plt.scatter(x, y)
f = interp1d(x, y, kind='cubic') # fonction d'interpolation
result = f(new_x)
plt.scatter(new_x, result)


##############################################
# OPTIMIZE: curve_fit
#############################################
""" Lorsqu'on parle d'optimisation, on parle le plus souvent de minimisation
on trouve également des fonctions qui permettent d'optimiser le placement d'une courbe à 
l'intérieur d'un nuage de point
on trouve aussi de l'algèbre linéaire.
"""
x = np.linspace(0, 2, 100)
y = 1/3*x**3 - 3/5*x**2 + 2 + np.random.randn(x.shape[0])/20
plt.scatter(x, y)

# curve_fit
""" On voudrait développer un modèle statistique qui rentre parfaitement bien dans notre nuage de points
On doit d'abord définir un modèle, avant d'utiliser curve_fit"""
def f(x, a, b, c, d):
    return a*x**3 + b*x**2 +c*x + d

from scipy import optimize

params, param_cov = optimize.curve_fit(f, x, y)
""" Renvoie un talbeau numpy avec les différents paramètres a, b, c, d dans le 1er tableau
Dans le 2ème tableau se trouve les différentes covariances entre ces deux paramètres"""

plt.scatter(x, y)
plt.plot(x, f(x, params[0], params[1], params[2], params[3])) # La fonction rentre parfaitement dans le nuage de points !
plt.show()

##############################################
# OPTIMIZE: minimisation
#############################################
def f(x):
    return x**2 + 15*np.sin(x)

x = np.linspace(-10, 10, 100)
plt.plot(x, f(x))
plt.show()

optimize.minimize(f, x0=-8) #ceci est une fonction
""" optimize.minimize execute un algorithme de minimisation, selon un point de départ x0 -> minimum LOCAL

Pour trouver le maximum global
-on peut essayer un autre point de départ

Ci-dessous, on affiche le résultat avec le point de départ.
"""
x0=-5
result = optimize.minimize(f, x0=x0).x

plt.plot(x, f(x), lw=3, zorder=-1)
plt.scatter(result, f(result), s=100, c='r', zorder=1)
plt.scatter(x0, f(x0), s=200, marker='+', c='g', zorder=1)
plt.show()

# Minimisation en 2 dimensions
def f(x):
    return np.sin(x[0]) + np.cos(x[0]+x[1])*np.cos(x[0])

x, y = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
plt.contour(x, y, f(np.array([x, y])), 20)
plt.show()

x0 =  np.zeros((2, 1))
plt.scatter(x0[0], x0[1], marker='+', c='r', s=100)

result = optimize.minimize(f, x0=x0).x

plt.scatter(result[0], result[1], c='g', s=100)
plt.show()
print(result)



##############################################
# TRAITEMENT DU SIGNAL: Detrend
#############################################
# Signal preprocessing
#fonction detrend: : permet d'éliminer toute tendance linéaire dans un signla
x = np.linspace(0, 20, 100)
y = x + 4*np.sin(x) + np.random.randn(x.shape[0])
plt.plot(x, y)

from scipy import signal
new_y = signal.detrend(y)
plt.plot(x, y)
plt.plot(x, new_y)
plt.show()



##############################################
# TRAITEMENT DU SIGNAL: transformation de fourier
#############################################
""" Transformation de fourier: analyse les fréquences présentes dans un signal périodique
Le résultat est un spectre
"""
x = np.linspace(0, 30, 1000)
y = 3*np.sin(x) + 2*np.sin(5*x) + np.sin(10*x)
plt.plot(x, y)

from scipy import fftpack

fourier = fftpack.fft(y) #numpy.ndarray
power = np.abs(fourier) #numpy.ndarray
frequences = fftpack.fftfreq(y.size)
plt.plot(np.abs(frequences), power)

""" Les applications de la TF
-filtrer un signal
    1) produire le scpectre du signal
    2) utiliser du boolean indexing pour filtrer les signals
    3) appliquer la TF inverse pour obtenir le signal
"""

# Filtre d'un spectre
# 1) Création du signal à filtrer
x = np.linspace(0, 30, 1000)
y = 3*np.sin(x) + 2*np.sin(5*x) + np.sin(10*x) + np.random.randn(x.shape[0])
plt.plot(x, y)
plt.show()

# 2) Application de la TF
fourier = fftpack.fft(y)
power = np.abs(fourier)
frequences = fftpack.fftfreq(y.size)
plt.plot(np.abs(frequences), power)
plt.show()

# 3) Application d'un filtre boolean indexing
fourier[power<400] = 0

# 4) On affiche le nouveau spectre
plt.plot(np.abs(frequences), np.abs(fourier))
plt.show()

# 5) Application de la TF inverse
filtered_signal = fftpack.ifft(fourier)

# Graphe
plt.plot(x, y)
plt.plot(x, filtered_signal)
plt.show()




