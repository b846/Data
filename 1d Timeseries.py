#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 16:23:22 2020

@author: b
Pandas a été spécialement développé pour traiter les séries temporelles
pandas s'adapte au format de date^
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os
os.getcwd()
os.chdir ('/home/b/Documents/Python/Data/Tutorial')

bitcoin = pd.read_csv('BTC-EUR.csv')

bitcoin.head()
bitcoin.columns

# Graphe
bitcoin['Close'].plot(figsize=(9,6)) #pb on ne voit pas les dates
plt.show()

# On définit un nouveau type d'index: le Date TimeIndex
bitcoin = pd.read_csv('BTC-EUR.csv', index_col='Date', parse_dates=True)
bitcoin['Close'].plot(figsize=(9,6)) #pb on ne voit pas les dates
plt.show()

# indexing sur des dates
bitcoin['2019']['Close'].plot()
plt.show()
bitcoin['2019-09']['Close'].plot()
plt.show()

# Slicing
bitcoin['2017':'2019']['Close'].plot()
plt.show()

# Conversion de date
pd.to_datetime('2019/03/20')


###################################
# RESAMPLE
###################################
bitcoin.loc['2019', 'Close'].resample('M').plot()
plt.show()

# Evolution de la moyenne du bitcoin toutes les 2 semaines !!!
bitcoin.loc['2019', 'Close'].resample('2W').mean().plot()
plt.show()

# Evolution de lécart type pour toutes les semaines
bitcoin.loc['2019', 'Close'].resample('2W').std().plot()
plt.show()

# on affiche toutes les courbes sur un même graphique
plt.figure(figsize=(12,8))
bitcoin.loc['2019', 'Close'].plot()
bitcoin.loc['2019', 'Close'].resample('M').mean().plot(label='moyenne par mois', lw=3, ls=':', alpha=0.8)
bitcoin.loc['2019', 'Close'].resample('W').mean().plot(label='moyenne par semaine', lw=2, ls='--', alpha=0.8)
plt.legend()
plt.show()


###################################
# AGGREGATE
###################################
# Aggregate rassemble dans un mee tableau plusieurs statistiques
m = bitcoin.loc['2019', 'Close'].resample('W').agg(['mean', 'std', 'min', 'max'])

plt.figure(figsize=(12,8))
m['mean']['2019'].plot(label='moyenne par semaine')
plt.fill_between(m.index, m['max'], m['min'], alpha=0.2, label='min-max par semaine')
plt.legend()
plt.show()



###################################
# MOVING AVERAGE
###################################
# moving average permet de calculer une moyenne sur une plage de données
# en se décalant jour après jour, c'est ce qui s'appelle le rolling
bitcoin.loc['2019', 'Close'].rolling(window=7).mean().plot()
plt.show()

plt.figure(figsize=(12,8))
bitcoin.loc['2019', 'Close'].plot()
bitcoin.loc['2019', 'Close'].rolling(window=7).mean().plot(label='moving average non center', lw=3, ls='-')
bitcoin.loc['2019', 'Close'].rolling(window=7, center=True).mean().plot(label='moving average center', lw=3, ls='-')

plt.legend()
plt.show()


###################################
# EXP WEIGHTED FUNCTION
###################################
# permet de retourner une moyenne mobile exponentielle
# les valeurs perdent peu à peu du poids avec le temps
plt.figure(figsize=(12,8))
bitcoin.loc['2019-09', 'Close'].plot()
bitcoin.loc['2019-09', 'Close'].rolling(window=7).mean().plot(label='non center', lw=3, ls ='dotted')
bitcoin.loc['2019-09', 'Close'].rolling(window=7, center=True).mean().plot(label='center', lw=3, ls ='dotted')
bitcoin.loc['2019-09', 'Close'].ewm(alpha=0.6).mean().plot(label='ewm', lw=3, ls ='dotted')
plt.legend()
plt.show()

plt.figure(figsize=(12,8))
bitcoin.loc['2019-09', 'Close'].plot()
for i in np.arange(0.1, 1, 0.1):
    bitcoin.loc['2019-09', 'Close'].ewm(alpha=i).mean().plot(label=f'ewm {i}', lw=2, ls ='--')
plt.legend()
plt.show()




###################################
# MERGE et JOIN
###################################
ethereum = pd.read_csv('ETH-EUR.csv', index_col='Date', parse_dates=True)
ethereum.head()
ethereum.loc['2019', 'Close'].plot()

# méthode inner: les données non communes sont éliminées
pd.merge(bitcoin, ethereum, on='Date', how='inner', suffixes=('_btc', '_etc')).head()
btc_eth = pd.merge(bitcoin, ethereum, on='Date', how='inner', suffixes=('_btc', '_eth'))

# méthode outer: toutes les données sont gardées
pd.merge(bitcoin, ethereum, on='Date', how='outer', suffixes=('_btc', '_eth')).head()

# On affiche les données communes
btc_eth['Close_btc'].plot()
btc_eth['Close_eth'].plot()
plt.show()

btc_eth[['Close_btc', 'Close_eth']].plot()
plt.show()
# Les deux cryptomonnaires ne partagent pas la même échelle

btc_eth[['Close_btc', 'Close_eth']].plot(subplots=True)
plt.show()

# calcul de la corrélation
btc_eth[['Close_btc', 'Close_eth']].corr() # La corrélation est de 74%, c'est énorme


###################################
# Exercice: trading strategy
###################################
# le but est de trouver quand acheter et quand vendre à partir de la moyenne des 28 derniers jours
"""
La stratégie de la tortue
1. Utiliser rolling() pour calculer
 * max sur les 28 derniers jours
 * min sur les 28 derniers jours
2. Boolean indexing:
 * Si 'Close' >max28 alors Buy =1
 * Si 'Close' < min2_ alors sell=-1
"""
# Initialisation
data=bitcoin.copy()
data['Buy'] = np.zeros(len(data))
data['Sell'] = np.zeros(len(data))

""" il est très important de décaler les jours de 1 vers la droite"""
data['RollingMax'] = bitcoin['Close'].shift(1).rolling(window=28).max()
data['RollingMin'] = bitcoin['Close'].shift(1).rolling(window=28).min()
data.loc[data['RollingMax'] < data['Close'], 'Buy'] = 1
data.loc[data['RollingMin'] > data['Close'], 'Sell'] = -1

# graphe, méthode orienté objet pour partager l'axe X
start ='2019'
end = '2019'
fig, ax = plt.subplots(2, figsize=(12, 8), sharex=True)
ax[0].plot(data['Close'][start:end])
ax[0].plot(data['RollingMin'][start:end])
ax[0].plot(data['RollingMax'][start:end])
ax[0].legend()
ax[1].plot(data['Buy'][start:end], c='g')
ax[1].plot(data['Sell'][start:end], c='r')
ax[1].legend(['buy', 'sell'])


