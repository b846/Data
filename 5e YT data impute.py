#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 18:36:07 2020

@author: b
https://www.youtube.com/watch?v=QVEJJNsz-eM&list=PLO_fdPEVlfKoHQ3Ua2NtDL4nmynQC8YiS&index=7
sklearn.impute: nettoyer les données
permet de remplacer les données manquantes

Cela peut-être utile de rajouter une colonne pour dire si nous avons une valeur manquante
un manque d'information dans le nom de la classe peut indiquer qu'il s'agit d'un membre de l'équipage
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import SGDClassifier

# SimpleImputer
from sklearn.impute import SimpleImputer

X = np.array([[10, 3],
              [0, 4],
              [5, 3],
              [np.nan, 3]])

imputer = SimpleImputer(missing_values=np.nan,
                        strategy='mean')

imputer.fit_transform(X)

# Si on applique cette méthode au X_test, les valeurs
# manquantes seront remplacées par 5 !

X_test = np.array([[12, 3],
                  [40, 2],
                  [5, 5],
                  [np.nan, np.nan]])
imputer.transform(X_test)


# KNNImputer
from sklearn.impute import KNNImputer

X = np.array([[1, 100],
              [2, 30],
              [3, 15],
              [np.nan, 20]])
imputer = KNNImputer(n_neighbors=1)
imputer.fit_transform(X)


# MissingIndicator
# Variable booléenne qui indique l'absence de valeurs dans le dataset
from sklearn.impute import MissingIndicator
X = np.array([[1, 100],
              [2, 30],
              [3, 15],
              [np.nan, 20]])
MissingIndicator().fit_transform(X)

# On applique en parallèle 2 méthodes
from sklearn.compose import make_union
pipeline = make_union(SimpleImputer(strategy='constant', fill_value=-99),
                      MissingIndicator())
pipeline.fit_transform(X)


#################################################
# Application au Titanic
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
import seaborn as sns

titanic = sns.load_dataset('titanic')
X = titanic[['pclass', 'age']]
y = titanic['survived']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5) #test_size avec 20%

model = make_pipeline(KNNImputer(), SGDClassifier())

params = {
        'knnimputer__n_neighbors': [1, 2, 3, 4]
        }

grid = GridSearchCV(model, param_grid=params, cv=5)
grid.fit(X_test)

grid.best_params_