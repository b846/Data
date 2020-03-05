#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 11:02:42 2020

@author: b
https://www.youtube.com/watch?v=41mnga4ptso&list=PLO_fdPEVlfKoHQ3Ua2NtDL4nmynQC8YiS&index=6

Ce tutoriel python français montre comment développer des pipelines de machine learning avec Sklearn.
Pour développer une pipeline simple, je vous conseille d'utiliser 
la fonction make_pipeline() du module sklearn.pipeline
Mais pour traiter des data sets hétérogènes (avec un mélange de type de variables : 
   continues, discrètes, strings...) il faut utiliser des fonctions un peu plus techniques.

make_column_transformer() permet ainsi de créer un transformer qui ne s'applique que 
sur certaines colonnes de votre dataset. Il est souvent utilisé pour traiter les variables numériques 
et les variables catégorielles de façon différente. Cette fonction existe également sous 
forme de Classe avec ColumnTransformer, mais je préfère utiliser la fonction make_column_transformer
 car  sa syntaxe est plus simple.

make_column_selector() est une nouvelle fonctionnalité de sklearn 0.22 qui permet de séléctionner 
les colonnes d'un dataset selon leur dtype. Tres utile également !

Pour finir, la fonction make_union permet de construire des pipelines paralleles, dont les résultats 
sont concaténé dans un tableau final. Cette fonction existe également sous forme de Classe avec FeatureUnion,
 mais je préfère utiliser la fonction make_union car  sa syntaxe est plus simple.


Combinés ensemble, ces trois fonctions sont redoutables et permettent de traiter des datasets de la 
vraie vie, qui combinent plusieurs types de variables, afin de créer un modele de machine learning très performant.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
import seaborn as sns

titanic = sns.load_dataset('titanic')
titanic.head()

""" Pour appliquer différent tansformeur comme StandardScaler, il va falloir faire le tri entre les colonnes
"""

y = titanic['survived']
X = titanic.drop('survived', axis=1)

#make_column_transformer
# Création d'un mécanisme pour trier les colonnes
# make_column_transformer: permet de trier les colonnes
from sklearn.compose import make_column_transformer
transformer = make_column_transformer((StandardScaler(), ['age', 'fare']))
transformer.fit_transform(X) # le transformeur ne va traiter que les colonne age et fare

# make_column_selector
# TrieSélection  entre les variables catégorielles et numériques
from sklearn.compose import make_column_selector
numerical_features = make_column_selector(dtype_include=np.number)
categorical_features = make_column_selector(dtype_exclude=np.number)



from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
numerical_features = ['pclass', 'age', 'fare']
categorical_features = ['sex', 'deck', 'alone']

#SimpleImputer va enlever les valeurs manquantes
numerical_pipeline = make_pipeline(SimpleImputer(), 
                                   StandardScaler())

categorical_pipeline = make_pipeline(SimpleImputer(strategy = 'most_frequent'), #remplace les valeurs manquantes par les plus fréquences
                                   OneHotEncoder())

preprocessor = make_column_transformer((numerical_pipeline, numerical_features),
                                       (categorical_features, categorical_features))

model = make_pipeline(preprocessor(), SGDClassifier())
model.fit(X,y)

model = preprocessor()


# make_union