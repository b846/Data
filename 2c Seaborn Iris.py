#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 19:56:31 2020

@author: b
Seaborn est une librairie basée sur Matplotlib et sur Pandas
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import seaborn as sns

iris = load_iris()

data = pd.DataFrame(iris.data, columns=iris.feature_names)
target = pd.DataFrame(iris.target, columns=['target'])
df = pd.concat([data, target], axis=1)
df.columns

############################################
# Visualiser la relations entre 2 colonnes
############################################
plt.scatter(df['sepal length (cm)'], df['sepal width (cm)'])
plt.show()

sns.pairplot(df)

sns.pairplot(df, hue='target')



###############################################"
# TITANIC
###############################################
titanic = sns.load_dataset('titanic')
titanic.head()

titanic.drop(['alone', 'alive', 'who', 'adult_male', 'embark_town', 'class'], axis=1, inplace=True)
titanic.dropna(axis=0, inplace=True)
titanic.head()

sns.pairplot(titanic)


# catplot
sns.catplot(x='pclass', y='age', data=titanic)
sns.catplot(x='pclass', y='age', data=titanic, hue='sex')


# boxplot
sns.boxplot(x='pclass', y='age', data=titanic, hue='sex')


# Histogram
sns.distplot(titanic['fare'])

# joinplot
sns.jointplot(x='fare', y='age', data=titanic, kind='kde')

#heatmap
sns.heatmap(titanic.corr())