#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 21:12:40 2020

@author: b
IMT Mines Alès
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool
from sklearn.preprocessing import OneHotEncoder

# Importing the data
import os
os.chdir ('/home/b/Documents/Python/Data/Tutorial')
data = pd.read_csv('https://raw.githubusercontent.com/lgi2p/decision_tree_labwork/master/gobelins.csv')

data.head()
data.info()
data.columns

col_Id, col_Target = 'id', 'type'
X = data.drop([col_Id, col_Target], axis = 1)
Y = data.loc[:,col_Target]

X = pd.get_dummies(X) # X only contains numerical values
conv = [['Ghost', 0], ['Ghoul',1], ['Goblin',2]]
enc = OneHotEncoder()
enc.fit(conv)
Y = enc.OneHotEncoder().fit_transform(df[feature].values.reshape(-1, 1)).toarray()

# Learn a model
