#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 21:35:01 2020

@author: b
"""


# LabelEncoder: Encore categorical values to value between 0 and n-1
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit([1, 2, 2, 6])
le.classes_
le.transform([1, 1, 2, 6])
le.inverse_transform([0, 0, 1, 2])

le = preprocessing.LabelEncoder()
le.fit(["paris", "paris", "tokyo", "amsterdam"])
list(le.classes_)
le.transform(["tokyo", "tokyo", "paris"])
list(le.inverse_transform([2, 2, 1]))

#Hous price
houce_price = pd.read_csv('/home/b/Documents/Python/Data/house_price_train.csv')
col_Id, col_Target = 'Id', 'SalePrice'
# Qualitative and quantitative
quantitative = [f for f in houce_price.columns if houce_price.dtypes[f] != 'object']
if col_Target in quantitative:
    quantitative.remove(col_Target)
if col_Id in quantitative:
    quantitative.remove(col_Id)
qualitative = [f for f in houce_price.columns if houce_price.dtypes[f] == 'object']

# Differentiate numerical features (minus the target) and categorical features
categorical_features = houce_price.select_dtypes(include=['object']).columns
print(categorical_features)
numerical_features = houce_price.select_dtypes(exclude = ["object"]).columns
print(numerical_features)

print("Numerical features : " + str(len(numerical_features)))
print("Categorical features : " + str(len(categorical_features)))
feat_num = houce_price[numerical_features]
feat_cat = houce_price[categorical_features]


# titanic
titanic = pd.read_csv('/home/b/Documents/Python/Data/titanic_train.csv')
col_Id, col_Target = 'PassengerId', 'Survived'

quantitative = [f for f in titanic.columns if titanic.dtypes[f] != 'object']
if col_Target in quantitative:
    quantitative.remove(col_Target)
if col_Id in quantitative:
    quantitative.remove(col_Id)
qualitative = [f for f in titanic.columns if titanic.dtypes[f] == 'object']

columns_to_remove = ['Name', 'PassengerId', 'Ticket', 'Cabin', col_Target]
titanic_num = titanic.drop(columns_to_remove, axis=1)

### Categorical data to numerical data
# the object type and category type will be converted to numerical type with LabelEncoder
object_category_col = ['Embarked', 'Parch', 'Sex', 'SibSp', 'Deck', 'Title']
#sklearn.preprocessing.LabelEncoder: Encode target labels with value between 0 and n_classes-1.
le = preprocessing.LabelEncoder()
for feature in object_category_col:        
        titanic_num[feature] = LabelEncoder().fit_transform(titanic[feature])
        
        
        
train_data_csv_df['Embarked'] = train_data_csv_df['Embarked'].map({'S':0, 'C':1, 'Q':2})