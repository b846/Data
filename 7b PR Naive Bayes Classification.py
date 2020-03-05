#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 21:02:36 2020

@author: b

Naive Bayes Classification
Classification using the Maximum A Posterio decision rule in a Bayesian setting
Naive Bayes Classification is popular for text classification

A probabilistic classifier doit déterminer les probabilités de n classes
On doit utiliser Bayes rule: P(A|B)= P(B|A).P(A)/P(B)


In the context of classificatin, we can replace A with a class c_i
and B with our set of features c_0 to x_n

On cherche P(x_0,..., x_n)
On suppose que A et B sont indépendants
On suppose que P(c_i|x_0,..., x_n) ~ P(x_0,..., x_n|c_i).P(c_i)

P(x_0,..., x_n) = P(c_i).P(x_0,..., x_n|c_i)/P(c_i|x_0,..., x_n)
"""

"""
https://www.kaggle.com/blackblitz/gaussian-naive-bayes
In this kernel, we will apply Bayesian inference on Santander Customer Transaction data, 
which has a binary target and 200 continuous features.
target : Y, bernouilli, so it can be specified by setting the positive probability
Observations: X
likehood f(x|y) models the distribution of the distribution given that we know the class
posterior p(y|x) is our updated knowledge about the unknown after observation

MAP (Maximum A Posteriori) estimators pick the class with the hightest posterior probability
For binary classification, it has the same effect as setting a threshold of 0.5 for the positive posterior probability.
LMS (Least Mean Squares) estmator E[Y|X] picks the mean of the posterior distribution
For binary classification, this is just the positive posterior probability pY|X(1|x), which is what we need to submit for the competition.
"""

#Checking Assumptions
#we need to check the likehood distribution are normal and independant

#Import of librairies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer




# changement du repertoire de travail
import os
os.getcwd()
os.chdir ('/home/b/Documents/Python/Data/santander-customer-transaction-prediction')
#Montre les fichiers disponibles
for root, directories, files in os.walk("."):  
    for file in files:
        print(file)

# Définition des paramètres d'affichages
plt.style.use('bmh')
plt.rcParams['figure.figsize'] = (10, 10)
title_config = {'fontsize': 20, 'y': 1.05}

#Load of the data
train = pd.read_csv('train.csv', nrows=10000)
test = pd.read_csv('test.csv',  nrows=10000)
train.head()

X_train = train.iloc[:, 2:].values.astype('float64')
y_train = train['target'].values

#We will look at the likelihood distributions by plotting the KDE 
#(Kernel Density Estimates) using the pandas.DataFrame.plot.kde.
pd.DataFrame(X_train[y_train == 0]).plot.kde(ind=100, legend=False)
plt.title('Likelihood KDE Plots for the Negative Class', **title_config);
plt.show()

pd.DataFrame(X_train[y_train == 1]).plot.kde(ind=100, legend=False)
plt.title('Likelihood KDE Plots for the Positive Class', **title_config);
plt.show()
#The KDE plots above suggest that the likelihood distributions have different centers and spread. We will standardize them 

#Standardization of the data
scaled = pd.DataFrame(StandardScaler().fit_transform(X_train))

# Kernel Density Estimates with the standardized data with target 0
scaled[y_train == 0].plot.kde(ind=100, legend=False)
plt.title('Likelihood KDE Plots for the Negative Class after Standardization', **title_config);
plt.show()

#Kernel Density Estimates with the standardized data with target 1
scaled[y_train == 1].plot.kde(ind=100, legend=False)
plt.title('Likelihood KDE Plots for the Positive Class after Standardization', **title_config);
plt.show()
#Now the KDE plots above look approximately normal, but some have small bumps on the left or right. 
#We can proceed without doing anything, or we can use quantile transformation to remove the small bumps. 
#It turns out that the transformation provides only marginal improvement in performance (0.001 in cross-validation AUC) 
#despite requiring significantly more computation. 

#Ideally, we need to apply the transformation to the features separately for the positive and negative classes. 
#However, we cannot because it becomes a trouble when we are predicting the test data (we do not know the target value). 
#We will instead apply it to the features as a whole

#Apply Quantile transformation
transformed = pd.DataFrame(QuantileTransformer(output_distribution='normal').fit_transform(X_train))

# Plot of the data, after quantile transformation
transformed[y_train == 0].plot.kde(ind=100, legend=False)
plt.title('Likelihood KDE Plots for the Negative Class after Quantile Transformation', **title_config);
plt.show()

transformed[y_train == 1].plot.kde(ind=100, legend=False)
plt.title('Likelihood KDE Plots for the Positive Class after Quantile Transformation', **title_config);
plt.show()
#In the KDE plots above, the likelihood distributions have become normal as we desire.

#Correlation
plt.imshow(transformed.corr())
plt.colorbar()
plt.title('Correlation Matrix Plot of the Features', **title_config)
plt.show()
#The correlation matrix plot above shows very small correlation coefficients between the features.

"""
Finally, it is important that Y is dependent on X. If X and Y were independent, 
then the posterior would be equal to the prior pY|X(y|x)=pY(y),
 and we would not need to do any calculation! 
 We have already seen above that the positive and negative likelihood distributions are slightly different. 
 Let us look at how the sample means and sample variances differ.
"""

# histogram of the difference of the mean between the 2 classes
plt.hist(transformed[y_train == 0].mean() - transformed[y_train == 1].mean())
plt.title('Histogram of Sample Mean Differences between Two Classes', **title_config)
plt.show()


# histogram of the difference of the variance between the 2 classes
plt.hist(transformed[y_train == 0].var() - transformed[y_train == 1].var())
plt.title('Histogram of Sample Variance Differences between Two Classes', **title_config);
plt.show()

"""
While the sample mean differences are more or less balanced around zero, 
the sample variance differences are almost entirely on the negative side. 
This means that the negative likelihood distributions are more concentrated around 
their means than the positive ones.
The plot below shows two features with the least sample variance difference 
(greatest absolute difference where the variance of the positive class is higher). 
Surprisingly, the negative class looks more spread out despite having lower sample 
variance than the positive class.
"""
# Pandas nsmallest() method is used to get n least values from a data frame or a series.
select = (transformed[y_train == 0].var() - transformed[y_train == 1].var()).nsmallest(2).index
plt.scatter(transformed.loc[y_train == 0, select[0]], transformed.loc[y_train == 0, select[1]], alpha=0.5, label='Negative')
plt.scatter(transformed.loc[y_train == 1, select[0]], transformed.loc[y_train == 1, select[1]], alpha=0.5, label='Positive')
plt.xlabel(f'Transformed var_{select[0]}')
plt.ylabel(f'Transformed var_{select[1]}')
plt.title('Positive Class Looks More Concentrated Despite Higher Sample Variance', **title_config)
plt.legend()
plt.show();

#Différence de moyenne entre la positive et la negative class for select[0]
transformed.loc[y_train == 0, select[0]].mean() - transformed.loc[y_train == 1, select[0]].mean()
#Différence de moyenne entre la positive et la negative class for select[1]
transformed.loc[y_train == 0, select[1]].mean() - transformed.loc[y_train == 1, select[1]].mean()

"""
The center of the negative class is above and to the right of that of the positive
 class, but in the above plot, we see straight lines on the lower and left edges.
 The bounds have remained even after quantile transformation. It looks like these
 bounds have prevented the positive class from expanding to the lower and left
 sides. The bounds are more obvious when you look at the original data.
"""
# Display the original data
plt.scatter(X_train[y_train == 0, select[0]], X_train[y_train == 0, select[1]], alpha=0.5, label='Negative')
plt.scatter(X_train[y_train == 1, select[0]], X_train[y_train == 1, select[1]], alpha=0.5, label='Positive')
plt.xlabel(f'var_{select[0]}')
plt.ylabel(f'var_{select[1]}')
plt.title('Bounds in Data', **title_config)
plt.legend()
plt.show()
#the bounds are obvious in the data

"""
Despite the presence of bounds, we are going to assume that the transformed data is normal 
and proceed anyway. We can sample data from normal distributions using np.random.normal 
and plot them for comparison.
"""
# Plot of the normalized distribution
size0 = (y_train == 0).sum() #nb de 0
size1 = y_train.size - size0 #nb de 1
#np.random.normal: Draw random samples from a normal (Gaussian) distribution.
x0 = np.random.normal(transformed.loc[y_train == 0, select[0]].mean(),
                      scale = transformed.loc[y_train == 0, select[0]].std(), size=size0)
y0 = np.random.normal(transformed.loc[y_train == 0, select[1]].mean(),
                      transformed.loc[y_train == 0, select[1]].std(), size=size0)
x1 = np.random.normal(transformed.loc[y_train == 1, select[0]].mean(),
                      transformed.loc[y_train == 1, select[0]].std(), size=size1)
y1 = np.random.normal(transformed.loc[y_train == 1, select[1]].mean(),
                      transformed.loc[y_train == 1, select[1]].std(), size=size1)
plt.scatter(x0, y0, alpha=0.5, label='Negative')
plt.scatter(x1, y1, alpha=0.5, label='Positive')
plt.xlabel(f'Simulated var_{select[0]}')
plt.ylabel(f'Simulated var_{select[1]}')
plt.title('Simulated Data for the Puzzle', **title_config)
plt.legend()
plt.show()

###############################################################################
# Training and Evaluating the Model
"""
Now we are ready to train our model. We combine the quantile transformer and Gaussian naive Bayes classifer, 
sklearn.naive_bayes.GaussianNB, into a pipeline using sklearn.pipeline.make_pipeline.
"""
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB

pipeline = make_pipeline(QuantileTransformer(output_distribution='normal'), GaussianNB())
pipeline.fit(X_train, y_train)

"""
After training the model, we plot the ROC curve on training data and evaluate the model 
by computing the training AUC and cross-validation AUC. We can use sklearn.metrics.roc_curve 
to obtain the values for plotting the curve and sklearn.metrics.auc for computing the AUC.
"""

from sklearn.metrics import roc_curve, auc
# Réalisation d'nue ROC curve
fpr, tpr, thr = roc_curve(y_train, pipeline.predict_proba(X_train)[:,1])
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Plot', **title_config)
auc(fpr, tpr)


from sklearn.model_selection import cross_val_score
#We compute the 10-fold cross-validation score by using sklearn.model_selection.cross_val_score.
print(cross_val_score(pipeline, X_train, y_train, scoring='roc_auc', cv=10).mean())


"""
We achieved good AUC on both training and cross-validation. But is this the best that this model 
can achieve? Let us use simulation to get an estimate of the optimal AUC that this model can achieve. 
We will draw samples from the normal distribution with the 800 parameters of the likelihood. 
The amount of samples to draw from each class will be determined by the prior so that the classes 
have the same proportions as the training data.
"""
from sklearn.metrics import roc_auc_score

pipeline.fit(X_train, y_train)
model = pipeline.named_steps['gaussiannb']
size = 1000000
size0 = int(size * model.class_prior_[0])
size1 = size - size0
sample0 = np.concatenate([[np.random.normal(i, j, size=size0)]
                          for i, j in zip(model.theta_[0], np.sqrt(model.sigma_[0]))]).T
sample1 = np.concatenate([[np.random.normal(i, j, size=size1)]
                          for i, j in zip(model.theta_[1], np.sqrt(model.sigma_[1]))]).T
X_sample = np.concatenate([sample0, sample1])
y_sample = np.concatenate([np.zeros(size0), np.ones(size1)])
roc_auc_score(y_sample, model.predict_proba(X_sample)[:,1])