#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 22:41:45 2020

@author: b
https://www.kaggle.com/rtatman/data-cleaning-challenge-scale-and-normalize-data
"""

# modules we'll use
import pandas as pd
import numpy as np

# for Box-Cox Transformation
from scipy import stats

# for min_max scaling
from mlxtend.preprocessing import minmax_scaling

# plotting modules
import seaborn as sns
import matplotlib.pyplot as plt

# read in all our data
import os
os.chdir ('/home/b/Documents/Python/Data/Kick Starter project')
kickstarters_2017 = pd.read_csv("ks-projects-201801.csv")

# set seed for reproducibility
np.random.seed(0)


# Scaling vs Normalization
"""
Scaling:you're changing the range of your data 
normalization you're changing the shape of the distribution of your data. 

Scaling
This means that you're transforming your data so that it fits within a specific scale,
like 0-100 or 0-1. You want to scale data when you're using methods based on measures 
of how far apart data points, like support vector machines, or SVM or k-nearest neighbors, 
or KNN. With these algorithms, a change of "1" in any numeric feature is given the same 
importance.

For example, you might be looking at the prices of some products in both Yen and US Dollars.
 One US Dollar is worth about 100 Yen, but if you don't scale your prices methods like SVM 
 or KNN will consider a difference in price of 1 Yen as important as a difference of 1 US 
 Dollar! This clearly doesn't fit with our intuitions of the world. With currency, 
 you can convert between currencies. But what about if you're looking at something 
 like height and weight? It's not entirely clear how many pounds should equal one inch 
 (or how many kilograms should equal one meter).

By scaling your variables, you can help compare different variables on equal footing. 
To help solidify what scaling looks like, let's look at a made-up example. 
(Don't worry, we'll work with real data in just a second, this is just to help illustrate 
my point.)
"""

# generate 1000 data points randomly drawn from an exponential distribution
original_data = np.random.exponential(size = 1000)

# mix-max scale the data between 0 and 1
scaled_data = minmax_scaling(original_data, columns = [0])

# plot both together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(original_data, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(scaled_data, ax=ax[1])
ax[1].set_title("Scaled data")


"""
Normalization
Scaling just changes the range of your data. Normalization is a more radical transformation. 
The point of normalization is to change your observations so that they can be described as 
a normal distribution.
Normal distribution: Also known as the "bell curve", this is a specific statistical 
distribution where a roughly equal observations fall above and below the mean, 
the mean and the median are the same, and there are more observations closer to the mean. 
The normal distribution is also known as the Gaussian distribution.

In general, you'll only want to normalize your data if you're going to be using
 a machine learning or statistics technique that assumes your data is normally distributed.
 Some examples of these include t-tests, ANOVAs, linear regression, linear discriminant
 analysis (LDA) and Gaussian naive Bayes. (Pro tip: any method with "Gaussian" in the name 
 probably assumes normality.)

The method were using to normalize here is called the Box-Cox Transformation. 
Let's take a quick peek at what normalizing some data looks like:
    """

# normalize the exponential data with boxcox
normalized_data = stats.boxcox(original_data)

# plot both together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(original_data, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(normalized_data[0], ax=ax[1])
ax[1].set_title("Normalized data")

"""
Your turn!

For the following example, decide whether scaling or normalization makes more sense.

You want to build a linear regression model to predict someone's grades given how much time they spend 
on various activities during a normal school week. You notice that your measurements for how much
 time students spend studying aren't normally distributed: some students spend almost no time studying
 and others study for four or more hours every day. Should you scale or normalize this variable?
 -> Scale the data

You're still working on your grades study, but you want to include information on how students perform 
on several fitness tests as well. You have information on how many jumping jacks and push-ups each student 
can complete in a minute. However, you notice that students perform far more jumping jacks than push-ups:
the average for the former is 40, and for the latter only 10. Should you scale or normalize these variables?
-> Normlize the data
"""


"""
Practice scaling¶
To practice scaling and normalization, we're going to be using a dataset of 
Kickstarter campaigns. (Kickstarter is a website where people can ask people to invest in various projects 
and concept products.)
Let's start by scaling the goals of each campaign, which is how much money they were asking for.
"""
# select the usd_goal_real column
usd_goal = kickstarters_2017.usd_goal_real

# scale the goals from 0 to 1
scaled_data = minmax_scaling(usd_goal, columns = [0])

# plot the original & scaled data together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(kickstarters_2017.usd_goal_real, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(scaled_data, ax=ax[1])
ax[1].set_title("Scaled data")

"""
You can see that scaling changed the scales of the plots dramatically 
(but not the shape of the data: it looks like most campaigns have small goals but a few have very large ones)
"""

"""

Practice normalization¶
Ok, now let's try practicing normalization. 
We're going to normalize the amount of money pledged to each campaign.
"""
# get the index of all positive pledges (Box-Cox only takes postive values)
index_of_positive_pledges = kickstarters_2017.usd_pledged_real > 0

# get only positive pledges (using their indexes)
positive_pledges = kickstarters_2017.usd_pledged_real.loc[index_of_positive_pledges]

# normalize the pledges (w/ Box-Cox)
normalized_pledges = stats.boxcox(positive_pledges)[0]

# plot both together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(positive_pledges, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(normalized_pledges, ax=ax[1])
ax[1].set_title("Normalized data")

