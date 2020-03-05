#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 16:01:15 2020

@author: b
This challenge serves as final project for the "How to win a data science competition" Coursera course.
In this competition you will work with a challenging time-series dataset consisting of daily sales data,
 kindly provided by one of the largest Russian software firms - 1C Company. 

We are asking you to predict total sales for every product and store in the next month. 
By solving this competition you will be able to apply and enhance your data science skills.
https://www.kaggle.com/jagangupta/time-series-basics-exploring-traditional-ts

 

Competition and data overview:
In this playground competition, we are provided with the challenge of predicting total sales
for every product and store in the next month for Russian Software company-1c company.

What does the IC company do?:
1C: Enterprise 8 system of programs is intended for automation of everyday enterprise activities:
various business tasks of economic and management activity, such as management accounting,
business accounting, HR management, CRM, SRM, MRP, MRP, etc.
Data: We are provided with daily sales data for each store-item combination,
but our task is to predict sales at a monthly level.
"""


# Imports
# always start with checking out the files!

# Basic packages
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random as rd # generating random numbers
import datetime # manipulating date formats

# Viz
import matplotlib.pyplot as plt # basic plotting
import seaborn as sns # for prettier plots


# TIME SERIES
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pandas.plotting import autocorrelation_plot # Autocorrelation plot for time series.
from statsmodels.tsa.stattools import adfuller, acf, pacf,arma_order_select_ic
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
 

# settings
import warnings
warnings.filterwarnings("ignore")
# Option d'affichage
pd.options.display.max_columns = 8    #option d'affichage, none means no maximum value
pd.options.display.max_rows = 20

# Import all of them
import os
os.chdir ('/home/b/Documents/Python/Data/competitive-data-science-predict-future-sales')
sales=pd.read_csv("sales_train.csv")
item_cat=pd.read_csv("item_categories.csv")
item=pd.read_csv("items.csv")
sub=pd.read_csv("sample_submission.csv")
shops=pd.read_csv("shops.csv")
test=pd.read_csv("test.csv")

#formatting the date column correctly
sales.date=sales.date.apply(lambda x:datetime.datetime.strptime(x, '%d.%m.%Y'))
# check
print(sales.info())

# Aggregate to monthly level the required metrics
monthly_sales=sales.groupby(["date_block_num","shop_id","item_id"])[
        "date","item_price","item_cnt_day"].agg({"date":["min",'max'],
        "item_price":"mean","item_cnt_day":"sum"})

## Lets break down the line of code here:
# aggregate by date-block(month),shop_id and item_id
# select the columns date,item_price and item_cnt(sales)
# Provide a dictionary which says what aggregation to perform on which column
# min and max on the date
# average of the item_price
# sum of the sales

# take a peak
monthly_sales.head(20)


# number of items per cat
x=item.groupby(['item_category_id']).count()
x=x.sort_values(by='item_id',ascending=False)
x=x.iloc[0:10].reset_index()
x

# #plot
plt.figure(figsize=(8,4))
ax= sns.barplot(x.item_category_id, x.item_id, alpha=0.8)
plt.title("Items per Category")
plt.ylabel('# of items', fontsize=12)
plt.xlabel('Category', fontsize=12)
plt.show()


"""
Of course, there is a lot more that we can explore in this dataset, but let's dive into the time-series part.
Single series:
The objective requires us to predict sales for the next month at a store-item combination.
Sales over time of each store-item is a time-series in itself. Before we dive into all the combinations,
first let's understand how to forecast for a single series.

I've chosen to predict for the total sales per month for the entire company.
First let's compute the total sales per month and plot that data.
"""

 

# Plot of Total Sales of the compagny pour 1 mois
ts=sales.groupby(["date_block_num"])["item_cnt_day"].sum()
ts.astype('float')
plt.figure(figsize=(12,6))
plt.title('Total Sales of the company')
plt.xlabel('Time')
plt.ylabel('Sales')
plt.plot(ts);

#Plot
plt.figure(figsize=(10,5))
plt.plot(ts.rolling(window=12,center=False).mean(),label='Rolling Mean');
plt.plot(ts.rolling(window=12,center=False).std(),label='Rolling sd');
plt.legend();

"""
Quick observations: There is an obvious "seasonality" (Eg: peak sales around a time of year) and a decreasing "Trend".
Let's check that with a quick decomposition into Trend, seasonality and residuals.
"""
import statsmodels.api as sm
# multiplicative
res = sm.tsa.seasonal_decompose(ts.values,freq=12,model="multiplicative")
#plt.figure(figsize=(16,12))
fig = res.plot()
#fig.show()


# Additive model
res = sm.tsa.seasonal_decompose(ts.values,freq=12,model="additive")
#plt.figure(figsize=(16,12))
fig = res.plot()
#fig.show()


# STATIONARITY
"""
What does it mean for data to be stationary ?
1) The mean of the series should not be a function of time.
2) The variance of the series should not be a function of time.
This property is also known as homoscedasticity.
3) Finally, the covariance of the i term and the (i+m)th term should not be a function of time.

Stationarity refers to time-invariance of a series. (ie) Two points in a time series are related to each other by only how far apart they are, and not by the direction(forward/backward)
When a time series is stationary, it can be easier to model. Statistical modeling methods assume or require the time series to be stationary.

There are multiple tests that can be used to check stationarity.
- ADF( Augmented Dicky Fuller Test)
- KPSS
- PP (Phillips-Perron test)

Let's just perform the ADF which is the most commonly used one.
"""

# Stationarity tests
def test_stationarity(timeseries):
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)

test_stationarity(ts)

# to remove trend
from pandas import Series as Series
# create a differenced series
def difference(dataset, interval=1):
    #réalise la différence entre la donnée k et la donné k - interval
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)

# invert differenced forecast
def inverse_difference(last_ob, value):
    return value + last_ob

#Plot of f(Time) = Sales, Original
ts=sales.groupby(["date_block_num"])["item_cnt_day"].sum()
ts.astype('float')
plt.figure(figsize=(16,16))
plt.subplot(311)
plt.title('Original')
plt.xlabel('Time')
plt.ylabel('Sales')
plt.plot(ts)

 
#Plot of f(Time) = Sales, After De-trend
plt.subplot(312)
plt.title('After De-trend')
plt.xlabel('Time')
plt.ylabel('Sales')
new_ts=difference(ts)
plt.plot(new_ts)
plt.plot()


#Plot of f(Time) = Sales, After De-seasonalization
plt.subplot(313)
plt.title('After De-seasonalization')
plt.xlabel('Time')
plt.ylabel('Sales')
new_ts=difference(ts,12) # assuming the seasonality is 12 months long
plt.plot(new_ts)
plt.plot()

# now testing the stationarity again after de-seasonality
test_stationarity(new_ts)

 
"""
Now after the transformations, our p-value for the DF test is well within 5 %. Hence we can assume Stationarity of the series
We can easily get back the original series using the inverse transform function that we have defined above.

Now let's dive into making the forecasts!
AR, MA and ARMA models:
TL: DR version of the models:
MA - Next value in the series is a function of the average of the previous n number of values AR -
The errors(difference in mean) of the next value is a function of the errors in the previous n number of values ARMA - a mixture of both.

Now, How do we find out, if our time-series in AR process or MA process?
Let's find out!
"""
def tsplot(y, lags=None, figsize=(10, 8), style='bmh',title=''):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    with plt.style.context(style):    
        fig = plt.figure(figsize=figsize)
        #mpl.rcParams['font.family'] = 'Ubuntu Mono'
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        qq_ax = plt.subplot2grid(layout, (2, 0))
        pp_ax = plt.subplot2grid(layout, (2, 1))
        
        y.plot(ax=ts_ax)
        ts_ax.set_title(title)
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)
        sm.qqplot(y, line='s', ax=qq_ax)
        qq_ax.set_title('QQ Plot')        
        scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)

        plt.tight_layout()


#AR(1) process -- has ACF tailing out and PACF cutting off at lag=1
# Simulate an AR(1) process with alpha = 0.6
np.random.seed(1)
n_samples = int(1000)
a = 0.6
x = w = np.random.normal(size=n_samples)

for t in range(n_samples):
    x[t] = a*x[t-1] + w[t]
limit=12    
tsplot(x, lags=limit,title="AR(1)process")

 

#AR(2) process -- has ACF tailing out and PACF cutting off at lag=2
# Simulate an AR(2) process
n = int(1000)
alphas = np.array([.444, .333])
betas = np.array([0.])

# Python requires us to specify the zero-lag value which is 1
# Also note that the alphas for the AR model must be negated
# We also set the betas for the MA equal to 0 for an AR(p) model
# For more information see the examples at statsmodels.org
ar = np.r_[1, -alphas]
ma = np.r_[1, betas]

ar2 = smt.arma_generate_sample(ar=ar, ma=ma, nsample=n) 
tsplot(ar2, lags=12,title="AR(2) process")


#AR(2) process -- has ACF tailing out and PACF cutting off at lag=2
# Simulate an MA(1) process
n = int(1000)
# set the AR(p) alphas equal to 0
alphas = np.array([0.])
betas = np.array([0.8])
# add zero-lag and negate alphas
ar = np.r_[1, -alphas]
ma = np.r_[1, betas]
ma1 = smt.arma_generate_sample(ar=ar, ma=ma, nsample=n) 
limit=12
tsplot(ma1, lags=limit,title="MA(1) process")
 

#MA(1) process -- has ACF cut off at lag=1
# Simulate MA(2) process with betas 0.6, 0.4
n = int(1000)
alphas = np.array([0.])
betas = np.array([0.6, 0.4])
ar = np.r_[1, -alphas]
ma = np.r_[1, betas]

ma3 = smt.arma_generate_sample(ar=ar, ma=ma, nsample=n)
tsplot(ma3, lags=12,title="MA(2) process")


#MA(2) process -- has ACF cut off at lag=2
# Simulate an ARMA(2, 2) model with alphas=[0.5,-0.25] and betas=[0.5,-0.3]
max_lag = 12
n = int(5000) # lots of samples to help estimates
burn = int(n/10) # number of samples to discard before fit

alphas = np.array([0.8, -0.65])
betas = np.array([0.5, -0.7])
ar = np.r_[1, -alphas]
ma = np.r_[1, betas]

arma22 = smt.arma_generate_sample(ar=ar, ma=ma, nsample=n, burnin=burn)
tsplot(arma22, lags=max_lag,title="ARMA(2,2) process")

 

 

#Now things get a little hazy. Its not very clear/straight-forward.

"""
A nifty summary of the above plots:

ACF Shape Indicated Model
Exponential, decaying to zero: Autoregressive model. Use the partial autocorrelation plot to identify the order of the autoregressive model
Alternating positive and negative, decaying to zero Autoregressive model. : Use the partial autocorrelation plot to help identify the order.
One or more spikes, rest are essentially zero: Moving average model, order identified by where plot becomes zero.
Decay, starting after a few lags : Mixed autoregressive and moving average (ARMA) model.
All zero or close to zero : Data are essentially random.
High values at fixed intervals : Include seasonal autoregressive term.
No decay to zero : Series is not stationary
"""


#Let's use a systematic approach to finding the order of AR and MA processes.
# pick best order by aic 
# smallest aic value wins
best_aic = np.inf 
best_order = None
best_mdl = None

rng = range(5)
for i in rng:
    for j in rng:
        try:
            tmp_mdl = smt.ARMA(arma22, order=(i, j)).fit(method='mle', trend='nc')
            tmp_aic = tmp_mdl.aic
            if tmp_aic < best_aic:
                best_aic = tmp_aic
                best_order = (i, j)
                best_mdl = tmp_mdl
        except: continue


print('aic: {:6.5f} | order: {}'.format(best_aic, best_order))
 

"""
We've correctly identified the order of the simulated process as ARMA(2,2).
Lets use it for the sales time-series.
"""

#
# pick best order by aic 
# smallest aic value wins
best_aic = np.inf 
best_order = None
best_mdl = None

rng = range(5)
for i in rng:
    for j in rng:
        try:
            tmp_mdl = smt.ARMA(new_ts.values, order=(i, j)).fit(method='mle', trend='nc')
            tmp_aic = tmp_mdl.aic
            if tmp_aic < best_aic:
                best_aic = tmp_aic
                best_order = (i, j)
                best_mdl = tmp_mdl
        except: continue


print('aic: {:6.5f} | order: {}'.format(best_aic, best_order))

 

# Simply use best_mdl.predict() to predict the next values

# adding the dates to the Time-series as index
ts=sales.groupby(["date_block_num"])["item_cnt_day"].sum()
ts.index=pd.date_range(start = '2013-01-01',end='2015-10-01', freq = 'MS')
ts=ts.reset_index()
ts.head()



 

"""
Prophet:
Recently open-sourced by Facebook research. It's a very promising tool, that is often a very handy and quick solution to the frustrating flatline :P

FLATLINE

Sure, one could argue that with proper pre-processing and carefully tuning the parameters the above graph would not happen.

But the truth is that most of us don't either have the patience or the expertise to make it happen.

Also, there is the fact that in most practical scenarios- there is often a lot of time-series that needs to be predicted. Eg: This competition. It requires us to predict the next month sales for the Store - item level combinations which could be in the thousands.(ie) predict 1000s of parameters!

Another neat functionality is that it follows the typical sklearn syntax.
At its core, the Prophet procedure is an additive regression model with four main components:

A piecewise linear or logistic growth curve trend. Prophet automatically detects changes in trends by selecting changepoints from the data.

A yearly seasonal component modeled using Fourier series.

A weekly seasonal component using dummy variables.

A user-provided list of important holidays.

"""