#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 19:40:39 2020

@author: b
https://www.kaggle.com/thebrownviking20/everything-you-can-do-with-a-time-series

    1. Introduction to date and time
        1.1 Importing time series data
        1.2 Cleaning and preparing time series data
        1.3 Visualizing the datasets
        1.4 Timestamps and Periods
        1.5 Using date_range
        1.6 Using to_datetime
        1.7 Shifting and lags
        1.8 Resampling
    2. Finance and Statistics
        2.1 Percent change
        2.2 Stock returns
        2.3 Absolute change in successive rows
        2.4 Comaring two or more time series
        2.5 Window functions
        2.6 OHLC charts
        2.7 Candlestick charts
        2.8 Autocorrelation and Partial Autocorrelation
    3. Time series decomposition and Random Walks
        3.1 Trends, Seasonality and Noise
        3.2 White Noise
        3.3 Random Walk
        3.4 Stationarity
    4. Modelling using statsmodels
        4.1 AR models
        4.2 MA models
        4.3 ARMA models
        4.4 ARIMA models
        4.5 VAR models
        4.6 State space methods
            4.6.1 SARIMA models
            4.6.2 Unobserved components
            4.6.3 Dynamic Factor models


"""

#1.1- Importing libraries
import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight') 
# Above is a special style template for matplotlib, highly useful for visualizing time series data
from pylab import rcParams
from plotly import tools
import plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff
import statsmodels.api as sm
from numpy.random import normal, seed
from scipy.stats import norm
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.arima_model import ARIMA
import math
from sklearn.metrics import mean_squared_error
chemin1='/home/b/Documents/Python/Data/Time series analysis/stock-time-series-20050101-to-20171231/'
chemin2='/home/b/Documents/Python/Data/Time series analysis/historical-hourly-weather-data/'
print(os.listdir('/home/b/Documents/Python/Data/Time series analysis/stock-time-series-20050101-to-20171231'))


# Option d'affichage
pd.options.display.max_columns = 7    #option d'affichage, none means no maximum value
pd.options.display.max_rows = 20

#Load of the data
os.chdir (chemin1)
google = pd.read_csv(chemin1 + 'GOOGL_2006-01-01_to_2018-01-01.csv', index_col='Date', parse_dates=['Date'])
google.head()

humidity = pd.read_csv(chemin2 + 'humidity.csv', index_col='datetime', parse_dates=['datetime'])
humidity.tail() 



#1.2 Cleaning and preparing time series data
"""
How to prepare data?
Google stocks data doesn't have any missing values but humidity data does have its fair 
share of missing values. It is cleaned using fillna() method with ffill parameter which 
propagates last valid observation to fill gaps
"""
humidity = humidity.iloc[1:]
humidity = humidity.fillna(method='ffill')
humidity.head()


#1.3 Visualizing the datasets¶
humidity["Kansas City"].asfreq('M').plot() # asfreq method is used to convert a time series to a specified frequency. Here it is monthly frequency.
plt.title('Humidity in Kansas City over time(Monthly frequency)')
plt.show()

google['2008':'2010'].plot(subplots=True, figsize=(10,12))
plt.title('Google stock attributes from 2008 to 2010')
plt.savefig('stocks.png')
plt.show()


#1.4 Timestamps and Periods
"""
What are timestamps and periods and how are they useful?
Timestamps are used to represent a point in time. Periods represent an interval in time. 
Periods can used to check if a specific event in the given period. 
They can also be converted to each other's form.
"""

# Creating a Timestamp
timestamp = pd.Timestamp(2017, 1, 1, 12)
timestamp

# Creating a period
period = pd.Period('2017-01-01')
period

# Checking if the given timestamp exists in the given period
period.start_time < timestamp < period.end_time

# Converting timestamp to period
new_period = timestamp.to_period(freq='H')
new_period

# Converting period to timestamp
new_timestamp = period.to_timestamp(freq='H', how='start')
new_timestamp




#1.5 Using date_range
"""
What is date_range and how is it useful?¶
date_range is a method that returns a fixed frequency datetimeindex. 
It is quite useful when creating your own time series attribute for pre-existing 
data or arranging the whole data around the time series attribute created by you.
"""
# Creating a datetimeindex with daily frequency
dr1 = pd.date_range(start='1/1/18', end='1/9/18')
dr1

# Creating a datetimeindex with monthly frequency
dr2 = pd.date_range(start='1/1/18', end='1/1/19', freq='M')
dr2

# Creating a datetimeindex without specifying start date and using periods
dr3 = pd.date_range(end='1/4/2014', periods=8)
dr3

# Creating a datetimeindex specifying start date , end date and periods
dr4 = pd.date_range(start='2013-04-24', end='2014-11-27', periods=3)
dr4




#1.6 Using to_datetime
"""pandas.to_datetime() is used for converting arguments to datetime. Here, a DataFrame is converted to a datetime series.
"""
df = pd.DataFrame({'year': [2015, 2016], 'month': [2, 3], 'day': [4, 5]})
df


df = pd.to_datetime(df)
df


#1.7 Shifting and lags
"""We can shift index by desired number of periods with an optional time frequency. 
This is useful when comparing the time series with a past of itself
"""

humidity["Vancouver"].asfreq('M').plot(legend=True)
shifted = humidity["Vancouver"].asfreq('M').shift(10).plot(legend=True)
shifted.legend(['Vancouver','Vancouver_lagged'])
plt.show()



#1.8 Resampling
"""
Upsampling - Time series is resampled from low frequency to high frequency(Monthly to daily frequency).
It involves filling or interpolating missing data

Downsampling - Time series is resampled from high frequency to low frequency
(Weekly to monthly frequency). It involves aggregation of existing data.
"""
# Let's use pressure data to demonstrate this
pressure = pd.read_csv(chemin2 + 'pressure.csv', index_col='datetime', parse_dates=['datetime'])
pressure.tail()

#Clean of the data
pressure = pressure.iloc[1:]
pressure = pressure.fillna(method='ffill')
pressure.tail()

pressure = pressure.fillna(method='bfill')
pressure.head()
"""First, we used ffill parameter which propagates last valid observation to fill gaps. 
Then we use bfill to propogate next valid observation to fill gaps."""

# We downsample from hourly to 3 day frequency aggregated using mean
pressure = pressure.resample('3D').mean()
pressure.head()

# Shape after resampling(downsampling)
pressure.shape

"""Much less rows are left. Now, we will upsample from 3 day frequency to daily frequency"""
pressure = pressure.resample('D').pad()
pressure.head()

# Shape after resampling(upsampling)
pressure.shape
"""Again an increase in number of rows. Resampling is cool when used properly."""


#2. Finance and statistics
#2.1 Percent change
google['Change'] = google.High.div(google.High.shift())
google['Change'].plot(figsize=(20,8))


#2.2 Stock returns
google['Return'] = google.Change.sub(1).mul(100)
google['Return'].plot(figsize=(20,8))

google.High.pct_change().mul(100).plot(figsize=(20,6)) # Another way to calculate returns



#2.3 Absolute change in successive rows
google.High.diff().plot(figsize=(20,6))


#2.4 Comaring two or more time series
"""We will compare 2 time series by normalizing them. 
This is achieved by dividing each time series element of all time series by the first element. 
This way both series start at the same point and can be easily compared."""

# We choose microsoft stocks to compare them with google
microsoft = pd.read_csv(chemin1 + 'MSFT_2006-01-01_to_2018-01-01.csv', index_col='Date', parse_dates=['Date'])

# Plotting before normalization
google.High.plot()
microsoft.High.plot()
plt.legend(['Google','Microsoft'])
plt.show()

# Normalizing and comparison
# Both stocks start from 100
normalized_google = google.High.div(google.High.iloc[0]).mul(100)
normalized_microsoft = microsoft.High.div(microsoft.High.iloc[0]).mul(100)
normalized_google.plot()
normalized_microsoft.plot()
plt.legend(['Google','Microsoft'])
plt.show()
"""You can clearly see how google outperforms microsoft over time."""


#2.5 Window functions
"""
Window functions are used to identify sub periods, calculates sub-metrics of sub-periods.
Rolling - Same size and sliding
Expanding - Contains all prior values
"""

# Rolling window functions
rolling_google = google.High.rolling('90D').mean()
google.High.plot()
rolling_google.plot()
plt.legend(['High','Rolling Mean'])
# Plotting a rolling mean of 90 day window with original High attribute of google stocks
plt.show()

# Expanding window functions
microsoft_mean = microsoft.High.expanding().mean()
microsoft_std = microsoft.High.expanding().std()
microsoft.High.plot()
microsoft_mean.plot()
microsoft_std.plot()
plt.legend(['High','Expanding Mean','Expanding Standard Deviation'])
plt.show()


#2.6 OHLC charts
"""
An OHLC chart is any type of price chart that shows the open, high, low and close price 
of a certain time period. Open-high-low-close Charts (or OHLC Charts) are used as a trading 
tool to visualise and analyse the price changes over time for securities, currencies, 
stocks, bonds, commodities, etc. 
OHLC Charts are useful for interpreting the day-to-day sentiment of the market and 
forecasting any future price changes through the patterns produced.

The y-axis on an OHLC Chart is used for the price scale, while the x-axis is the timescale. 
On each single time period, an OHLC Charts plots a symbol that represents two ranges: 
the highest and lowest prices traded, and also the opening and closing price on that 
single time period (for example in a day). On the range symbol, the high and low price 
ranges are represented by the length of the main vertical line. 
The open and close prices are represented by the vertical positioning of tick-marks 
that appear on the left (representing the open price) and on right (representing the close price)
sides of the high-low vertical line.

Colour can be assigned to each OHLC Chart symbol, to distinguish whether the market is "bullish" 
(the closing price is higher then it opened) or "bearish" (the closing price is lower then it opened).
"""

# OHLC chart of June 2008
trace = go.Ohlc(x=google['06-2008'].index,
                open=google['06-2008'].Open,
                high=google['06-2008'].High,
                low=google['06-2008'].Low,
                close=google['06-2008'].Close)
data = [trace]
iplot(data, filename='simple_ohlc')

# OHLC chart of 2008
trace = go.Ohlc(x=google['2008'].index,
                open=google['2008'].Open,
                high=google['2008'].High,
                low=google['2008'].Low,
                close=google['2008'].Close)
data = [trace]
iplot(data, filename='simple_ohlc')


#2.7 Candlestick charts
"""
This type of chart is used as a trading tool to visualise and analyse the price movements over time 
for securities, derivatives, currencies, stocks, bonds, commodities, etc. 
Although the symbols used in Candlestick Charts resemble a Box Plot, they function differently 
and therefore, are not to be confused with one another.

Candlestick Charts display multiple bits of price information such as the open price, close price,
 highest price and lowest price through the use of candlestick-like symbols. Each symbol represents 
 the compressed trading activity for a single time period (a minute, hour, day, month, etc). 
 Each Candlestick symbol is plotted along a time scale on the x-axis, 
 to show the trading activity over time.

The main rectangle in the symbol is known as the real body, which is used to display 
the range between the open and close price of that time period. While the lines extending
 from the bottom and top of the real body is known as the lower and upper shadows (or wick). 
 Each shadow represents the highest or lowest price traded during the time period represented. 
 When the market is Bullish (the closing price is higher than it opened),
 then the body is coloured typically white or green. But when the market is Bearish 
 (the closing price is lower than it opened), then the body is usually coloured either black or red.
 """
 # Candlestick chart of march 2008
trace = go.Candlestick(x=google['03-2008'].index,
                open=google['03-2008'].Open,
                high=google['03-2008'].High,
                low=google['03-2008'].Low,
                close=google['03-2008'].Close)
data = [trace]
iplot(data, filename='simple_candlestick')
 
 
 
#2.8 Autocorrelation and Partial Autocorrelation
"""
Autocorrelation - The autocorrelation function (ACF) measures 
how a series is correlated with itself at different lags.
Partial Autocorrelation - The partial autocorrelation function can be interpreted as 
a regression of the series against its past lags. 
The terms can be interpreted the same way as a standard linear regression, 
that is the contribution of a change in that particular lag while holding others constant.
"""

# Autocorrelation of humidity of San Diego
plot_acf(humidity["San Diego"],lags=25,title="San Diego")
plt.show()
"""As all lags are either close to 1 or at least greater than the confidence interval, they are statistically significant."""

 
#Partial Autocorrelation

# Partial Autocorrelation of humidity of San Diego
plot_pacf(humidity["San Diego"],lags=25)
plt.show()
"""Though it is statistically signficant, partial autocorrelation after first 2 lags is very low."""

# Partial Autocorrelation of closing price of microsoft stocks
plot_pacf(microsoft["Close"],lags=25)
plt.show()
"""Here, only 0th, 1st and 20th lag are statistically significant."""



#3. Time series decomposition and Random walks¶
#3.1. Trends, seasonality and noise
"""These are the components of a time series
Trend - Consistent upwards or downwards slope of a time series
 Seasonality - Clear periodic pattern of a time series(like sine funtion)
 Noise - Outliers or missing values"""

# Let's take Google stocks High for this
google["High"].plot(figsize=(16,8))


# Now, for decomposition...
rcParams['figure.figsize'] = 11, 9
decomposed_google_volume = sm.tsa.seasonal_decompose(google["High"],freq=360) # The frequncy is annual
figure = decomposed_google_volume.plot()
plt.show()
"""There is clearly an upward trend in the above plot.
   You can also see the uniform seasonal change.
    Non-uniform noise that represent outliers and missing values"""
    


#3.2. White noise
"""White noise has...
    Constant mean
    Constant variance
    Zero auto-correlation at all lags"""

# Plotting white noise
rcParams['figure.figsize'] = 16, 6
white_noise = np.random.normal(loc=0, scale=1, size=1000)
# loc is mean, scale is variance
plt.plot(white_noise)
"""
See how all lags are statistically insigficant as they lie inside the confidence interval(shaded portion).
"""



#4. Modelling using statstools
#4.1 AR models
"""
An autoregressive (AR) model is a representation of a type of random process; as such, 
it is used to describe certain time-varying processes in nature, economics, etc. 
The autoregressive model specifies that the output variable depends linearly 
on its own previous values and on a stochastic term (an imperfectly predictable term); 
thus the model is in the form of a stochastic difference equation.
AR(1) model

Rt = μ + ϕRt-1 + εt
As RHS has only one lagged value(Rt-1)this is called AR model of order 1 where μ is mean and ε is noise at time t

If ϕ = 1, it is random walk. Else if ϕ = 0, it is white noise. Else if -1 < ϕ < 1, it is stationary. If ϕ is -ve, there is men reversion. If ϕ is +ve, there is momentum.
AR(2) model

Rt = μ + ϕ1Rt-1 + ϕ2Rt-2 + εt
AR(3) model

Rt = μ + ϕ1Rt-1 + ϕ2Rt-2 + ϕ3Rt-3 + εt
"""
#Simulating AR(1) model

# AR(1) MA(1) model:AR parameter = +0.9
rcParams['figure.figsize'] = 12, 10
plt.subplot(4,1,1)
ar1 = np.array([1, -0.9]) # We choose -0.9 as AR parameter is +0.9
ma1 = np.array([1])
AR1 = ArmaProcess(ar1, ma1)
sim1 = AR1.generate_sample(nsample=1000)
plt.title('AR(1) model: AR parameter = +0.9')
plt.plot(sim1)
# We will take care of MA model later
# AR(1) MA(1) AR parameter = -0.9
plt.subplot(4,1,2)
ar2 = np.array([1, 0.9]) # We choose +0.9 as AR parameter is -0.9
ma2 = np.array([1])
AR2 = ArmaProcess(ar2, ma2)
sim2 = AR2.generate_sample(nsample=1000)
plt.title('AR(1) model: AR parameter = -0.9')
plt.plot(sim2)
# AR(2) MA(1) AR parameter = 0.9
plt.subplot(4,1,3)
ar3 = np.array([2, -0.9]) # We choose -0.9 as AR parameter is +0.9
ma3 = np.array([1])
AR3 = ArmaProcess(ar3, ma3)
sim3 = AR3.generate_sample(nsample=1000)
plt.title('AR(2) model: AR parameter = +0.9')
plt.plot(sim3)
# AR(2) MA(1) AR parameter = -0.9
plt.subplot(4,1,4)
ar4 = np.array([2, 0.9]) # We choose +0.9 as AR parameter is -0.9
ma4 = np.array([1])
AR4 = ArmaProcess(ar4, ma4)
sim4 = AR4.generate_sample(nsample=1000)
plt.title('AR(2) model: AR parameter = -0.9')
plt.plot(sim4)
plt.show()


#Forecasting a simulated model¶
model = ARMA(sim1, order=(1,0))
result = model.fit()
print(result.summary())
print("μ={} ,ϕ={}".format(result.params[0],result.params[1]))
#ϕ is around 0.9 which is what we chose as AR parameter in our first simulated model.



#Predicting the models
# Predicting simulated AR(1) model 
result.plot_predict(start=900, end=1010)
plt.show()

rmse = math.sqrt(mean_squared_error(sim1[900:1011], result.predict(start=900,end=999)))
print("The root mean squared error is {}.".format(rmse))