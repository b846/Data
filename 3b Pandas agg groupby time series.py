#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 21:25:35 2020

@author: b
"""
import pandas as pd

# .agg
df = pd.DataFrame([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9],
                   [np.nan, np.nan, np.nan]],
                  columns=['A', 'B', 'C'])

#Aggregate these functions over the rows.

df.agg(['sum', 'min'])

df.agg({'A' : ['sum', 'min'], 'B' : ['min', 'max']})

df.agg("mean", axis="columns")


# .groupby
df = pd.DataFrame({'Animal': ['Falcon', 'Falcon',
                              'Parrot', 'Parrot'],
                   'Max Speed': [380., 370., 24., 26.]})

df.groupby(['Animal']).mean()

#Hierarchical Indexes
#We can groupby different levels of a hierarchical index using the level parameter:
arrays = [['Falcon', 'Falcon', 'Parrot', 'Parrot'],
          ['Captive', 'Wild', 'Captive', 'Wild']]

index = pd.MultiIndex.from_arrays(arrays, names=('Animal', 'Type'))

df = pd.DataFrame({'Max Speed': [390., 350., 30., 20.]},
                  index=index)

df.groupby(level=0).mean()

df.groupby(level="Type").mean()



# Rolling
"""
Rolling-Window Analysis of Time-Series Models
Rolling-window analysis of a time-series model assesses:
    The stability of the model over time. 
    A common time-series model assumption is that the coefficients are constant 
    with respect to time. Checking for instability amounts to examining whether 
    the coefficients are time-invariant..
    The forecast accuracy of the model.
    """
df = pd.DataFrame({'B': [0, 1, 2, np.nan, 4]})
print(df)

#Rolling sum with a window length of 2, using the ‘triang’ window type.
df.rolling(2, win_type='triang').sum()


#Rolling sum with a window length of 2, using the ‘gaussian’ window type (note how we need to specify std).
df.rolling(2, win_type='gaussian').sum(std=3)

#Rolling sum with a window length of 2, min_periods defaults to the window length.
df.rolling(2).sum()

#Same as above, but explicitly set the min_periods
df.rolling(2, min_periods=1).sum()

#A ragged (meaning not-a-regular frequency), time-indexed DataFrame
df = pd.DataFrame({'B': [0, 1, 2, np.nan, 4]},
                  index = [pd.Timestamp('20130101 09:00:00'),
                           pd.Timestamp('20130101 09:00:02'),
                           pd.Timestamp('20130101 09:00:03'),
                           pd.Timestamp('20130101 09:00:05'),
                           pd.Timestamp('20130101 09:00:06')])
df
"""Contrasting to an integer rolling window, 
this will roll a variable length window corresponding to the time period. 
The default for min_periods is 1.
"""
df.rolling('2s').sum()
