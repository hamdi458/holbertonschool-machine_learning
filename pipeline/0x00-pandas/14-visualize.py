#!/usr/bin/env python3

from datetime import date
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
df = df.drop(['Weighted_Price'], axis=1)
df = df.rename(columns={'Timestamp': 'Date'})

df['Date'] = pd.to_datetime(df['Date'], unit='s')
df = df.set_index('Date')
df = df[-(730 * 24 * 60):]

df['Close'].fillna(method='bfill', inplace=True)
df[['Open', 'High', 'Low']] = df[['Open', 'High', 'Low']].ffill()
df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)

df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

print(df)
aa = pd.DataFrame()
aa['Open'] = df['Open'].groupby(pd.Grouper(freq='D')).mean()
aa['High'] = df.High.resample('D').max()
aa['Low'] = df.Low.resample('D').min()
aa['Close'] = df['Close'].groupby(pd.Grouper(freq='D')).mean()
aa['Volume_(BTC)'] = df['Volume_(BTC)'].resample('D').sum()
aa['Volume_(Currency)'] = df['Volume_(Currency)'].resample('D').sum()

# Plotting data
aa.plot()

plt.show()
