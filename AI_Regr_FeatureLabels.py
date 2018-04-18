import quandl, math
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression


df = quandl.get("WIKI/GOOGL")
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True) #VERY IMPORTANT 'na' stands for not available, you DONT WANT to filter out the na data

forecast_out = int(math.ceil(0.1*len(df))) #makes integer value of WHAT YOU WANT TO PREDICT in this case, 10% of the dataframe: using data that came 10% of ur timeframe to predict today...?

df['label'] = df[forecast_col].shift(-forecast_out) #thats why we needed it to be an int.  This way each row will be the Adj Close price 10 days into the future.
df.dropna(inplace=True)
print(df.tail())
#print(df.head())
