import quandl, math
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection, svm  #IN THESE SCRIPTS, CROSS_VALIDATION MODULE HAS BEEN REPLACED BY MODEL_SELECTION DUE TO MODULE DEPRECATION
from sklearn.linear_model import LinearRegression


df = quandl.get("WIKI/GOOGL")

#print(df.head())
#print(df.tail())

df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]
#print(df.head())

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True) #VERY IMPORTANT 'na' stands for not available, you DONT WANT to filter out the na data
forecast_out = int(math.ceil(0.1*len(df))) #makes integer value of WHAT YOU WANT TO PREDICT in this case, 10% of the dataframe: using data that came 10% of ur timeframe to predict today...?
print(forecast_out)
df['label'] = df[forecast_col].shift(-forecast_out) #thats why we needed it to be an int.  This way each row will be the Adj Close price 10 days into the future.

df.dropna(inplace=True)

X = np.array(df.drop(['label'], 1)) #Generally features are capital X, labels lare lowercase y
y = np.array(df['label'])

X = preprocessing.scale(X)


df.dropna(inplace=True)
y = np.array(df['label'])

#print(len(X), len(y)) #checks we have the same array lengths

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

clf = LinearRegression(n_jobs=-1)  #clf means classifier n_jobs is for the amount of jobs u want to run parallely (-1 for no limit)...applications with deep learning
#clf =  svm.SVR(kernel='poly') #testing out SVM algorithm, then making kernel polynomoial
#A kernel is like a transformation against your data.  It's a way to frossly simplify the data, making processing much faster.  In the case of svm.SVR, the default is rdf, which is a type of kernel.
#there are other kernel opsitiopns such as linear, poly, rbf, sigmoid, precomputed or a callable.
#lets do a few
for k in ['linear','poly','rbg','sigmoid']:
    clf = svm.SVR(kernel=k)
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print(k, confidence)
#as we can see, linear kernel performed the best, then rbf, poly,...

clf.fit(X_train, y_train) #fit is synonymous with train
accuracy = clf.score(X_test, y_test) #score is synonymous with test
#accuracy/confidence can be computed as 2 different values
print(accuracy) #accuracy is the squared error

#print(df.tail())
#print(df.head())
