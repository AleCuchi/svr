import pandas as pd
import quandl as qndl
import math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

df = qndl.get("WIKI/GOOGL")
df = df[["Adj. Open", "Adj. High", "Adj. Low", "Adj. Close", "Adj. Volume"]]
df["HL_PRC"] = (df["Adj. High"] - df["Adj. Close"])/ df["Adj. Close"] * 100
df["PRC_change"] = (df["Adj. Close"] - df["Adj. Open"])/ df["Adj. Open"] * 100

df = df[["Adj. Close", "HL_PRC", "PRC_change", "Adj. Volume"]]

forecast_col = "Adj. Close"
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.00001*len(df)))

df["label"] = df[forecast_col].shift(-forecast_out)

df.dropna(inplace=True)

#_----------------------------------------------------------

X = np.array(df.drop(["label"], 1))
y = np.array(df["label"])
X = preprocessing.scale(X)
y = np.array(df["label"])

x_train, x_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

clf = svm.SVR()
clf.fit(x_train, y_train)
prediction = clf.predict(x_test)
accuracy = clf.score(x_test, y_test)

print(accuracy)
print(prediction)
