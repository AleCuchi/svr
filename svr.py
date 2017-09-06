import quandl
import math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
import math

import numpy as np
import quandl
from sklearn import preprocessing, cross_validation, svm


class DataGet():
    def __init__(self):
        self.arr_df = {}
        self.forecast_out = {}
        self.forecast_col = {}
        self.X = self.y = self.x_train = self.y_train = self.x_test = self.y_test = {}

    def get_data(self, arr_company, start_date='1900-01-01', end_date='1900-01-01'):
        # Pegando informação do site
        for i in arr_company:
            self.arr_df[arr_company] = quandl.get(i, start_date=start_date, end_date=end_date)

    def formular_data(self):

        self.arr_df["WIKI/GOOGL"]["HL_PRC"] = (self.arr_df["WIKI/GOOGL"]["Adj. High"] -
                                               self.arr_df["WIKI/GOOGL"]["Adj. Close"]) / \
                                              self.arr_df["WIKI/GOOGL"]["Adj. Close"] * 100
        self.arr_df["WIKI/GOOGL"]["PRC_change"] = (self.arr_df["WIKI/GOOGL"]["Adj. Close"] -
                                                   self.arr_df["WIKI/GOOGL"]["Adj. Open"]) / \
                                                  self.arr_df["WIKI/GOOGL"]["Adj. Open"] * 100
        self.arr_df["WIKI/GOOGL"] = self.arr_df["WIKI/GOOGL"][["Adj. Close", "HL_PRC", "PRC_change", "Adj. Volume"]]

    '''
    PArametro Array no formato [[Codigo_compania1,[campo1,campo2...]],
                                [Codigo_compania2,[campo1,campo2...]]
    '''

    def select_data(self, arr_campos):
        for i in arr_campos:
            self.arr_df[i[0]] = self.arr_df[i[0]][[j for j in i[1]]]

    def set_data_config(self, arr_campos):
        self.forecast_col = "Adj. Close"
        for _ in arr_campos:
            self.arr_df[_].fillna(-99999, inplace=True)
            self.forecast_out = int(math.ceil(0.01 * len(self.arr_df[_])))
            self.arr_df[_]["label"] = self.arr_df[_][self.forecast_col].shift(-self.forecast_out)
            self.arr_df[_].dropna(inplace=True)

#_----------------------------------------------------------

    def separa_treino(self):

        for _ in self.arr_df:
            self.X[_] = np.array(self.arr_df[_].drop(["label"], 1))
            self.X[_] = preprocessing.scale(self.X)
            self.y[_] = np.array(self.arr_df[_]["label"])
            self.y[_] = np.array(self.arr_df[_]["label"])
            self.x_train[_], self.x_test[_], self.y_train[_], self.y_test[_] = \
                cross_validation.train_test_split(self.X[_], self.y[_], test_size=0.2)


df = DataGet(["WIKI/GOOGL", "FED/PC073164013_Q"])
clf = svm.SVR()
clf.fit(df.x_train, df.y_train)
prediction = clf.predict(df.x_test[-4:])
accuracy = clf.score(df.x_test[-4:], df.y_test[-4:])

print(accuracy)
print(df.x_test[""])
print(prediction)
# print(x_test)
# WIKI/AMZN   - Amazon
# WIKI/AAPL   - Apple
# WIKI/DELL   - Dell
# WIKI/MSFT   - Microsoft
# WIKI/CBS    - CBS
# WIKI/F      - Ford
# WIKI/FB     -Facebook
#WIKI/MCD    - Mc Donalds
