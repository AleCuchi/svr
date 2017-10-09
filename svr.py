import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import quandl
from sklearn import preprocessing, cross_validation, svm, neural_network


class DataGet():
    def __init__(self):
        self.arr_df = {}
        self.forecast_out = {}
        self.forecast_col = {}
        self.X = self.y = self.x_train = self.y_train = self.x_test = self.y_test = {}

    def get_data(self, arr_company, start_date='2016-01-01', end_date='2016-31-12'):
        # Pegando informação do site
        for i in arr_company:
            self.arr_df[arr_company] = quandl.get(i, start_date=start_date, end_date=end_date)

    def calcula_percentual(self, col1, col2):

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


# df_google = quandl.get("WIKI/GOOGL",start_date='2016-01-01',end_date='2017-01-01')

df_google = pd.DataFrame(np.load("/home/alexandre/Documents/file.npy"))

df_google = df_google[["Adj. Open", "Adj. High", "Adj. Low", "Adj. Close", "Adj. Volume"]]

df_google["Perc_High"] = (df_google["Adj. High"] - df_google["Adj. Close"]) / df_google["Adj. Close"] * 100
df_google["Pert_Low"] = (df_google["Adj. Close"] - df_google["Adj. Open"]) / df_google["Adj. Open"] * 100
df_google["media_10"] = df_google["Adj. Close"].rolling(window=10).mean()
df_google.drop(df_google.index[range(10)])

df_google = df_google[["Adj. Close", "Perc_High", "Pert_Low", "media_10", "Adj. Volume"]]

forecast_col = "Adj. Close"
forecast_out = int(math.ceil(0.0001 * len(df_google)))
df_google["label"] = df_google[forecast_col].shift(-forecast_out)

df_google.fillna(-99999, inplace=True)
df_google.dropna(inplace=True)

X = np.array(df_google.drop(["label"], 1))
y = np.array(df_google["label"])
X = preprocessing.scale(X)
y = np.array(df_google["label"])

Valor_treino, Valor_teste, Resposta_treino, Resposta_teste = cross_validation.train_test_split(X, y, test_size=0.2)

clf_svr = svm.SVR(kernel="linear")
clf_neuralnet = neural_network.MLPRegressor()
clf_svr.fit(Valor_treino, Resposta_treino)
clf_neuralnet.fit(Valor_treino, Resposta_treino)
# prediction = clf.predict(Valor_teste)
accuracySVM = clf_svr.predict(Valor_teste)
# accuracyNN = clf_neuralnet.predict(Valor_teste)



# grafico = pd.DataFrame(accuracySVM).tail(60)
# grafico.cumsum()
# plt.figure(); grafico.plot(); plt.legend(loc="best")
valor_grafico = 300
plt.plot(range(len(accuracySVM[:valor_grafico])), accuracySVM[:valor_grafico], '',
         range(len(Resposta_teste[:valor_grafico])), Resposta_teste[:valor_grafico], '')
plt.show()

print("mostro")

# print(prediction)


# print(x_test)
# WIKI/AMZN   - Amazon
# WIKI/AAPL   - Apple
# WIKI/DELL   - Dell
# WIKI/MSFT   - Microsoft
# WIKI/CBS    - CBS
# WIKI/F      - Ford
# WIKI/FB     -Facebook
# WIKI/MCD     - Mc Donalds
#WIKI/AVP     - Avon
