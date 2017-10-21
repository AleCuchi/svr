import matplotlib.pyplot as plt
import numpy as np
import quandl
from sklearn import preprocessing, cross_validation, svm

# variaveis de controle

janela_media = 10
forecast_col = "Adj. Close"

valor_grafico = 50
df_google = quandl.get("WIKI/GOOGL", start_date="20090101")

# df_google = pd.DataFrame(np.load("/home/alexandre/Documents/file.npy"))

df_google = df_google[["Adj. Open", "Adj. High", "Adj. Low", "Adj. Close", "Adj. Volume"]]

df_google["Perc_High"] = (df_google["Adj. High"] - df_google["Adj. Close"]) / df_google["Adj. Close"] * 100
df_google["Pert_Low"] = (df_google["Adj. Close"] - df_google["Adj. Open"]) / df_google["Adj. Open"] * 100
df_google["media_10"] = df_google["Adj. Close"].rolling(window=janela_media).mean()
# df_google.drop(df_google.index[range(janela_media)])

df_google = df_google[["Adj. Close", "Perc_High", "Pert_Low", "media_10", "Adj. Volume"]]

forecast_out = 40  # int(math.ceil(0.001*len(df_google)))

df_google["label"] = df_google[forecast_col].shift(forecast_out)

df_google = df_google[["Adj. Close", "Perc_High", "Pert_Low", "media_10", "Adj. Volume", "label"]]
df_google = df_google[forecast_out if forecast_out > janela_media else janela_media:]
# df_google.fillna(-99999, inplace=True)
# df_google.dropna(inplace=True)

X = np.array(df_google.drop(["label"], 1))
X_A = preprocessing.scale(X)

y = np.array(df_google["label"])

Valor_treino, Valor_teste, Resposta_treino, Resposta_teste = cross_validation.train_test_split(X_A, y, test_size=0.2)
# tamanho_teste = math.ceil(0.2*len(X_A))
# Valor_treino, Valor_teste = X_A[:-tamanho_teste], X_A[-tamanho_teste:]
# Resposta_treino, Resposta_teste = y[:-tamanho_teste], y[:-tamanho_teste]

clf_svr = svm.SVR(kernel="linear")
clf_svr.fit(Valor_treino, Resposta_treino)


# print(clf_neuralnet.score(, np.array([1007.87])))
accuracySVM = clf_svr.score(Valor_teste[-forecast_out:], Resposta_teste[-forecast_out:])
predict = clf_svr.predict(Valor_teste[-forecast_out:])
# accuracyNN = clf_neuralnet.predict(Valor_teste)
print(predict, accuracySVM, Resposta_teste[-forecast_out:], sep='\n')
print(df_google.tail())
forecast_out = 40
plt.plot([i for i in range(forecast_out)], predict[-forecast_out:], 'red',
         [j for j in range(forecast_out)], Resposta_teste[-forecast_out:], 'blue')
plt.show()

# WIKI/AMZN   - Amazon
# WIKI/AAPL   - Apple
# WIKI/DELL   - Dell
# WIKI/MSFT   - Microsoft
# WIKI/CBS    - CBS
# WIKI/F      - Ford
# WIKI/FB     -Facebook
# WIKI/MCD    - Mc Donalds
# WIKI/AVP    - Avon
