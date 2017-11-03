import matplotlib.pyplot as plt
import numpy as np
import quandl
import time
from sklearn import preprocessing, cross_validation, svm, neural_network
plt.style.use("ggplot")
# variaveis de controle


janela_media = 10
forecast_col = "Adj. Close"

valor_grafico = 50
df_google = quandl.get("WIKI/GOOGL", start_date="20140101",end_date="20160101")

# df_google = pd.DataFrame(np.load("/home/alexandre/Documents/file.npy"))

df_google = df_google[["Adj. Open", "Adj. High", "Adj. Low", "Adj. Close", "Adj. Volume"]]

df_google["Perc_High"] = (df_google["Adj. High"] - df_google["Adj. Close"]) / df_google["Adj. Close"] * 100
df_google["Pert_Low"] = (df_google["Adj. Close"] - df_google["Adj. Open"]) / df_google["Adj. Open"] * 100
df_google["media_10"] = df_google["Adj. Close"].rolling(window=janela_media).mean()
# df_google.drop(df_google.index[range(janela_media)])






df_google = df_google[["Adj. Close", "Perc_High", "Pert_Low", "media_10", "Adj. Volume"]]

forecast_out = 40  # int(math.ceil(0.001*len(df_google)))
df_google = df_google[janela_media:]
predict=[]
accuracySVM =[]
real=[]
for i in range(forecast_out):
    df_google["label"] = df_google[forecast_col].shift(-1)

    df_google = df_google[["Adj. Close", "Perc_High", "Pert_Low", "media_10", "Adj. Volume", "label"]]
    df_google = df_google[:-1]

    # df_google.fillna(-99999, inplace=True)
    # df_google.dropna(inplace=True)

    X = np.array(df_google.drop(["label"], 1))
    X_A = preprocessing.scale(X)

    y = np.array(df_google["label"])

    Valor_treino, Valor_teste, Resposta_treino, Resposta_teste = cross_validation.train_test_split(X_A, y, test_size=0.2)
    ini = time.time()
    clf_svr = svm.SVR(kernel="linear")
    clr_svr = neural_network.MLPRegressor()
    clf_svr.fit(Valor_treino, Resposta_treino)
    fim = time.time()
    accuracySVM.append(clf_svr.score(Valor_teste, Resposta_teste))
    predict.append(clf_svr.predict(Valor_teste)[0])
    real.append(Resposta_teste[0])
    print(fim-ini)

# accuracyNN = clf_neuralnet.predict(Valor_teste)
print(accuracySVM, predict, sep="\n")
plt.plot(predict,'red',real,'blue')
plt.show()
#print(df_google.tail())
neural_network.ML
forecast_out = 0
# plt.plot([i for i in range(forecast_out)], predict[-forecast_out:], 'red',
#          [j for j in range(forecast_out)], Resposta_teste[-forecast_out:], 'blue')
# plt.plot([i for i in predict], 'red',
#          [j for j in Resposta_teste], 'blue')
# plt.show()

# WIKI/AMZN   - Amazon
# WIKI/AAPL   - Apple
# WIKI/DELL   - Dell
# WIKI/MSFT   - Microsoft
# WIKI/CBS    - CBS
# WIKI/F      - Ford
# WIKI/FB     -Facebook
# WIKI/MCD    - Mc Donalds
# WIKI/AVP    - Avon
