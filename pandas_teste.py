import matplotlib.pyplot as plt
import numpy as np
import quandl
import time, math
from sklearn import preprocessing, cross_validation, svm, neural_network
plt.style.use("ggplot")
# variaveis de controle
janela_media = 10               # Define a janela de média
forecast_col = "Adj. Close"
forecast_out = 2
valor_grafico = 50

#Buscando informações
df = quandl.get("WIKI/CBS")

# df_google = pd.DataFrame(np.load("/home/alexandre/Documents/file.npy"))

# Ajustando dados para trenamento
df = df[["Adj. Open", "Adj. High", "Adj. Low", "Adj. Close", "Adj. Volume"]]

df["Perc_High"] = (df["Adj. High"] - df["Adj. Close"]) / df["Adj. Close"] * 100
df["Perc_Low"] = (df["Adj. Close"] - df["Adj. Open"]) / df["Adj. Open"] * 100
df["media_10"] = df["Adj. Close"].rolling(window=janela_media).mean()
# df.drop(df.index[range(janela_media)])

# Selecionando campos para utilização
df = df[["Adj. Close", "Perc_High", "Perc_Low", "media_10", "Adj. Volume"]]

# Definindo tempo de superioridade da previsão
#forecast_out = 40   int(math.ceil(0.001*len(df)))

# Retirando valores nulos causados pela média
df = df[janela_media:]

# Variaveis de verificação
predict_svm = []
predict_ann = []
accuracy_svm = 0
accuracy_ann = 0
tempo_execucao_svm = 0
tempo_execucao_ann = 0
tempo_treino_svm = 0
tempo_treino_ann = 0

# Início do algoritmo de treino e verificação de resultados

df["label"] = df[forecast_col].shift(-forecast_out)

df = df[["Adj. Close", "Perc_High", "Perc_Low", "media_10", "Adj. Volume", "label"]]
df = df[:-forecast_out]

X = np.array(df.drop(["label"], 1))
X_A = preprocessing.scale(X)

y = np.array(df["label"])

valor_treino, valor_teste, resposta_treino, resposta_teste = cross_validation.train_test_split(X_A, y, test_size=0.2)

# divisao_treino = math.ceil(len(X_A) * 0.2)
#
# valor_treino, valor_teste = X_A[:-divisao_treino], X_A[-divisao_treino:]
# resposta_treino, resposta_teste = y[:-divisao_treino], y[-divisao_treino:]
valor_inicial = 200
intervalo = 500
for i in range(valor_inicial,valor_inicial+intervalo):
    print(i , "de", valor_inicial+ intervalo, sep=' ')
    # Usando SVM
    clf_svm = svm.SVR(kernel="linear")
    ini_svm = time.time()
    clf_svm.fit(valor_treino[:i], resposta_treino[:i])
    fim_svm = time.time()

    # Usando ANN
    clf_ann = neural_network.MLPRegressor(activation='logistic')
    ini_ann = time.time()
    clf_ann.fit(valor_treino[:i], resposta_treino[:i])
    fim_ann = time.time()

    # Calculando precisão SVM
    var_svm = clf_svm.score(valor_teste, resposta_teste)
    accuracy_svm += var_svm
    # predict_svm.append(var_svm)

    # Realiza previsão SVM
    ini_svm_predic = time.time()
    svm_previsto = clf_svm.predict(valor_teste)
    fim_svm_predict = time.time()

    #Calcula precisão ANN
    var_ann = clf_ann.score(valor_teste, resposta_teste)
    accuracy_ann += var_ann
    # predict_ann.append(var_ann)

    # Realiza Previsão ANN
    ini_ann_predic = time.time()
    ann_previsto = clf_ann.predict(valor_teste)
    fim_ann_predict = time.time()

    var_svm = fim_svm - ini_svm
    var_ann = fim_ann - ini_ann
    tempo_treino_svm += var_svm
    tempo_treino_ann += var_ann

    predict_svm.append(var_svm)
    predict_ann.append(var_ann)

    tempo_execucao_svm += fim_svm - ini_svm
    tempo_execucao_ann += var_ann



accuracy_plot_svm = plt.plot(predict_svm,label='SVM')
accuracy_plot_ann = plt.plot(predict_ann,label='ANN')
plt.legend(loc='upper center', bbox_to_anchor=(1, 0.5))


print(accuracy_ann/intervalo, tempo_execucao_ann/intervalo, tempo_treino_ann, sep=' - ')
print(accuracy_svm/intervalo, tempo_execucao_svm/intervalo, tempo_treino_svm, sep=' - ')
# time_plot = plt
# time_plot.plot(tempo_execucao_ann, 'red', tempo_execucao_svm, 'blue')

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
