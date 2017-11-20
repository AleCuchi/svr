import matplotlib.pyplot as plt
import numpy as np
import quandl
import pandas as pd
import time
import math
from sklearn import preprocessing, cross_validation, svm, neural_network

def normaliza(vetor, media=10):
    new_vetor = []
    if len(vetor) >2*media:
        for i in range(len(vetor)-media):
            new_vetor.append(sum(vetor[i:i+media-1])/media)
    return new_vetor


def split_list(a_list,perc=0.5):
    half = math.ceil(len(a_list)*perc)
    return a_list[:-half], a_list[-half:]

plt.style.use("ggplot")
# variaveis de controle


janela_media = 10
amostra_dados = 0.1
forecast_out = 6
forecast_col = "Adj. Close"
valor_grafico = 50
item_verificado=[]

# WIKI/AMZN   - Amazon
# WIKI/AAPL   - Apple
# WIKI/DELL   - Dell
# WIKI/MSFT   - Microsoft
# WIKI/CBS    - CBS
# WIKI/F      - Ford
# WIKI/FB     -Facebook
# WIKI/MCD    - Mc Donalds
# WIKI/AVP    - Avon


df = []
# df.append(quandl.get("WIKI/GOOGL", start_date="20150102",end_date="20170102"))
# df.append(quandl.get("WIKI/CBS", start_date="20150102",end_date="20170102"))
# df.append(quandl.get("WIKI/AVP", start_date="20150102",end_date="20170102"))
# df.append(quandl.get("WIKI/MCD", start_date="20150102",end_date="20170102"))
# #
# df[0].to_pickle('/root/Documents/file_google.pkl')
# df[1].to_pickle('/root/Documents/file_cbs.pkl')
# df[2].to_pickle('/root/Documents/file_avon.pkl')
# df[3].to_pickle('/root/Documents/file_mcd.pkl')
df.append(pd.read_pickle('/root/Documents/file_google.pkl'))
df.append(pd.read_pickle('/root/Documents/file_cbs.pkl'))
df.append(pd.read_pickle('/root/Documents/file_avon.pkl'))
df.append(pd.read_pickle('/root/Documents/file_mcd.pkl'))

df_google = df[3]
# df_google = pd.DataFrame(np.load("/home/alexandre/Documents/file.npy"))

df_google = df_google[["Adj. Open", "Adj. High", "Adj. Low", "Adj. Close", "Adj. Volume"]]

df_google["Perc_High"] = (df_google["Adj. High"] - df_google["Adj. Close"]) / df_google["Adj. Close"]*100
df_google["Pert_Low"] = (df_google["Adj. Close"] - df_google["Adj. Open"]) / df_google["Adj. Open"]*100
df_google["media_10"] = df_google["Adj. Close"].rolling(window=janela_media).mean()
# df_google.drop(df_google.index[range(janela_media)])

df_google = df_google[["Adj. Close", "Perc_High", "Pert_Low", "media_10", "Adj. Volume"]]

 # int(math.ceil(0.001*len(df_google)))
df_google = df_google[janela_media:]

df_google["label"] = df_google[forecast_col].shift(-forecast_out)

df_google = df_google[["Adj. Close", "Perc_High", "Pert_Low", "media_10", "Adj. Volume", "label"]]

df_google = df_google[df_google.label.notnull()]

X = np.array(df_google.drop(["label"], 1))
X_A = preprocessing.scale(X)

y = np.array(df_google["label"])
X_A, Valor_teste1 = split_list(X_A,amostra_dados)
y, Resposta_teste1 = split_list(y,amostra_dados)
Valor_treino, Valor_teste, Resposta_treino, Resposta_teste = cross_validation.train_test_split(X_A, y, test_size=amostra_dados)

scores=[]
tempos = []


Vaereee = True
if Vaereee : # Dados Lineares
    clf_svr = svm.SVR(kernel="linear", C=50)
    clf_svr.fit(Valor_treino, Resposta_treino)
    scores = clf_svr.score(Valor_teste1, Resposta_teste1)
    predict = clf_svr.predict(Valor_teste1)
    print(scores)
    predict_normal = normaliza(predict)
    Resposta_teste_normal = normaliza(Resposta_teste1)
else: # Dados Aleatorios
    clf_svr = svm.SVR(kernel="linear", C=50)
    clf_svr.fit(Valor_treino, Resposta_treino)
    scores = clf_svr.score(Valor_teste, Resposta_teste)
    predict = clf_svr.predict(Valor_teste)
    print(scores)
    predict_normal = predict
    Resposta_teste_normal = Resposta_teste

valores_svm = plt.plot(predict_normal,label='SVM')
valores_real = plt.plot(Resposta_teste_normal,label='Real')


plt.legend(loc='upper center', bbox_to_anchor=(1, 0.5))
plt.title("McDonald's")

# plt.plot(vetor,predict,'blue', vetor, Resposta_teste, 'red')
plt.show()

# plt.plot(vetor_contador1,predict,'blue',vetor_contador1,verificacao[:len(predict)],'red')
# plt.show()
#print(df_google.tail())


# plt.show()

