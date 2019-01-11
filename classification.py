import cv2
import numpy as np
from scipy import sparse
from sklearn.svm import SVR
from sklearn.svm import SVC


diva_n = cv2.imread("img/divadepressao.jpg")
filhos_n = cv2.imread("img/filhosdagravida.jpg")
tested1 = cv2.imread("img/teste1.jpg")
tested2 = cv2.imread("img/teste2.jpg")
testefg = cv2.imread("img/testefg.jpg")


#print(diva_n.shape)
#print(filhos_n.shape)
#print(tested1.shape)
#print(tested2.shape)
#print(testefg.shape)

diva = cv2.resize(diva_n, (10,10))
filhos = cv2.resize(filhos_n, (10,10))
teste1 = cv2.resize(tested1, (10,10))
teste2 = cv2.resize(tested2, (10,10))
testef3 = cv2.resize(testefg, (10,10))

X = np.concatenate((diva,filhos), axis=0)

y = [1, 2]

y = np.array(y)
Y = y.reshape(-1)

X = X.reshape(len(y), -1)

clf_lin = SVC(kernel='linear')

clf_lin.fit(X,Y)

predicao = clf_lin.predict(teste2.reshape(1,-1))
score = clf_lin.score(X,Y)

print(predicao)
print(score)

if predicao == 1:
	resultado = diva_n
if predicao == 2:
	resultado = filhos_n

cv2.imshow("Resultado", resultado)
cv2.imshow("Teste", tested2)
cv2.waitKey(0)
