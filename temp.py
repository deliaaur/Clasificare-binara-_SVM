import pandas as pd
from sklearn import metrics
from sklearn import svm

#incarc data set-ul
f = pd.read_excel("data_set.xlsx")

#impart data_setul in 75% train si 25%test
X_train=f.iloc[0:74,:5].values
Y_train=f.iloc[0:74,5:6].values

X_test=f.iloc[75:,:5].values
Y_test=f.iloc[75:,5:6].values

#variez valoarea costului
cost=[1/32, 1/8,1/2, 2, 8, 32, 128]

#antrenez algoritmul pentru fiecare valoare a costului 
for i in range(len(cost)):
    clf=svm.SVC(kernel='rbf',C=cost[i],gamma=1)
    clf.fit(X_train,Y_train.ravel())
    predictie=clf.predict(X_test) #realizez o predictie pentru antrenamentul facut


count=0

for j in range(len(Y_test)):
    if Y_test[j]==predictie[j]: #compar valorile de testare cu cele obtinute la predictie
        count=count+1

print('Pentru costul ' + str(cost[i]) + ' acuratetea predictiei este ' + str((count/len(Y_test))*100) + '%')
#Folosind functia din metrics pentru acuratete obtinem acelasi rezultat print("Accuracy:",metrics.accuracy_score(Y_test, predictie))    