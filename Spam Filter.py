# -*- coding: utf-8 -*-
"""
Created on Tue May 11 17:48:15 2021

@author: Ayush
"""

import pandas as pd
import numpy as np

dataset=pd.read_csv('emails.csv')

X= dataset['text']
y=dataset['spam']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=10)

from sklearn.feature_extraction.text import CountVectorizer
spam_fil = CountVectorizer(stop_words='english')
X_train=spam_fil.fit_transform(X_train).toarray()
X_test=spam_fil.transform(X_test).toarray()

print(spam_fil.get_feature_names())

from sklearn.neighbors import KNeighborsClassifier  #KNeighbors
kneigh = KNeighborsClassifier(n_neighbors=5)
kneigh.fit(X_train,y_train)

pred_test= kneigh.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, pred_test))


from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(X_train,y_train)
pred_test_nb= nb.predict(X_test)
print(accuracy_score(y_test,pred_test_nb))

import joblib
joblib.dump(spam_fil,'trained-countVectorizer.pkl')
joblib.dump(nb,'NaiveBayes-model.pkl')

email=['Subject: password  login kkindal  password marketcredit !']
X_e = spam_fil.transform(email).toarray()
pre = nb.predict(X_e)
if pre[0]==0:
    print("Spam")
else:
    print("Not Spam")
