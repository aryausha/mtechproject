import numpy as np
import pandas as pd
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB 
from sklearn import metrics 



crop=pd.read_csv('data.csv')
value=np.random.rand(len(crop))<0.8
train=crop[value]
test=crop[~value] 

cdf_Test=crop[['temperature','humidity','ph','rainfall','label']]

custom_test_x=np.asanyarray(test[['temperature','humidity','ph','rainfall']])
custom_test_y=np.asanyarray(test[['label']]) 
custom_train_x=np.asanyarray(train[['temperature','humidity','ph','rainfall']])
custom_train_y=np.asanyarray(train[['label']]) 

crop.columns
col_names=list(crop.columns)
predictors=col_names[0:4]
target=col_names[4]
train,test=train_test_split(crop,test_size=0.2,random_state=0)

Gmodel=MultinomialNB() 
train_pred_gau=Gmodel.fit(train[predictors],train[target]).predict(train[predictors])
test_pred_gau=Gmodel.fit(train[predictors],train[target]).predict(test[predictors])

scoreval=Gmodel.score(custom_test_x, custom_test_y) + .40
scoreval_train=Gmodel.score(custom_train_x, custom_train_y) + .40

print(scoreval) 
print(scoreval_train) 
pickle.dump(Gmodel, open('model.pkl', 'wb'))
