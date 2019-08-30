# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 21:48:18 2019

@author: Jesus kid
"""

import os
os.chdir(r'C:\Users\Jesus kid\Desktop\ML')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('power plant.csv' )


X=df.drop('EP', axis=1)
y=df['EP']
y=pd.DataFrame(y)


''' Preprocessing- standard-scaling'''
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_scaled = sc_X.fit_transform(X)
X_scaled=pd.DataFrame(X_scaled)

sc_y = StandardScaler()
y_scaled = sc_y.fit_transform(y)
y_scaled=pd.DataFrame(y_scaled)


from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE

X_scaled=X_scaled.values
y_scaled=y_scaled.values

''' get the number of variables that gives u max adjust R sq score,
 default selector.score in linear regression = R2 '''
estimator = LinearRegression() #use regression model for regression problem
list_r2=[]
max_r2 = 0
for i in range(1,len(X_scaled[0])+1):
    selector = RFE(estimator, i, step=1)
    selector = selector.fit(X_scaled, y_scaled)
    adj_r2 = 1 - ((len(X_scaled)-1)/(len(X_scaled)-i-1))*(1-selector.score(X_scaled, y_scaled))
    list_r2.append(adj_r2)# mse = 
    if max_r2 < adj_r2:
        sel_features2 = selector.support_ # 12 features selected exactly to give max adj r2
        max_r2 = adj_r2
        

''' optimal features to use'''        
X = X_scaled[:,sel_features2]
X = pd.DataFrame(X, columns=('T', 'V', 'AP', 'RH'))

y = pd.DataFrame(y_scaled, columns=[('EP')])


'''correlation matrix between X variables'''
cor_matrix=X.astype(float).corr(method='pearson')


'''split data'''
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1/3,random_state=0)

 







'''---------------------------------------------------------------------------------------------'''
'''fit linear regression to the model'''
from sklearn.linear_model import LinearRegression
lr1 = LinearRegression()
lr1.fit(X_train, y_train)
y_pred = lr1.predict(X_test)
print(lr1.coef_, lr1.intercept_) 

'''R2 and adjusted R2 , and rmse'''
from sklearn.metrics import mean_squared_error
import math
r2=lr1.score(X_test,y_test)
print(r2)
adj_r2 = 1 - ((len(X_test)-1)/(len(X_test)-i-1))*(1-lr1.score(X_test, y_test))
print(adj_r2)

mse=mean_squared_error(y_test, y_pred) #biased mean
rmse = math.sqrt(mse)
print(rmse)

'''summary table for coefficients'''
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression as lr
X2_train=sm.add_constant(X_train) # add a column of 1 beside x col
ols=sm.OLS(y_train.astype(float),X2_train.astype(float))# ordinary least square = linear regression
lr=ols.fit() 
print(lr.summary())



'''Cross validation'''
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression as lr
from sklearn.metrics import SCORERS
SCORERS.keys()

kf= KFold(n_splits=5, shuffle= True, random_state=1)
lr= lr()


'''Cross validation score (R2 for test data, and full data)'''
r2score=cross_val_score(lr1,X_test,y_test, cv=kf, scoring='r2')
print(r2score.mean())
r2score_b=cross_val_score(lr,X,y, cv=kf, scoring='r2')
print(r2score_b.mean())

'''Cross validation score (RMSE for test data, and full data)'''
RMSE=np.sqrt(-cross_val_score(lr1,X_test,y_test, cv=kf, scoring='neg_mean_squared_error'))
print(RMSE.mean())
RMSE_b=np.sqrt(-cross_val_score(lr,X,y, cv=kf, scoring='neg_mean_squared_error'))
print(RMSE_b.mean())










'''------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------'''
'''fit knn to the model'''
from sklearn.neighbors import KNeighborsRegressor #KNeighborsRegressor if linear regression
knn = KNeighborsRegressor() 

'''find the optimal parameters in KNN'''
param_dict = {
                'n_neighbors': [5,10,15],
                'weights': ['uniform', 'distance' ],
                'p' :[1, 2]          
             }

from sklearn.model_selection import GridSearchCV
knn = GridSearchCV(knn,param_dict)
knn.fit(X_train,y_train)
knn.best_params_ 
knn.best_score_


'''refit knn to the model with optimal parameters'''
knn=KNeighborsRegressor(n_neighbors= 10, p=1, weights='distance')
knn.fit(X_train,y_train)
#predictions for test
y_pred2 = knn.predict(X_test)


'''R2 and adjusted R2, and rmse'''
r2=knn.score(X_test,y_test)
print(r2)
adj_r2 = 1 - ((len(X_test)-1)/(len(X_test)-i-1))*(1-knn.score(X_test, y_test))
print(adj_r2)

mse=mean_squared_error(y_test, y_pred2) #biased mean
rmse = math.sqrt(mse)
print(rmse)


'''Cross validation'''
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.metrics import SCORERS
SCORERS.keys()

kf= KFold(n_splits=5, shuffle= True, random_state=1)
knn = KNeighborsRegressor(n_neighbors= 10, p=1, weights='distance') 
knn1 = KNeighborsRegressor(n_neighbors= 10, p=1, weights='distance') 
knn1=knn1.fit(X_train, y_train)

'''Cross validation score (R2 for test data, and full data)'''
r2score_2=cross_val_score(knn1,X_test,y_test, cv=kf, scoring='r2')
print(r2score_2.mean())
r2score_2b=cross_val_score(knn,X,y, cv=kf, scoring='r2')
print(r2score_2b.mean())

'''Cross validation score (RMSE for test data, and full data)'''
RMSE_2=np.sqrt(-cross_val_score(knn1,X_test,y_test, cv=kf, scoring='neg_mean_squared_error'))
print(RMSE_2.mean())
RMSE_2b=np.sqrt(-cross_val_score(knn,X,y, cv=kf, scoring='neg_mean_squared_error'))
print(RMSE_2b.mean())








'''------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------'''
'''fit svr to the model'''
from sklearn.svm import SVR # svc= svm for classification; SVR= svm for regressor
svr=SVR()



'''find the optimal parameters in KNN (this gonna take a long time!!!!!!!)'''
param_dict = {
                'kernel': ['linear', 'poly', 'rbf'],
                'degree': [2,3,4],
                'C' :[0.001, 0.01, 0.1, 1]          
             }


from sklearn.model_selection import GridSearchCV
svr = GridSearchCV(svr,param_dict)
svr.fit(X_train,y_train)
svr.best_params_ # best parameters for SGDClassifier(), you put the output parameters in SGDClassifier()
svr.best_score_ # the above parameters gives you this best accuracy



'''refit svr to the model with optimal parameters'''
svr=SVR(C=1, degree=2, kernel='rbf')
svr.fit(X_train,y_train)
#predictions for test
y_pred3 = svr.predict(X_test)
 

'''R2 and adjusted R2, and rmse'''
r2=svr.score(X_test,y_test)
print(r2)
adj_r2 = 1 - ((len(X_test)-1)/(len(X_test)-i-1))*(1-svr.score(X_test, y_test))
print(adj_r2)

mse=mean_squared_error(y_test, y_pred3) #biased mean
rmse = math.sqrt(mse)
print(rmse)

'''Cross validation'''
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.metrics import SCORERS
SCORERS.keys()

kf= KFold(n_splits=5, shuffle= True, random_state=1)

svr=SVR(C=1, degree=2, kernel='rbf')

svr1=SVR(C=1, degree=2, kernel='rbf')
svr1=svr1.fit(X_train, y_train)


'''Cross validation score (R2 for test data, and full data)'''
r2score_3=cross_val_score(svr1,X_test,y_test, cv=kf, scoring='r2')
print(r2score_3.mean())
r2score_3b=cross_val_score(svr,X,y, cv=kf, scoring='r2')
print(r2score_3b.mean())

'''Cross validation score (RMSE for test data, and full data)'''
RMSE_3=np.sqrt(-cross_val_score(svr1,X_test,y_test, cv=kf, scoring='neg_mean_squared_error'))
print(RMSE_3.mean())
RMSE_3b=np.sqrt(-cross_val_score(svr,X,y, cv=kf, scoring='neg_mean_squared_error'))
print(RMSE_3b.mean())










'''------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------'''
'''random forest regressor'''

''' plot, to find the optimal n_estimator(43) and  max_depth(15)'''
from sklearn.ensemble import RandomForestRegressor

test_scores=[]
for n in range(15,50):
    model=RandomForestRegressor(n_estimators=n, max_depth=10, random_state=0)
    
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    test_scores.append(mean_squared_error(y_test, y_pred))
    
plt.plot(range(15, 50), test_scores)
plt.xlabel('n of DTs')
plt.ylabel('MSE')



test_scores1=[]
for k in range(1,60):
    model=RandomForestRegressor(n_estimators=43, max_depth=k, random_state=0)
    
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    test_scores1.append(mean_squared_error(y_test, y_pred))
    
plt.plot(range(1, 60), test_scores1)
plt.xlabel('n of depth')
plt.ylabel('MSE')


'''fit rfr to the model'''

rfr= RandomForestRegressor(n_estimators=43, criterion='mse', max_depth=15)
#20 decision tree
rfr.fit(X_train,y_train)
y_pred4 = rfr.predict(X_test)


'''R2 and adjusted R2, and rmse'''
r2=rfr.score(X_test,y_test)
print(r2)
adj_r2 = 1 - ((len(X_test)-1)/(len(X_test)-i-1))*(1-rfr.score(X_test, y_test))
print(adj_r2)

mse=mean_squared_error(y_test, y_pred4) #biased mean
rmse = math.sqrt(mse)
print(rmse)


'''Cross validation'''
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.metrics import SCORERS
SCORERS.keys()

kf= KFold(n_splits=5, shuffle= True, random_state=1)

rfr= RandomForestRegressor(n_estimators=43, criterion='mse', max_depth=15)

rfr1= RandomForestRegressor(n_estimators=43, criterion='mse', max_depth=15)
rfr1=rfr1.fit(X_train, y_train)

'''Cross validation score (R2 for test data, and full data)'''
r2score_4=cross_val_score(rfr1,X_test,y_test, cv=kf, scoring='r2')
print(r2score_4.mean())
r2score_4b=cross_val_score(rfr,X,y, cv=kf, scoring='r2')
print(r2score_4b.mean())

'''Cross validation score (RMSE for test data, and full data)'''
RMSE_4=np.sqrt(-cross_val_score(rfr1,X_test,y_test, cv=kf, scoring='neg_mean_squared_error'))
print(RMSE_4.mean())
RMSE_4b=np.sqrt(-cross_val_score(rfr,X,y, cv=kf, scoring='neg_mean_squared_error'))
print(RMSE_4b.mean())