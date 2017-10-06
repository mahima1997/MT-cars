# -*- coding: utf-8 -*-
"""
Created on Thu May 25 13:15:41 2017

@author: hp
"""
#Applying linear regression(with weight as predictor(X) results in a bit underfitting line 
# so we switch to polynomial regression(with weight+weight^2 as predictor(X)) which has grester Rms than linear regression 
#APPLYING MULTIPLE LINEAR REGRESSION WITH POLYNOMIAL TERMS will be thus suitable for this kind of data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import matplotlib
matplotlib.style.use('ggplot')
from pandas.tools.plotting import scatter_matrix
import pylab
from sklearn import model_selection
from sklearn import linear_model
#from sklearn.metrics import accuracy_score  ...accuracy score but R-squared is calculated for linear regression 
from sklearn.metrics import mean_squared_error
from math import sqrt

data=pd.read_csv('mtcars.csv',encoding = "iso-8859-15")
data.isnull().any()
data = data.fillna(method='ffill')      #to fill the NaN using ffill, so they are populated by the last known value.
print(data)                             #e.g. (32,hp) has been replaced by 95                                    
print("shape of data=",data.shape)
print("header of data=\n",data.head(5))

#data.plot and scatter_matrix both plot the same thing. It's just that data.plot plots it individually.The top row
#of scatter plot plots all the predictors against the response(mpg)
data.plot(kind="scatter",
           x="acceleration",
           y="mpg",
           figsize=(9,9),
           color="black")
scatter_matrix(data)
plt.show()
#the plot shows a linear relation suggesting linear regression model
array=data.values
X = array[:,1:8]
Y = array[:,0]
validation_size = 0.30
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
X_validation_frame = pd.DataFrame(X_validation)
print(X_validation_frame)
model=linear_model.LinearRegression()
model.fit(X_train,Y_train)

# Check trained model y-intercept
print("model.intercept=",model.intercept_)

# Check trained model coefficients
print("model.coefs=",model.coef_)

Y_pred=model.predict(X_validation)
residuals = Y_validation-Y_pred
print("residuals:",residuals)
    
#SS residuals is the sum of the squares of the model residuals and  
#SSTotal is the sum of the squares of the difference between each data point and the mean of the data
SSResiduals = (residuals**2).sum()
SSTotal = ((Y_validation - Y_validation.mean())**2).sum()

#The output of the score function for linear regression is "R-squared", a value that ranges from 0 to 1 
#which describes the proportion of variance in the response variable that is explained by the model. 
#In this case, car weight explains roughly 81.8% of the variance in mpg.
# R-squared
print("R-squared=",1 - (SSResiduals/SSTotal))
print("Mean squared error=",sqrt(mean_squared_error(Y_validation, Y_pred)))
print("R-squared=",model.score(X = X, y = Y))

#print("hello")

plt.figure(figsize=(9,9))
stats.probplot(residuals, dist="norm", fit=True, plot=plt)    #the obtained plot has residuals following a slightly non
pylab.show()                                         #linear pattern which is straighter than in case of single

df_output = pd.DataFrame()
aux = pd.read_csv('mtcars.csv')
df_output['weight'] = aux['weight'].iloc[120:]
df_output['mpg'] = Y_pred
df_output[['weight','mpg']].to_csv('mtcars predicted.csv',index=False)

pred_data=pd.read_csv('mtcars predicted.csv')
data.plot(kind="scatter",
           x="weight",
           y="mpg",                     #------RESOLVE-----multiple regression is applied on data 
           figsize=(9,9),               #but is checked for only weight
           color="black")

# Plot regression line
plt.plot(pred_data,'weight',      # Explanitory variable  #what is the syntax error
               'mpg',               # Predicted values            -----RESOLVE--------
               color="blue")


#We could continue adding more explanatory variables in an attempt to improve the model. Adding variables that 
#have little relationship with the response or including variables that are too closely related to one another 
#can hurt your results when using linear regression. You should also be wary of numeric variables 
#that take on few unique values since they often act more like categorical variables than numeric ones.