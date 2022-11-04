# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 09:10:22 2022

@author: Shaolin Jahan Eidee (20227009)
"""

a) Using your Roll No as random seed/state, select a random sample of 3000 records of
patients as your working dataset. Calculate all descriptive statistics and comment on
them.
Ans: 
Roll_No = 20227009 = 09
Code: 
import pandas as pd
data=pd.read_csv("D:\\eidee\\ASDS\\Course Materials\\Data Science\\New folder\\framingham2.csv")
datadf=pd.DataFrame.sample(data,3000,random_state=9)
import statistics as stat
x=datadf["totChol"]
mean=stat.mean(x)
print("Mean =", mean)
median=stat.median(x)
print("Median =",median)
med_high=stat.median_high(x)
med_low=stat.median_low(x)
print("Median_high =", med_high)
print("Median_low=" , med_low)
mode=stat.mode(x)
print("Mode =", mode)
h_mean = stat.harmonic_mean(x)
print("Harmonic_Mean = " ,h_mean)
std_dev=stat.stdev(x)
print("Standard deviation= ", std_dev)
var=stat.variance(x)
print("Variance =" , var)
Solution:
Mean = 237.055
Median = 234.0
Median_high = 234
Median_low= 234
Mode = 240
Harmonic_Mean = 229.09096789942902
Standard deviation= 44.286669303821085
Variance = 1961.3090780260086
b) Test at 5% significance level that the total cholesterol level is more than 240 mg/dL
Ans:
ztest, pval = stests.ztest(x, x > 240)
print(float(pval))
if pval<0.05 :
print("reject the null hpothesis")
else:
print("Accept the null hpothesis")
Soluttion:
P_value = 0.0
We reject the null hypothesis
c) Construct 95% confidence interval of heart rate.
Ans:
#CI Interval of heartrate
import numpy as np
from scipy import stats
from statsmodels.stats import weightstats as stests
import scipy.stats as ss
import statistics as sc
da = datadf["heartRate"]
m=sc.mean(da)
s=sc.stdev(da)
n=len(da)
dof=n-1
tc=ss.t.ppf(.975,dof)
print("t-critical value=",tc)
u=m+tc*s/np.sqrt(n);u
l=m-tc*s/np.sqrt(n);l
print("95% confidence interval for population mean is (",l,",",u,")")
ci = ss.t.interval(.95, dof, m, s/np.sqrt(n))
print("95% confidence interval of population mean is",ci)
Solution:
t-critical value= 1.9607553192053147
95% confidence interval for population mean is ( 75.34910707463489 , 76.20822625869845 )
d) Determine the correlation coefficients among the medical conditions and comment on
them.
Ans:
#Correlation between male & diseases
import numpy as np
from sklearn.linear_model import LinearRegression
a = datadf[['diabetes','totChol','sysBP','diaBP','BMI','heartRate','glucose']]
b = datadf['male']
reg = LinearRegression().fit(a, b)
print(reg.intercept_,"\n",reg.coef_,'\n',reg.score(a, b))
Solution: 
Intercept = 0.6284859908650391
Diabetes = 0.02620149, totChol = -0.00075348, sysBP = -0.00453414, DiaBP =
0.00883334, BMI = 0.00923747, heartrate = -0.00514665, Glucose = 0.00020044
Reg_score = 0.04598308656318373 (low positive co-relation)
e) Using your roll as the random_state, split the data into training set (70%) and test set
(30%). In order to predict CHD, develop the training model by the following
methods/models: i) Logistic regression ii) Random forest iii) Bagging (any), and iv)
Boosting (any).
Code: #Randomly select and split data into training 70% and test 30%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier,AdaBoostClassifier, BaggingClassifier
x1 = datadf[['diabetes','totChol','sysBP','diaBP','BMI','heartRate','glucose']]
y1 = datadf['TenYearCHD']
X_train, X_test, y_train, y_test = train_test_split(x1,y1,test_size=0.30, random_state=9)
lr = LogisticRegression()
lr.fit(X_train, y_train)
model = lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print(model.classes_,"\n",model.intercept_,'\n',model.coef_)
model.predict_proba(X_test)
lr_acc = metrics.accuracy_score(y_test, y_pred)
print('Logistics Regression Accuracy: ',lr_acc )
#confusion matrix
con_mat = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(con_mat,annot=True)
# Random Forest Regression
rf = RandomForestClassifier(n_estimators=3000,max_features="auto", random_state=32)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_acc = metrics.accuracy_score(y_test, y_pred_rf)
print('Random Forest Accuracy: ', y_acc)
# Boosting Regression - AdaBoosting
adb = AdaBoostClassifier(n_estimators=3000)
adb.fit(X_train, y_train)
y_pred_adb = adb.predict(X_test)
adb_acc = metrics.accuracy_score(y_test, y_pred_adb)
print("AdaBoost Accuracy : ",adb_acc)
# Bagging Regression
bgr = BaggingClassifier(n_estimators=3000)
bgr.fit(X_train, y_train)
y_pred_bgr = bgr.predict(X_test)
bgr_acc = metrics.accuracy_score(y_test, y_pred_bgr)
print("Bagging Regression Accuracy : ",bgr_acc)
Solution:
Logistics Regression Accuracy: 0.8111111111111111
Random Forest Accuracy: 0.826666666666677777
Bagging Regression Accuracy : 0.845555555555555
AdaBoost Accuracy : 0.766666666666667
f) Identify the best model from the above and using the best model.
Ans: From the above models the logistic regression model provided the best result.