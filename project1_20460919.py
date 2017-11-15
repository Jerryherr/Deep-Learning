# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 13:01:23 2017

@author: Jerryho
"""

#Project 1
import os
import pandas as pd
import numpy as np
#from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectKBest#,SelectFromModel
from sklearn.feature_selection import chi2,mutual_info_classif,f_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.svm import SVC
from sklearn.cross_validation import KFold


work_dir = os.getcwd()
x_train = pd.read_csv(work_dir + '/traindata.csv', header = None)
y_train = pd.read_csv(work_dir + '/trainlabel.csv', header = None)

#==============================================================================
# #Normalization
# x_train -= np.mean(x_train, axis=0)
# x_train /= np.std(x_train, axis=0)
#==============================================================================

kbest = SelectKBest(f_classif, k = 40)
x_train_new_fit = kbest.fit(x_train, y_train)
selected = x_train_new_fit.get_support()
x_train_new = x_train_new_fit.transform(x_train)

#==============================================================================
# #PCA
# pcA = PCA(n_components = 30)
# x_train_new = pcA.fit_transform(x_train_new)
#==============================================================================

#Normalization
#Subtract the mean for each feature
mean = np.mean(x_train_new, axis=0)
x_train_new -= mean
#Divide each feature by its standard deviation
std = np.std(x_train_new, axis=0)
x_train_new /= std

#==============================================================================
# #guiyihua
# scaler = MinMaxScaler()
# x_train_new = scaler.fit_transform(x_train_new)
#==============================================================================

kf = KFold(n=len(y_train), n_folds=5, shuffle=True)
tst_gb = []

for tr,tst in kf:
    #Train Test Split
    tr_features = x_train_new[tr,:]
    tr_target = np.array(y_train)[tr]
    tst_features = x_train_new[tst,:]
    tst_target = np.array(y_train)[tst]
        
    clf_gb = GradientBoostingClassifier(n_estimators = 300)
    clf_gb.fit(tr_features,tr_target)
    predict_gb = clf_gb.predict(tst_features)
    
    label = tst_target.T
    accuracy_gb = np.mean(predict_gb == label)

    tst_gb.append(accuracy_gb)
    
print 'The accuracy of classification: ',np.mean(tst_gb)

test = pd.read_csv(work_dir + '/testdata.csv', header = None)
x_test = np.array(test)
x_test_new = x_test[:,selected]
x_test_new -= mean
x_test_new /= std
CLF_gb = GradientBoostingClassifier(n_estimators = 300)
CLF_gb.fit(x_train_new,y_train)
result = CLF_gb.predict(x_test_new)

np.savetxt('project1_20460919.csv',result)

