#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 12:37:38 2019

@author: matthewpickett
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score



names = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
     "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
     "hours_per_week", "native_country", "income"]

# Load the file
data = pd.read_csv("adult.data.txt", header=None, skipinitialspace=True,
                   names=names, na_values=["?"])

#data processing
data.dropna(how="any", inplace=True)

data.workclass.value_counts()
data.workclass = data.workclass.astype('category')
data["workclass_cat"] = data.workclass.cat.codes

data.education.value_counts()
data.education = data.education.astype('category')
data["education_cat"]= data.education.cat.codes


data.marital_status.value_counts()
data.marital_status = data.marital_status.astype('category')
data["marital_status_cat"]= data.marital_status.cat.codes


data.occupation.value_counts()
data.occupation = data.occupation.astype('category')
data["occupation_cat"]= data.occupation.cat.codes


data.relationship.value_counts()
data.relationship = data.relationship.astype('category')
data["relationship_cat"]= data.relationship.cat.codes


data.race.value_counts()
data = pd.get_dummies(data, columns=["race"])

data.sex.value_counts()
data["isMale"] = data.sex.map({"Male": 1, "Female": 0})

data.native_country.value_counts()
data.native_country = data.native_country.astype('category')
data["native_country_cat"]= data.native_country.cat.codes


data.income.value_counts()
data["incomeHigh"] = data.income.map({">50K": 1, "<=50K": 0})


data = data.drop(columns=["workclass", "education", "marital_status", "occupation",
                   "relationship", "sex", "native_country", "income", "capital_gain", 
                   "capital_loss"])

y = data["incomeHigh"]
X = data.drop(columns=["incomeHigh"])


#X = data.drop(columns=["incomeHigh"]).as_matrix()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#recommended to scale data
scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test)


mlp = MLPRegressor(max_iter=20000, hidden_layer_sizes=(20, 20, 20)) 

mlp.fit(X_train, y_train)
predict = mlp.predict(X_test)



"""
print("############# ALL DEFAULT HYPER-PARAMETERS #################")
print(confusion_matrix(y_test, predict.round()))
print(classification_report(y_test, predict.round()))
"""



