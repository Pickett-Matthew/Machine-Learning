#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 18:35:13 2019

@author: matthewpickett
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix





def noParamsCar():
    
    headers = ["buying", "maint", "doors", "person","boot", "safety", "class"]
    
    data = pd.read_csv("cars.csv", skiprows=1, header=None, 
                       names=headers, na_values="?",)
    
    
    #convert headers/columns into categories, then use category values
    #for your label encoding
    
    data.buying = data.buying.astype('category')
    data["buying_cat"] = data.buying.cat.codes
    
    
    data.maint = data.maint.astype('category')
    data["maint_cat"] = data.maint.cat.codes
    
    data.boot = data.boot.astype('category')
    data["boot_cat"] = data.boot.cat.codes
    
    data.safety = data.safety.astype('category')
    data["safety_cat"] = data.safety.cat.codes
    
    data = data.replace({"doors": {"5more": 6}})
    data = data.replace({"person": {"more": 5}})
    
    data = data.drop(columns=["buying", "maint", "boot", "safety"])
    
    
    X = data.drop(columns=["class"]).values
    y = data["class"].values.flatten()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        random_state=0)
    
    #start the MLP calssifier no Params
    clf = MLPClassifier()
    model = clf.fit(X_train, y_train)
    predict = model.predict(X_test)
    
    #print Out the accuracy
    accuracy = accuracy_score(y_test, predict)
    
    print(" ")
    print("Accuracy NO params: {}".format(accuracy))
    print(" ")
    

    
    
def paramsCar():
    #same as noParams() but adding parameters to MLP()
    
    headers = ["buying", "maint", "doors", "person","boot", "safety", "class"]
    
    data = pd.read_csv("cars.csv", skiprows=1, header=None, 
                       names=headers, na_values="?",)
    
    #cleaning up the data giving numeric values
    data.buying = data.buying.astype('category')
    data["buying_cat"] = data.buying.cat.codes
    data.maint = data.maint.astype('category')
    data["maint_cat"] = data.maint.cat.codes
    data.boot = data.boot.astype('category')
    data["boot_cat"] = data.boot.cat.codes
    data.safety = data.safety.astype('category')
    data["safety_cat"] = data.safety.cat.codes
    data = data.replace({"doors": {"5more": 6}})
    data = data.replace({"person": {"more": 5}})
    data = data.drop(columns=["buying", "maint", "boot", "safety"])
    
    
    X = data.drop(columns=["class"]).values
    y = data["class"].values.flatten()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        random_state=0)
    
    #Using scaler to scale the data
    scaler = StandardScaler()
    
    scaler.fit(X_train)
    
    X_train = scaler.transform(X_train)
    X_test  = scaler.transform(X_test)
    
    
    #start the MLP calssifier no Params
    mlp = MLPClassifier(hidden_layer_sizes=(30, 30, 30), max_iter=300, solver="adam")
    
    model = mlp.fit(X_train, y_train)
    predict = model.predict(X_test)
    
    from sklearn.metrics import classification_report,confusion_matrix
    
    print(confusion_matrix(y_test, predict))
    print(classification_report(y_test, predict))
    
    
    #print Out the accuracy
    accuracy = accuracy_score(y_test, predict)
    
    print(" ")
    print("Accuracy With params: {}".format(accuracy))
    print(" ")
    
    import scikitplot as skplt
    import matplotlib.pyplot as plt
    
    skplt.metrics.plot_confusion_matrix(y_test, predict, normalize=True)
    plt.show()
    



def income():
    
    names = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
         "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
         "hours_per_week", "native_country", "income"]

    # Load the file
    data = pd.read_csv("adult.data.txt", header=None, skipinitialspace=True,
                       names=names, na_values=["?"])
    
    #data processing
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
                       "relationship", "sex", "native_country", "income"])

    X = data.drop(columns=["incomeHigh"]).as_matrix()
    y = data["incomeHigh"].as_matrix().flatten()
    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    #recommended to scale data
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    scaler.fit(X_train)
    
    X_train = scaler.transform(X_train)
    X_test  = scaler.transform(X_test)
    
    
    mlp = MLPRegressor() 
    
    mlp.fit(X_train, y_train)
    predict = mlp.predict(X_test)

    print("############# ALL DEFAULT HYPER-PARAMETERS #################")
    print(confusion_matrix(y_test, predict.round()))
    print(classification_report(y_test, predict.round()))



    mlp = MLPRegressor(hidden_layer_sizes=(10, 10, 10), max_iter=10000, momentum=0.3) 
    
    mlp.fit(X_train, y_train)
    predict = mlp.predict(X_test)
    
    print("############# NEW HYPER-PARAMETERS #################")
    print(confusion_matrix(y_test, predict.round()))
    print(classification_report(y_test, predict.round()))
   
    

def main():
    print("""
              Option 1 - Car Classifier multiple targets NO HYPER-PARAMS\n 
              Option 2 - Car Classifier WITH HYPER-PARAMS\n
              Option 3 - Predictions for adults with income over 50k
                         mlpRegressor. Includes 2 results of mixed hyper-params\n 
                      """)
    
    X = input("Enter your option: ")
              
    
    
    if (X == "1"):
        noParamsCar()
    elif(X == "2"):
        paramsCar()
    elif(X == "3"):
        income()
        
        
        
    
    
if __name__ == "__main__": main()












