#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 21:05:20 2019

@author: matthewpickett
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from scipy.spatial import distance


iris = datasets.load_iris()
 
data_train, data_test, target_train, target_test = train_test_split(
        iris.data, iris.target, test_size = 0.30)

train = np.array(zip(data_train,target_train))
test = np.array(zip(data_test, target_test))

#variablies to work with
#data_train = attributes 
#data_test
#target_train = number corresponding to flower
#target_test

 
#****************existing Algorithm*******************************************

classifier = KNeighborsClassifier(n_neighbors=3)
#fit() trains the data
model = classifier.fit(data_train, target_train) 
#predict() gets the untrained data, or test_data, and makes a guess
predictions = classifier.predict(data_test)

#print(model)
#print(predictions)


from sklearn.metrics import accuracy_score

accuracy = accuracy_score(target_test, predictions)

#print(accuracy)
  

class KNNClassifier:
    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors
        
        
    def fit(self, data_train, target_train):
        self.data_train = data_train
        self.target_train = target_train
        model = KNNModel(data_train, target_train, self.n_neighbors)
        return model
      
        
   
       

class KNNModel:
    def __init__(self, data_train, target_train, n_neighbors):
        self.data_train = data_train
        self.target_train = target_train
        self.n_neighbors = n_neighbors
        
     
      
    def predict(self, data_test):
        pred = []
        for row in data_test:
            label = self.get_distance(row)
            pred.append(label)
            
        return pred
        
        
    def get_distance(self, row):
        dist = distance.euclidean(row, self.data_train[0])
        index = 0
        
        for i in range(1, len(self.data_train)):
            val = distance.euclidean(row, self.data_train[i])
            if val < dist:
                dist = val
                index = i
        
        return self.data_train[index]
        
            
        
        
       
classifier = KNNClassifier(n_neighbors=3)
model = classifier.fit(data_train, target_train)
predictions = model.predict(data_test)

from sklearn.metrics import accuracy_score

print(accuracy_score(target_train, predictions))

