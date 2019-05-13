#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 18:18:53 2019

@author: matthewpickett
"""

import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import graphviz

from sklearn.datasets import load_iris


def iris_data():
    iris = load_iris()
    clf = tree.DecisionTreeClassifier(splitter="best", max_leaf_nodes=6)
    clf = clf.fit(iris.data, iris.target)
    
    
    dot_data = tree.export_graphviz(clf, out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render("irisTree")
        


    """
    Had to use these commands in terminal for the pdf to work
    pip install graphviz
    conda install graphviz
    """

#Numeric = iris
#Categorical = lenses
#missing data = voting

def lenses_data():
    #dataset is complete
    #24 instances 4 attributes
    
    headers = ["age", "script", "astigmatic", "TPR", "target"]
    
    data = pd.read_csv("lenses.data.txt", delim_whitespace=True,
                       names=headers, index_col=None)


    target = data["target"]
   
   
    data = data.drop(columns=["target"])
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(data, target)
    
    
    
    dot_data = tree.export_graphviz(clf, out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render("lenseTree")







def student():
    
    data = pd.read_csv("student-mat.csv", na_values=["?"], sep=";")
    
    data["isGabrielPereira"]            = data.school.map({"GP": 1, "MS": 0})
    data["isMale"]                      = data.sex.map({"M": 1, "F": 0})
    data["isUrban"]                     = data.address.map({"U": 1, "R": 0})
    data["isLessOrEqual"]               = data.famsize.map({"LE3": 1, "GT3": 0})
    data["isLivingTogether"]            = data.Pstatus.map({"T": 1, "A": 0})
    data["isExtraEducationalSupport"]   = data.schoolsup.map({"yes": 1, "no": 0})
    data["isFamilyEducationalSupport"]  = data.famsup.map({"yes": 1, "no": 0})
    data["isExtraPaidClasses"]          = data.paid.map({"yes": 1, "no": 0})
    data["isExtraCurricularActivities"] = data.activities.map({"yes": 1, "no": 0})
    data["isAttendedNurserySchool"]     = data.nursery.map({"yes": 1, "no": 0})
    data["wantsHigheEducation"]         = data.higher.map({"yes": 1, "no": 0})
    data["internetAccessAtHome"]        = data.internet.map({"yes": 1, "no": 0})
    data["withRomanticRelationship"]    = data.romantic.map({"yes": 1, "no": 0})
    
    
    #Cat.code columns that will need to be dropped
    data.Mjob = data.Mjob.astype('category')
    data["Mjob_cat"] = data.Mjob.cat.codes
    
    data.Fjob = data.Fjob.astype('category')
    data["Fjob_cat"] = data.Fjob.cat.codes
    
    data.reason = data.Fjob.astype('category')
    data["reason_cat"] = data.reason.cat.codes
    
    data.guardian = data.Fjob.astype('category')
    data["guardian_cat"] = data.guardian.cat.codes
    
    
    data = data.drop(columns=["school", "sex", "address", "famsize",
                                          "Pstatus", "Mjob", "Fjob", "reason",
                                          "guardian", "schoolsup", "famsup",
                                          "paid", "activities", "nursery",
                                          "higher", "internet", "romantic"])
    #G# is the column we need
    X = data.drop(columns=["G3"]).values
    y = data["G3"].values.flatten()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                       
    #max Depth of three fits data nicely 
    Regress = tree.DecisionTreeRegressor(max_depth=3)
    Regress = Regress.fit(X_train, y_train)

    
    dot_data = tree.export_graphviz(Regress, out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render("studentTree")
    
    
    
    
    
lenses_data()   
student()   
iris_data()
    
    
    
    
    
    
    
    
    
    
    
    
