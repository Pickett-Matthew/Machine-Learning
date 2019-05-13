# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import csv 
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score

#http://pbpython.com/categorical-encoding.html



#define the headers
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

dh = data.head()

class_values = data.drop(columns=["class"]).values
class_targets = data["class"].values.flatten()

X_train, X_test, y_train, y_test = train_test_split(class_values,
                                                    class_targets, test_size=0.2)

classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Car Accuracy: {}".format(str(accuracy*100) + "%"))
        

headers = ["mpg","cylinders", "displacement", "horsepower", "weight", 
           "acceleration", "model_year","origin","car_name"]

data = pd.read_csv("mpg.csv", skiprows=1, names=headers, na_values=["?"],
                   delim_whitespace=True)

#data missing in the Horsepower column
#fill na with the median values of the column

data.horsepower = data.horsepower.fillna(data.horsepower.median())

data = data.drop(columns=["car_name"])


class_values = data.drop(columns=["mpg"]).values
class_targets = data["mpg"].values.flatten()   

 #need to do regression on this model
 
X_train, X_test, y_train, y_test = train_test_split(class_values,
                                                    class_targets, test_size=0.2)
        
from sklearn.neighbors import KNeighborsRegressor

# ...
# ... code here to load a training and testing set
# ...


regr = KNeighborsRegressor(n_neighbors=5)

regr.fit(X_train, y_train)

predictions = regr.predict(X_test)       
        
accuracy = r2_score(y_test, predictions)

print("MPG Accuracy is: {}".format(str(accuracy*100) + "%"))


#predict the final grade of the math class

data = pd.read_csv("student-mat.csv", sep=";")

#find out the non numeric data types
#print(data.dtypes)

"""
ATTRIBUTES to change
school        object
sex           object
address       object
famsize       object
Pstatus       object
Mjob          object
Fjob          object
reason        object
guardian      object
schoolsup     object
famsup        object
paid          object
activities    object
nursery       object
higher        object
internet      object
romantic      object
"""


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

regr = KNeighborsRegressor(n_neighbors=3)
regr.fit(X_train, y_train)

predictions = regr.predict(X_test)

# Compute and print the accuracy
accuracy = r2_score(y_test, predictions)

print("Student Accuracy: {}".format(str(accuracy*100) + "%"))























