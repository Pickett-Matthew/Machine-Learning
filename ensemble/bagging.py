# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestClassifier



import pandas as pd

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

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y, test_size=0.2)




classifier = DecisionTreeClassifier(random_state=42)
model = classifier.fit(X_train, y_train)
predictions = model.predict(X_test)

acc = accuracy_score(y_test, predictions)
print("accuracy for decisionTreeClassifier")
print(acc)


clf = GaussianNB()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
acc2 = accuracy_score(y_test, predictions)

print("Accuracy for GaussianNB")
print(acc2)


k = 7
classifier = KNeighborsClassifier(n_neighbors=k)
model = classifier.fit(X_train, y_train)
predictions = model.predict(X_test)
acc = accuracy_score(y_test, predictions)

print("acc for KNN")
print(acc)



clf = AdaBoostClassifier(n_estimators=100)
scores = cross_val_score(clf, X_train, y_train, cv=5)
print("adaBoost acc")
print(scores.mean())



bagging = BaggingClassifier(n_estimators=10)
scores = cross_val_score(classifier, X_train, y_train, cv=5, 
                         scoring=make_scorer(accuracy_score))
acc = scores.mean()


print("bagging score")
print(acc)


best_cross_val_score = 0
best_k = 100
for k in range(100, 1000, 100):
	classifier = RandomForestClassifier(n_estimators=k)
	scores = cross_val_score(classifier, X_train, y_train, cv=5, scoring=make_scorer(accuracy_score))
	metric = scores.mean()

	print("CV ACCURACY for (Iris_Binned) n_estimators={}: {}".format(k, metric))
	if metric > best_cross_val_score:
		best_cross_val_score = metric
		best_k = k
        
print("--> BEST CV ACCURACY for Random Forest n_estimators={} (Iris_Binned): {}".format(best_k, metric))
print(" ")








