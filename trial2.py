import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
from sklearn import preprocessing, model_selection
import os
import copy

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

#{4: 3071, 3: 1453, 1: 88, 2: 148}

#converting all columns to numberss
dataset = pd.read_csv('train.csv', low_memory=False)
le = preprocessing.LabelEncoder()
for i in dataset.columns:
    if dataset[i].dtype == 'object' or dataset[i].dtype == 'bool':
        dataset[i] = le.fit_transform(dataset[i])
print({a:list(dataset['importance']).count(a) for a in list(dataset['importance'])})
#splitting train and test
print(set(dataset['issue.25']),dataset['issue.25'].dtype)
labels_to_be_dropped = ['appno','itemid','application','country.name','respondent.0','respondent.1','respondentOrderEng','respondent.3','respondent.4','kpdate','introductiondate','originatingbody_type']
features = [i for i in dataset.columns if i not in labels_to_be_dropped]
dataset = dataset[features]

dataset.to_csv('Processed.csv')
y = dataset['importance']
X = dataset.drop('importance',1)



#clf = LogisticRegression(C=2, penalty='none', verbose=5)             #74.81%
#clf = DecisionTreeClassifier()                                     #84.52%
clf =  RandomForestClassifier(max_depth=6,max_features=5,n_estimators=600)        #89.09%
#clf = KNeighborsClassifier(21)                                     #70.56%
#clf.fit(X_train, y_train)
#accuracy = clf.score(X_test, y_test)
#print(accuracy)

clf.fit(X,y)

dataset2 = pd.read_csv('test.csv', low_memory=False)
print(set(dataset2['issue.25']),dataset2['issue.25'].dtype)
k = dataset2['appno'].copy()
k = list(k)
le = preprocessing.LabelEncoder()
for i in dataset2.columns:
    if dataset2[i].dtype == 'object' or dataset2[i].dtype == 'bool':
        dataset2[i] = le.fit_transform(dataset2[i])
dataset2[dataset2==np.inf]=np.nan
features = [i for i in dataset2.columns if i not in labels_to_be_dropped]
dataset2 = dataset2[features]
dataset2.fillna(0,inplace=True)
print(set(dataset2['issue.25']),dataset2['issue.25'].dtype)
a = clf.predict(dataset2)
print(k)
output = pd.DataFrame({'importance':a,'appno':k})
output.set_index('appno',inplace=True)
print({a:list(output['importance']).count(a) for a in list(output['importance'])})
output.to_csv('submit.csv')
print(output.head())
