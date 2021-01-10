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

#converting all columns to numberss
dataset = pd.read_csv('train.csv', low_memory=False)
le = preprocessing.LabelEncoder()
for i in dataset.columns:
    if dataset[i].dtype == 'object' or dataset[i].dtype == 'bool':
        dataset[i] = le.fit_transform(dataset[i])

#splitting train and test
labels_to_be_dropped = []#,'decisiondate']
features = [i for i in dataset.columns if i not in labels_to_be_dropped]
dataset = dataset[features]

dataset.to_csv('Processed.csv')
y = dataset['importance']
X = dataset.drop('importance',1)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.25)

print(X_train.shape,y_train.shape)


#clf = LogisticRegression(C=4, penalty='l2', verbose=5)             #74.81%
#clf = DecisionTreeClassifier()                                     #84.52%
clf =  RandomForestClassifier(max_depth=30,max_features=30)        #89.09%
#clf = KNeighborsClassifier(13)                                     #70.56%
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)

##dataset = pd.read_csv('train.csv', low_memory=False)
##le = preprocessing.LabelEncoder()
##for i in dataset.columns:
##    if dataset[i].dtype == 'object' or dataset[i].dtype == 'bool':
##        dataset[i] = le.fit_transform(dataset[i])
