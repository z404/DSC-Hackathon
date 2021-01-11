'''
--------------------------------------------
Grievance Resolution using ML
DSC-VIT Ice-breaker hackathon by Ius Humanum
--------------------------------------------
'''

#Importing Required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
from sklearn import preprocessing, model_selection
import os
import copy

#Importing the machine learning algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

#Reading the dataset
dataset = pd.read_csv('train.csv', low_memory=False)

#Converting the text in the dataset to integers, so sklearn can handle it
le = preprocessing.LabelEncoder()
for i in dataset.columns:
    if dataset[i].dtype == 'object' or dataset[i].dtype == 'bool':
        dataset[i] = le.fit_transform(dataset[i])

#Dropping unwanted
labels_to_be_dropped = []
features = [i for i in dataset.columns if i not in labels_to_be_dropped]
dataset = dataset[features]

#Saving integer dataset
dataset.to_csv('Processed.csv')

#Extracting Lable and Features
y = dataset['importance']
X = dataset.drop('importance',1)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.25)

#Creating the classifier and training the model
#clf = LogisticRegression(C=4, penalty='l2', verbose=5)             #74.81%
#clf = DecisionTreeClassifier()                                     #84.52%
clf =  RandomForestClassifier(max_depth=30,max_features=30)        #89.09%
#clf = KNeighborsClassifier(13)                                     #70.56%
#clf = AdaBoostClassifier()                                         #84.23%
clf.fit(X_train, y_train)

#Printing the accuracy of the model, on the testing part of the dataset
accuracy = clf.score(X_test, y_test)
print(accuracy)
