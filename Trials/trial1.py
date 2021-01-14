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
from sklearn.svm import SVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

#{4: 3071, 3: 1453, 1: 88, 2: 148}
#converting all columns to numberss
dataset = pd.read_csv('train.csv', low_memory=False)
le = preprocessing.LabelEncoder()
for i in dataset.columns:
    if dataset[i].dtype == 'object' or dataset[i].dtype == 'bool':
        dataset[i] = le.fit_transform(dataset[i])
print({a:list(dataset['importance']).count(a) for a in list(dataset['importance'])})
#splitting train and test


#--------------------------------------------------------------------------------------
labels_to_be_dropped = ['appno','application','country.alpha2','country.name','docname','ecli','itemid','languageisocode','originatingbody_name','originatingbody_type',\
                        'parties.0','parties.2','parties.1','respondent.0','respondent.1','respondent.2','respondent.3','respondent.4','respondentOrderEng','sharepointid','documentcollectionid=CASELAW',\
                        'documentcollectionid=JUDGMENTS','documentcollectionid=ENG','documentcollectionid=CHAMBER','documentcollectionid=COMMITTEE','documentcollectionid=GRANDCHAMBER'\
                        ]
mxdpth = 5 #increasing decreses the number of 4s
fea = 4 #increasing reduces sensitivity
n_est = 250
rs = 85
#--------------------------------------------------------------------------------------

features = [i for i in dataset.columns if i not in labels_to_be_dropped]
dataset = dataset[features]

dataset.to_csv('Processed.csv')
y = dataset['importance']
X = dataset.drop('importance',1)



#clf = LogisticRegression(C=5, penalty='l2', verbose=5)             #74.81%
#clf = DecisionTreeClassifier(max_depth=75)                                     #84.52%
clf =  RandomForestClassifier(max_depth=mxdpth,max_features=fea,random_state=rs,n_estimators=n_est)        #89.09%
string = str(clf).replace('\t','').replace('\n','').replace('                       ',' ')+' '+str(labels_to_be_dropped)
#clf = KNeighborsClassifier(23)                                     #70.56%
#clf = SVC()
#clf = QuadraticDiscriminantAnalysis()
#clf.fit(X_train, y_train)
#accuracy = clf.score(X_test, y_test)
#print(accuracy)

clf.fit(X,y)

dataset2 = pd.read_csv('test.csv', low_memory=False)
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
a = clf.predict(dataset2)
output = pd.DataFrame({'importance':a,'appno':k})
output.set_index('appno',inplace=True)
print({a:list(output['importance']).count(a) for a in list(output['importance'])})
output.to_csv('submit.csv')
print(output.head())
print(string)
with open('log.txt','a+') as file:
    accuracy = input('Enter the accuracy achieved')
    file.write('\n'+string+' '+accuracy)

