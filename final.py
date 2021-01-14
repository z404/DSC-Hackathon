'''
DSC Icebreaker hackathon by Ius Humanum

Program to predct the importance of citizens' grievances by assessing the significance of various articles, \
constitutional declarations, enforcement and other pertinent resources in relation to the aforementioned grievances.
'''
########################
## IMPORTING PACKAGES ##
########################

# Pandas and Numpy for maipulation of dataframes
import pandas as pd
import numpy as np
# Preprocessing for processing string labels in the dataset (as LBGM classifier doesn't accept string columns)
from sklearn import preprocessing

# LBGM Classifier from lightgbm classifier
from lightgbm import LGBMClassifier

# Sklearn for basic functions of ML, like cross validation and train test splitting
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score

####################################
## Loading the train and test set ##
####################################

# Reading both given CSVs into pandas.DataFrames
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#########################
## Feature Engineering ##
#########################

# Defining the label encoder for string features processing
le = preprocessing.LabelEncoder()

# To get a good idea of the dataset, train and test set will be combined
test['importance']=-1
train['label'] = 'train'
test['label'] = 'test'
# Concatenating the dataframes after labelling the datasets
combined = pd.concat([train,test],axis=0)

# Function to combine all "issues" columns in the dataset
def combine_issues(df):
    issue_columns = [
        'issue.0', 'issue.1', 'issue.2', 'issue.3', 'issue.4', 'issue.5', 'issue.6', 'issue.7', 'issue.8', 
        'issue.9', 'issue.10', 'issue.11', 'issue.12', 'issue.13', 'issue.14', 'issue.15', 'issue.16', 
        'issue.17', 'issue.18', 'issue.19', 'issue.20', 'issue.21', 'issue.22', 'issue.23']
    issue_df = combined[issue_columns]
    issue_df.fillna('',inplace=True)
    issue_df['issues'] = issue_df[issue_columns].apply(lambda x: '. '.join([val for val in x if val != '']), axis=1)
    df.drop(issue_columns, axis=1, inplace=True)
    issue_df.drop(issue_columns, axis=1, inplace=True)
    df = pd.concat([df, issue_df], axis=1)
    return df

# Function to make all strings lowercase in the dataframe
def lowercase_texts(df):
    for col in combined.columns:
        if combined[col].dtype=='object':
            combined[col] = combined[col].str.lower()
    return df

# Function to make all countries a single letter, so it's easier to manage, and doesn't cause outliers
def universalize_countries(df):
    country_dict_A = df[['respondentOrderEng','country.name']].set_index('country.name').T.to_dict('list')
    country_dict_C = df[['respondentOrderEng','respondent.0']].set_index('respondent.0').T.to_dict('list')
    print(country_dict_A)
    country_dict = {}
    for d in (country_dict_A, country_dict_C):
        country_dict.update(d)
        
    country_dict = {k: v for k, v in country_dict.items() if pd.notna(k)}
    df['respondent.0'] = df['respondent.0'].apply(lambda x: country_dict[x][0])
    df['respondent.1'] = df['respondent.1'].apply(lambda x: country_dict[x][0] if pd.notnull(x) else x)
    df['respondent.2'] = df['respondent.2'].apply(lambda x: country_dict[x][0] if pd.notnull(x) else x)
    df['respondent.3'] = df['respondent.3'].apply(lambda x: country_dict[x][0] if pd.notnull(x) else x)
    df['respondent.4'] = df['respondent.4'].apply(lambda x: country_dict[x][0] if pd.notnull(x) else x)
    del df['respondentOrderEng']
    return df

# Function to remove the columns which have just one unique value
# These columns don't contribute to the accuracy or the model
def remove_constant_values(df):
    print('Removing constant columns -> ',)
    for col in df.columns:
        if df[col].nunique()==1:
            print(col,end=', ' )
            del df[col]
    return df

# Function to drop unrelated and unrequired columns, which don't provide any valuable input to the model
def remove_unwanted_features(df):
    remove_cols =['parties.0', 'country.alpha2', 'parties.1', 'country.name', 'docname', 'appno', 'ecli', 'kpdate', 'originatingbody_name']
    for col in remove_cols:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
    return df

# Function to modify existing columns to obtain useful data
def featurize_columns(df):
    df['itemid'] = df['itemid'].apply(lambda x: x[4:7])
    df['sharepointid'] = df['sharepointid'].apply(lambda x: str(x)[:3])
    # Combining the respondent columns into one column to give equal importance to each
    df['total_respondents'] = 5- df[['respondent.0','respondent.1','respondent.2','respondent.3','respondent.4']].isna().sum(axis=1)
    return df

# Function to split certain columns into more columns which give more data, like dates into months, days, and years.
def featurize_date_columns(df):
    df['daysbetween_intro_decision'] = (pd.to_datetime(df['decisiondate']) - pd.to_datetime(df['introductiondate'])).dt.days
    df['daysbetween_intro_judgement'] = (pd.to_datetime(df['judgementdate']) - pd.to_datetime(df['introductiondate'])).dt.days
    df['daysbetween_decision_judgement'] = (pd.to_datetime(df['judgementdate']) - pd.to_datetime(df['decisiondate'])).dt.days
    df.drop(['decisiondate','introductiondate','judgementdate'], axis=1, inplace=True)
    return df

# Function to apply the 
def encoding(df):
    df['doctypebranch'] = le.fit_transform(df['doctypebranch'])
    df['separateopinion'] = le.fit_transform(df['separateopinion'])
    df['typedescription'] = le.fit_transform(df['typedescription'])
    return df

# Function to fill empty cells with default value 0, and make sure the column datatype is int
def fill_missing(df):
    for col in df.columns:
        if col not in ['label', 'issues']:
            df[col].fillna(0,inplace=True)
            df[col] = df[col].astype('int')
    return df


# Running all the above functions on the combined train and test dataset
combined = combine_issues(combined)
print('combined shape after combining issues ->', combined.shape)
combined = lowercase_texts(combined)
combined = universalize_countries(combined)
combined = featurize_columns(combined)
combined = featurize_date_columns(combined)
combined = encoding(combined)
combined = remove_constant_values(combined)
print('\ncombined shape after removing constant features->', combined.shape)
combined = remove_unwanted_features(combined)
combined = fill_missing(combined)
# The data is now ready to be fed into the model


#########################
## Preparing the model ##
#########################

# Setting the target column to "importance"
target_col = 'importance'

# Dropping the issues and label columns from the dataset wherever the label reads "train"
combined_train = combined.query('label == "train"').drop(['issues', 'label'] , axis=1)

# Splitting test and train datasets
X_train, X_test, Y_train, Y_test = \
    train_test_split(combined_train.drop([target_col], axis=1), 
                     combined_train[target_col], 
                     test_size=0.2, 
                     stratify=combined_train[target_col])

# Displaying train and test dataset lengths
print(len(X_train),' samples in training data\n', len(X_test),' samples in test data\n', )

# Creating a Classifier Dictionary, that tries out many combinations of different parameters on the dataset to achieve the best accuracy
clf_dict = {"LGBM Classifier":
            {'classifier': LGBMClassifier(),'params': [
                {'learning_rate': [0.01],
                 'n_estimators' :[1022],
                 'max_depth':[7],
                 'max_features' : [3],
                 'random_state' : [132]
                 }]},}

# Creating a datafram to keep track of the classifiers and thier accuracies on train and test sets
res_df  = pd.DataFrame()
num_clf = len(clf_dict.keys())
res_df = pd.DataFrame(data=np.zeros(shape=(num_clf, 3)),columns = ['classifier','train_score', 'test_score',])

# Looping through the classifier dictionary to go through all the combinations of parameters provided
# This for loop is not required as clf_dict has a single element, but is present so other algorithms can be added when required
count = 0
for key, clf in clf_dict.items():
    print(key, clf)

    # Starting the grid search for best model
    grid = GridSearchCV(clf["classifier"], clf["params"], refit = True, cv = 10, scoring = 'accuracy', n_jobs = -1, verbose=0)
    estimator = grid.fit(X_train,Y_train)
    train_score = estimator.score(X_train,Y_train)
    test_score = estimator.score(X_test,Y_test)
    count+=1
    
    res_df.loc[count,'classifier'] = key
    res_df.loc[count,'train_score'] = train_score
    res_df.loc[count,'test_score'] = test_score
    print(f"{key} best params: {grid.best_params_}")

# Creating another model for cross validation
xgbm = LGBMClassifier(max_depth=6, learning_rate=0.1, n_estimators=500,
                         min_child_weight=100, subsample=1.0, 
                         colsample_bytree=0.8, colsample_bylevel=0.8,
                         random_state=42, n_jobs=-1)

# Performing Cross validation and displaying the score
print("Cross Validating...")
oof_preds = cross_val_predict(xgbm, X_train, Y_train, cv=5, n_jobs=-1, method="predict")
print("cv score: ", accuracy_score(oof_preds, Y_train) * 100)

# Finally, running the model against the test set of the provided dataset
tst = combined.query('label == "test"').drop(['issues', 'label', target_col] , axis=1)
preds = grid.predict(tst)

# Saving the predicted set into a csv called "submission.csv"
sub = pd.DataFrame(columns=["appno","importance"])
sub["appno"] = test.appno
sub["importance"] = preds
sub.to_csv("submission.csv", index=False)
