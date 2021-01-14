import pandas as pd
import numpy as np
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

test['importance']=-1
train['label'] = 'train'
test['label'] = 'test'
combined = pd.concat([train,test],axis=0)

def combine_issues(df):
#     combining all the issues columns to form one issue column
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


def lowercase_texts(df):
    print('converting all text columns in lowercase.',)
    for col in combined.columns:
        if combined[col].dtype=='object':
            combined[col] = combined[col].str.lower()
    return df


def universalize_countries(df):
#     converting all the countries to single symbolic numerical value.(eg - Albania, albania, abl, ab -> 1)
    country_dict_A = df[['respondentOrderEng','country.name']].set_index('country.name').T.to_dict('list')
    country_dict_C = df[['respondentOrderEng','respondent.0']].set_index('respondent.0').T.to_dict('list')    
    country_dict = {}
    for d in (country_dict_A, country_dict_C):#, country_dict_C): #, country_dict_D, country_dict_E, country_dict_F): 
        country_dict.update(d)
        
    country_dict = {k: v for k, v in country_dict.items() if pd.notna(k)}
    df['respondent.0'] = df['respondent.0'].apply(lambda x: country_dict[x][0])
    df['respondent.1'] = df['respondent.1'].apply(lambda x: country_dict[x][0] if pd.notnull(x) else x)
    df['respondent.2'] = df['respondent.2'].apply(lambda x: country_dict[x][0] if pd.notnull(x) else x)
    df['respondent.3'] = df['respondent.3'].apply(lambda x: country_dict[x][0] if pd.notnull(x) else x)
    df['respondent.4'] = df['respondent.4'].apply(lambda x: country_dict[x][0] if pd.notnull(x) else x)
    del df['respondentOrderEng']
    return df

def remove_constant_values(df):
#     this function removes redundant constant features.
    print('Removing constant columns -> ',)
    for col in df.columns:
        if df[col].nunique()==1:
            print(col,end=', ' )
            del df[col]
    return df

def remove_unwanted_features(df):
#     these features dont add any valueable signal to the data.
    remove_cols =['parties.0', 'country.alpha2', 'parties.1', 'country.name', 'docname', 'appno', 'ecli', 'kpdate', 'originatingbody_name']
    for col in remove_cols:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
    return df

  
def featurize_columns(df):
#     making new columns.
    df['itemid'] = df['itemid'].apply(lambda x: x[4:7])
    df['sharepointid'] = df['sharepointid'].apply(lambda x: str(x)[:3])
    df['total_respondents'] = 5- df[['respondent.0','respondent.1','respondent.2','respondent.3','respondent.4']].isna().sum(axis=1)

    return df

def featurize_date_columns(df):
    #     making new columns based on dates.
    df['daysbetween_intro_decision'] = (pd.to_datetime(df['decisiondate']) - pd.to_datetime(df['introductiondate'])).dt.days
    df['daysbetween_intro_judgement'] = (pd.to_datetime(df['judgementdate']) - pd.to_datetime(df['introductiondate'])).dt.days
    df['daysbetween_decision_judgement'] = (pd.to_datetime(df['judgementdate']) - pd.to_datetime(df['decisiondate'])).dt.days
    df.drop(['decisiondate','introductiondate','judgementdate'], axis=1, inplace=True)
    return df

def encoding(df):
    df['doctypebranch'] = le.fit_transform(df['doctypebranch'])
    df['separateopinion'] = le.fit_transform(df['separateopinion'])
    df['typedescription'] = le.fit_transform(df['typedescription'])
    return df

def fill_missing(df):
    for col in df.columns:
        if col not in ['label', 'issues']:
            df[col].fillna(0,inplace=True)
            df[col] = df[col].astype('int')
    return df



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

#####################################################


from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score


target_col = 'importance'



combined_train = combined.query('label == "train"').drop(['issues', 'label'] , axis=1)

# split test and train
X_train, X_test, Y_train, Y_test = \
    train_test_split(combined_train.drop([target_col], axis=1), 
                     combined_train[target_col], 
                     test_size=0.2, 
                     stratify=combined_train[target_col])
print(len(X_train),' samples in training data\n',
      len(X_test),' samples in test data\n', )

clf_dict = {"LGBM Classifier": 
            {'classifier': LGBMClassifier(),
                 'params': [
                            {
                             'learning_rate': [0.01],
                             'n_estimators' :[1022],
                             'max_depth':[7],
                             'max_features' : [3],
                             'random_state' : [132]
                            }
                           ]
            },
           }

res_df  = pd.DataFrame()
num_clf = len(clf_dict.keys())
res_df = pd.DataFrame(
    data=np.zeros(shape=(num_clf, 3)),
    columns = ['classifier',
                   'train_score', 
                   'test_score',
            ]
)

count = 0
for key, clf in clf_dict.items():
    print(key, clf)

    grid = GridSearchCV(clf["classifier"],
                        clf["params"],
                        refit=True,
                        cv=10,
                        scoring = 'accuracy',
                        n_jobs = -1,
                        verbose=0
                        
                       )
    estimator = grid.fit(
                        X_train,
                        Y_train)
    train_score = estimator.score(X_train,
                                      Y_train)
    test_score = estimator.score(X_test,
                                 Y_test)
    count+=1
    
    res_df.loc[count,'classifier'] = key
    res_df.loc[count,'train_score'] = train_score
    res_df.loc[count,'test_score'] = test_score
    print(f"{key} best params: {grid.best_params_}")

#xgbm = LGBMClassifier(max_depth=6, learning_rate=0.1, n_estimators=500,
#                         min_child_weight=100, subsample=1.0, 
#                         colsample_bytree=0.8, colsample_bylevel=0.8,
#                         random_state=42, n_jobs=-1)


#print("Cross Validating...")
#oof_preds = cross_val_predict(xgbm, X_train, Y_train, cv=5, n_jobs=-1, method="predict")
#print("cv score: ", accuracy_score(oof_preds, Y_train) * 100)

tst = combined.query('label == "test"').drop(['issues', 'label', target_col] , axis=1)
preds = grid.predict(tst)

sub = pd.DataFrame(columns=["appno","importance"])
sub["appno"] = test.appno
sub["importance"] = preds

sub.to_csv("submission.csv", index=False)


#132 - 88.23390
#145 - 88.07500 
input()
