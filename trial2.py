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
                             'learning_rate': [0.01, 0.1, 1.0],
                             'n_estimators' :[10, 50, 500, 1000],
                             'max_depth':[5, 3,7],
                             'max_features' : [3, 5, 7, 11]
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
res_df.iloc[1:, :]

