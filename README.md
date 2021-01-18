# Grievances' Importance Prediction using ML
### DSC-VIT Ice-breaker hackathon by Ius Humanum

### Abstract
We aim to predict the importance of citizens' grievances by assessing the significance of various articles, constitutional declarations, enforcement and other pertinent resources in relation to the aforementioned grievances. We will utilize Machine Learning to bring this to fruition.
This idea was inspired by the [HackerEarth Machine Learning Challenge: Resolving citizens’ grievances.](https://www.hackerearth.com/challenges/competitive/hackerearth-machine-learning-challenge-predict-grievance-importance/)

In this problem, you are given a dataset that contains grievances of various people living in a country. Your task is to predict the importance of the grievance with respect to various articles, constitutional declarations, enforcement, resources, and so on, to help the government prioritize which ones to deal with and when.

### Team Members
1. [Anish Raghavendra](https://github.com/z404)
2. [Ananya Elizabeth George](https://github.com/ananya190)
3. [Rehaan Mazid](https://github.com/Rehaan1)
4. [Raggav Subramani](https://github.com/R-droid101)

### The Approach
To successfully apply machine learning to this problem, a series of steps needs to be followed to obtain an efficient and optimal machine learning model.

#### Step 1: Installing prerequisites
This program requires python, and a few python libraries to run. To install the packages on your local machine, run the below code in your terminal after installing python and pip from the [official website](https://www.python.org/).
```py
pip install numpy, pandas, sklearn, lightgbm
```
Once this has been executed, you're all set to run the program on your local machine. When the program was run on our local machine, 8GB of RAM was overloaded and a ```MemoryError``` was recieved. This is why we decided to shift to a google colab notebook, where we could run the program without downloading any prerequisites and not worrying about any MemoryError.
To run the program, the dataset files, train.csv and test.csv must be in the same directory. If not, their path needs to be changed in the program.

#### Step 2: Loading the dataset and feature engineering
To get a better feel of the provided dataset, we need to first go through each feature in the dataset and process it so it gives valuable inputs during our training of the model. 
We first combine the train and test datasets, so it's easier to work with. Next we decided to combine all the ```issue``` columns, as there are 23 of them and combining them would give equal importance to all of the ```issue``` columns. We then converted all the string features to lower case, and then encoded the columns ```['doctypebranch','seperateopinion','typedescription']``` into integers using the label encoder from the scikit learn library.

All country names were converted to single letters, as to "universalize" the country name, and not cause unnecessary outliers. Some other columns like ```['decisiondate','introductiondate','judgementdate']``` were used to get other data, like days between judgement etc. and then dropped.

Columns with constant values (only one unique value) were also dropped. Finally, columns ```['parties.0', 'country.alpha2', 'parties.1', 'country.name', 'docname', 'appno', 'ecli', 'kpdate', 'originatingbody_name']``` were all dropped, as they did not provide any useful information or data that wasn't already present in the dataset, or had no correlation with the importance of the claim.

Now that the preprocessing was over, the dataset was ready for machine learning to be applied on it.

#### Step 3: Training the model
To start choosing the model, we first need to decide whether the problem requires the model to be a regressive model or a classification problem. As the ```importance``` column in the train set has discrete values from 1 to 4, classification is the best approach to this. On doing research, we shortlisted 3 models for training our model. ```Light GBM```, ```Random Forest Classifier``` and ```Decition Tree Classifier```. We utilized the Grid Search CV tool from scikit learn's model selection library, to run multiple algorithms on the dataset and achieve the best efficiency and accuracy, and figured out that ```Light GBM``` was providing the best results for the given hyperparameters.

After the Grid search CV is done choosing the best possible choice, we run a cross-validation test on the model and the dataset. This is important, as it shows how your model fares on many segments of the dataset, and gives you an average cross validation score, which comes very close to the actual accuracy that can be achieved on the test set.

Finally, the model is run on the test set, and the prediction is saved to a csv.

#### Step 4: Uploading to HackerEarth 
The generated csv is uploaded to the website, and an accuracy score is shown. This score is not the exact accuracy, as, according to their instructions: ```Your output will be evaluated only for 50% of the test data while the contest is running. Once the contest is over, output for the remaining 50% of the data will be evaluated and the final rank will be awarded.```

We have managed to obtain a score of 88.23390%, which has, as of now, managed to get us to the 21st spot on the leaderboard, out of 1432 participants.

### References
- [Label encoding from strings to integers](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)
- [Random Forest and LightGBM](https://datascience.stackexchange.com/questions/63322/random-forest-vs-lightgbm)
- [Light GBM Docs](https://lightgbm.readthedocs.io/en/latest/)
- [Grid Search CV Docs](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
- [Light GBM Hyperparameter Tweaking](https://towardsdatascience.com/understanding-lightgbm-parameters-and-how-to-tune-them-6764e20c6e5b)
- Google Colab Notebook

This was an amazing ice breaker experience, and we made a lot of friends and inside jokes on the way, and we thank DSC-VIT for the opportunity to take part in this hackathon
