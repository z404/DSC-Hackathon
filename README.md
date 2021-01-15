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
Once this has been executed, you're all set to run the program on your local machine. When the program was run on our local machine, 8GB of ram was overloaded and a ```MemoryError``` was recieved. This is why we decided to shift to a google colab notebook, where we could run the program without downloading any prerequisites and not worrying about any MemoryError.
To run the program, the dataset files, train.csv and test.csv must be in the same directory. If not, thier path needs to be changed in the program.

#### Step 2: Loading the dataset and feature engineering
To get a better feel of the provided dataset, we need to first go through each feature in the dataset and process it so it gives valuable inputs during our training of the model. 
We first combine the train and test datasets, so it's easier to work with. Next we decided to combine all the ```issue``` columns, as there are 23 of them and combining them would give equal importance to all of the ```issue``` columns. We then converted all the string features to lower case, and then encoded the columns ```['doctypebranch','seperateopinion','typedescription']``` into integers using the label encoder from the scikit learn library.

All country names were converted to single letters, as to "universalize" the country name, and not cause unnessesary outliers. Some other columns like ```['decisiondate','introductiondate','judgementdate']``` were used to get other data, like days between judgement etc. and then dropped.

Columns with constant values (only one unique value) were also dropped. Finally, columns ```['parties.0', 'country.alpha2', 'parties.1', 'country.name', 'docname', 'appno', 'ecli', 'kpdate', 'originatingbody_name']``` were all dropped, as they did not provide any useful information or data that wasn't already present in the dataset, or had no corelation with the importance of the claim.

Now that the preprocessing was over, the dataset was ready for machine learning to be applied on it.
