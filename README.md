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

#### Step 1: Install prerequisites
This program requires python, and a few python libraries to run. To install the packages on your local machine, run the below code in your terminal after installing python and pip from the [official website](https://www.python.org/).
```py
pip install numpy, pandas, sklearn, lightgbm
```
Once this has been executed, you're all set to run the program on your local machine. When the program was run on our local machine, 8GB of ram was overloaded and a ```MemoryError``` was recieved. This is why we decided to shift to a google colab notebook, where we could run the program without downloading any prerequisites and not worrying about any MemoryError.
To run the program, the dataset files, train.csv and test.csv must be in the same directory. If not, thier path needs to be changed in the program.
