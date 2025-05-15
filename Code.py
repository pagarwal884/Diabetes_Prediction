import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score

#Reading csv file using pandas lib
diabetes_data = pd.read_csv('Diabetes_Prediction//diabetes.csv')

#printing and checking no. of rows and columns with null and non-null values
print(diabetes_data.head())
print(diabetes_data.info())

#In dataset of diabetes Values in Outcomes show Diabetic or Non-Diabetic 

#Finding no of rows and col in the datasets
print(diabetes_data.shape)

#Getting statistical measures of the dataset
print(diabetes_data.describe())


#Here we are counting the number of labels which is 1 and 0 from outcome col.
print(diabetes_data['Outcome'].value_counts())

#lable '0' --> non-diabetic
#lable '1' --> diabetic

#Here we are grouping all other 8 rows related to 0 and 1
print(diabetes_data.groupby('Outcome').mean())

#Seperating Data and labels
X = diabetes_data.drop(columns='Outcome' , axis=1)
Y = diabetes_data['Outcome']

