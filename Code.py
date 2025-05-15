import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score

# Reading csv file using pandas
diabetes_data = pd.read_csv('Diabetes_Prediction//diabetes.csv')

# Separating features and labels
X = diabetes_data.drop(columns='Outcome', axis=1)
Y = diabetes_data['Outcome']

# Standardizing the data
std = StandardScaler()
std.fit(X)
X = std.transform(X)

# Splitting the data
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=4)

# Training the model
classifier = svm.SVC(kernel='linear')
classifier.fit(X_Train, Y_Train)

# Taking User Input
print("\n--- Diabetes Prediction System ---")
name = input("Enter your name: ")
gender = input("Enter your gender (male/female): ").strip().lower()

# Handle pregnancies based on gender
if gender == 'female':
    pregnancies = int(input("Enter number of pregnancies: "))
elif gender == 'male':
    pregnancies = 0
else:
    print("Invalid gender input. Assuming male.")
    pregnancies = 0

# Collect other required inputs
glucose = float(input("Enter Glucose Level: "))
bp = float(input("Enter Blood Pressure value: "))
skin_thickness = float(input("Enter Skin Thickness value: "))
insulin = float(input("Enter Insulin Level: "))
bmi = float(input("Enter BMI value: "))
dpf = float(input("Enter Diabetes Pedigree Function value: "))
age = int(input("Enter Age: "))

# Taking input 
input_data = (pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age)

# Feature names
feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# Convert input to DataFrame with feature names
input_data_df = pd.DataFrame([input_data], columns=feature_names)

# Standardize the input
std_data = std.transform(input_data_df)

# Predict
prediction = classifier.predict(std_data)

# Output
print(f"\n{name}, based on your input data:")
if prediction[0] == 0:
    print("You are **not diabetic**.")
else:
    print("You are **diabetic**.")
