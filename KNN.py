#Import Libraries
import numpy as np
import matplotlib.pyplot as mtp 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import time

#Timing the program
TrueStart=time.time()

#Load data
start = time.time()
print("Loading Data")
dataset = pd.read_csv("DatsetFraud.csv") #Not included in github repository as filesize is over 100MB.
end = -1*(start-time.time())
print("Data loaded in " + str(end) + " seconds\n")

#Extracting dependent and independent variables
x = dataset.iloc[:, [1,2,3,4,7]].values
y = dataset.iloc[:, [10]].values

#Splitting dataset into training and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.25, random_state=0)  

#Feature Scaling
scaler = StandardScaler()
x_train= scaler.fit_transform(x_train)    
x_test= scaler.transform(x_test)  

#Intialize Classifier and Fit KNN
classifier= KNeighborsClassifier(
                    n_neighbors=5,
                    metric='minkowski',
                    p=2
                    )  

start = time.time()
print("Fitting KNN")
classifier.fit(x_train, y_train)
end =  -1*(start-time.time())
print("KNN fit in " + str(end) + " seconds\n")

#Predicting test set result
start = time.time()
print("Predicting test set result")
y_pred= classifier.predict(x_test)  
end =  -1*(start-time.time())
print("Predicted test set result in " + str(end) + " seconds\n")

#Creating Confusion Matrix
start = time.time()
print("Creating Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
print(cm)
end =  -1*(start-time.time())
print("Confusion Matrix created in " + str(end) + " seconds\n")