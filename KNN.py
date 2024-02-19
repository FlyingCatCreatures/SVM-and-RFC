#Import Libraries
import numpy as np
import matplotlib.pyplot as mtp 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
import joblib

#Timing the program
TrueStart=time.time()

#Load data
start = time.time()
print("Loading Data")
dataset = pd.read_csv("DatsetFraud.csv") #Not included in github repository as filesize is over 100MB.
end = -1*(start-time.time())
print("Data loaded in " + str(end) + " seconds\n")

#Separate labels
data_labels = np.asarray(dataset.isFraud)

#Encode labels
label_encoder = LabelEncoder()
label_encoder.fit(data_labels)


#Apply encoding to labels
data_labels = label_encoder.transform(data_labels)


#Drop labels from input data
data_selectedColums = dataset.drop(['isFraud', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest' ], axis=1)

#Extract features
start = time.time()
print("Extracting features from dataset")
data_features = data_selectedColums.to_dict(orient='records')
vectorizer = DictVectorizer()
data_features = vectorizer.fit_transform(data_features).toarray()
end =  -1*(start-time.time())
print("Features extracted from dataset in " + str(end) + " seconds\n")


#Splitting dataset into training and test set
x_train, x_test, y_train, y_test = train_test_split(data_features, data_labels, test_size= 0.99, random_state=0)  

#Feature Scaling
scaler = StandardScaler()
x_train= scaler.fit_transform(x_train)    
x_test= scaler.transform(x_test)  

#Intialize Classifier and Fit KNN
classifier= joblib.load("Classifier(KNN).joblib")
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
joblib.dump(classifier, "Classifier(KNN).joblib")

