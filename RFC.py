import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score
import time
import joblib

#Time entire program
TrueStart=time.time()

#Load data
start = time.time()
print("Loading Data")
dataset = pd.read_csv("Short.csv") #Not included in github repository as filesize is over 100MB.
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

#Split data
features_train, features_test, labels_train, labels_test = train_test_split(data_features, data_labels, test_size=0.50, random_state=507) #Random_state is just a seed value. Can be any number

#Initialize Classifier
classifier = RandomForestClassifier(
                        #min_samples_split=4, #Can be fucked around with
                        criterion='log_loss', #Testing this
                        class_weight={0:1,1:1000} #Weigh positives more heavily
                        )   


#Fit SVM
start = time.time()
print("Fitting SVM")
classifier.fit(features_train, labels_train)
end =  -1*(start-time.time())
print("SVM fit in " + str(end) + " seconds\n")

#Test SVM
start = time.time()
print("Testing SVM")
acc_test = classifier.score(features_test, labels_test)
end =  -1*(start-time.time())
print("SVM tested in " + str(end) + " seconds\n")

#Generate custom metrics
start = time.time()
print("Generating metric data")
stuff =classifier.predict(features_test)
TruePositives=0
FalseNegatives=0
FalsePositives = 0
TrueNegatives = 0
for i in range(len(stuff)):
    if(labels_test[i] == 1):
        if(stuff[i]==1):
            TruePositives+=1
        else:
            FalseNegatives+=1
    else:
        if(stuff[i]==1):
            FalsePositives+=1
        else:
            TrueNegatives+=1
end =  -1*(start-time.time())
print("Metrics generated in " + str(end) + " seconds")
print("Test:")
print(FalseNegatives, "   ", TruePositives)
print(FalsePositives, "   ", TrueNegatives)

#Get precision and recall
start = time.time()
print("\nCalculating Precision & Recall")
precision = precision_score(labels_test, classifier.predict(features_test), average="weighted")
recall = recall_score(labels_test, classifier.predict(features_test), average="weighted")
end =  -1*(start-time.time())
print("Precision & Recall calculated in " + str(end) + " seconds\n\n")

#Print Results
print("----------------------------------------------------------------------------------------------------")
print("Test Accuracy:", acc_test)
print ("Precision:", precision) 
print ("Recall:", recall) 
print("----------------------------------------------------------------------------------------------------")


TrueEndSeconds = -1*(TrueStart-time.time())
TrueEndMinutes = (TrueEndSeconds - TrueEndSeconds%60)/60
print("\n\nTotal runtime: " + str(int(TrueEndMinutes)) + ":" + str(TrueEndSeconds%60))

#Save the classifier
joblib.dump(classifier, "./Classifier.joblib")

#Example for how to load classifier
#loaded_classifier: RandomForestClassifier = joblib.load("./Classifier.joblib")
