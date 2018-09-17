import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#import the support vector machine classifier
from sklearn.svm import SVC

#get accuract of model on test set
from sklearn.metrics import accuracy_score

#read the data from the file
dataset = pd.read_csv('diabetes.csv')


#create features and labels
features = dataset.drop(['Outcome'], axis=1)
labels = dataset['Outcome']

#split the dataset into training and test set
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.25)


classifier = SVC(kernel="linear")

#fit data
classifier.fit(features_train, labels_train)

#get predicted class labels

pred = classifier.predict(features_test)

accuracy = accuracy_score(labels_test, pred)

print('Accuracy: {}'.format(accuracy))