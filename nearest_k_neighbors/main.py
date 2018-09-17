import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import  train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('diabetes.csv')

print(dataset.head(5))
print(dataset.shape)

print(dataset.describe())

#check if there are correlations, i.e. redundant features
corr = dataset.corr() #data frame correlation function
fig, ax = plt.subplots(figsize=(13,13))

ax.matshow(corr) #color code the rectangles by correlation value

plt.xticks(range(len(corr.columns)), corr.columns) #draw x tick marks
plt.yticks(range(len(corr.columns)), corr.columns) #draw y tick marks
plt.show()

#separate columns intofeatures
features = dataset.drop(['Outcome'], axis=1)
labels = dataset['Outcome']

features_train, feature_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.25)
classifier = KNeighborsClassifier()

classifier.fit(features_train, labels_train)

pred = classifier.predict(feature_test)
accuracy = accuracy_score(labels_test, pred)

print('Accuracy: {}'.format(accuracy))