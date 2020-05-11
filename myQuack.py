
'''

2020 

Scaffolding code for the Machine Learning assignment. 

You should complete the provided functions and add more functions and classes as necessary.
 
You are strongly encourage to use functions of the numpy, sklearn and tensorflow libraries.

You are welcome to use the pandas library if you know it.


'''
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from pandas import  pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import LabelEncoder

import seaborn as sns

from sklearn.preprocessing import StandardScaler

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)
    
    '''
    return [ (10107321, 'Law', 'HoFong'), (1234568, 'Kiki', 'Hopper') ]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def prepare_dataset(dataset_path):
    '''  
    Read a comma separated text file where 
	- the first field is a ID number 
	- the second field is a class label 'B' or 'M'
	- the remaining fields are real-valued

    Return two numpy arrays X and y where 
	- X is two dimensional. X[i,:] is the ith example
	- y is one dimensional. y[i] is the class label of X[i,:]
          y[i] should be set to 1 for 'M', and 0 for 'B'

    @param dataset_path: full path of the dataset text file

    @return
	X,y
    '''
    
    #read the dataset
    dataset = pd.read_csv(dataset_path, header=None)
    X = dataset.drop(dataset.columns[1], axis=1)
    #standardizing X value for better prediction
    sc = StandardScaler()
    X = sc.fit_transform(X)
    #change type to string to make sure corrrect type
    y = dataset.iloc[:,1].astype(str)
    print(X.shape)
    print(y.shape)
    return X,y
    

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_DecisionTree_classifier(X_training, y_training):
    '''  
    Build a Decision Tree classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    clf = DecisionTreeClassifier()
    clf = clf.fit(X_training,y_training)
    return clf

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NearrestNeighbours_classifier(X_training, y_training):
    '''  
    Build a Nearrest Neighbours classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    
    clf = KNeighborsClassifier()
    clf = clf.fit(X_training,y_training)
    return clf
    

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_SupportVectorMachine_classifier(X_training, y_training):
    '''  
    Build a Support Vector Machine classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    
    clf = SVC()
    clf = clf.fit(X_training,y_training)
    return clf

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NeuralNetwork_classifier(X_training, y_training):
    '''  
    Build a Neural Network classifier (with two dense hidden layers)  
    based on the training set X_training, y_training.
    Use the Keras functions from the Tensorflow library

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    inputs = keras.Input(shape=(31), name='input')
    x = layers.Dense(20, activation='relu', name='dense_1')(inputs)
    x = layers.Dense(10, activation='relu', name='dense_2')(x)
    outputs = layers.Dense(1,activation='sigmoid', name='output')(x)
    clf = keras.Model(inputs=inputs, outputs=outputs, name='TwoDense')
    clf.summary()
    clf.compile(loss='binary_crossentropy', optimizer='adam',metrics =['accuracy'])
    clf.fit(X_training,y_training,epochs=100)
    return clf


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def plot_roc(y_test,y_score):
    y_test = y_test.array
    
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(y_test.shape[0]):
        fpr[i], tpr[i], thresholds  = roc_curve(y_test, y_score,pos_label='B')
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    plt.figure()
    lw = 2
    plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    
    
def encode(y_training,y_test):
    le = LabelEncoder()
    le.fit(y_training)
    print(list(le.classes_))
    y_training = le.transform(y_training)
    y_test = le.transform(y_test)
    return y_training,y_test,le
    
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if __name__ == "__main__":
    pass
    # Write a main part that calls the different 
    # functions to perform the required tasks and repeat your experiments.
    # Call your functions here 

    print(my_team())
    #Plot tree graph and display the prediction accuracy
    X,y = prepare_dataset(r'C:\Users\user\Downloads\medical_records.data.txt')
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    clf = build_DecisionTree_classifier(x_train,y_train)
    score = clf.score(x_test, y_test)
    print('The accuracy of decision tree classifier:' + str(score))
    fig, ax = plt.subplots(figsize=(50, 50)) 
    tree.plot_tree(clf, ax=ax);
    
    #Display the KNN prediction accuracy
    clf = build_NearrestNeighbours_classifier(x_train,y_train)
    output = clf.predict(x_test)
    acc = accuracy_score(y_test, output)
    print('The accuracy of KNN:' + str(acc))
    
    #Display the SVM prediction accuracy
    clf = build_SupportVectorMachine_classifier(x_train,y_train)
    output = clf.predict(x_test)
    acc = accuracy_score(y_test, output)
    print('The accuracy of SVM:' + str(acc))
    
    #Display the NNC prediction accuracy
    y_train,y_test,le = encode(y_train,y_test)
    
    clf = build_NeuralNetwork_classifier(x_train,y_train)
    print(list(le.classes_))
    #inverse transform the label
    test_loss, test_acc = clf.evaluate(x_test,  y_test, verbose=2)
    print('\nTest accuracy:', test_acc)
    
    
    
    
    


