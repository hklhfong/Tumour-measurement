
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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from pandas import  pandas as pd

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)
    
    '''
    return [ (10107321, 'Law', 'HoFong'), (10031017, 'Kiki', 'Mutiara') ]

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
    #encode label result
    y = encode(y)   
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

    min_samples_leaf = [1, 2, 4]
    hyperparameters = dict(min_samples_leaf=min_samples_leaf)


    clf = DecisionTreeClassifier()
    clf = GridSearchCV(clf, hyperparameters, cv=10)
    clf = clf.fit(X_training,y_training)
    print('Best sample leaf:', clf.best_estimator_.get_params()['min_samples_leaf'])
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
    
    n_neighbors = list(range(1,30))

    hyperparameters = dict(n_neighbors=n_neighbors)

    clf = KNeighborsClassifier()
    clf = GridSearchCV(clf, hyperparameters, cv=10)
    clf = clf.fit(X_training,y_training)
    print('Best n_neighbors:', clf.best_estimator_.get_params()['n_neighbors'])
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
    
    C = [0.1,1, 10, 100]

    hyperparameters = dict(C=C)

    clf = SVC()
    clf = GridSearchCV(clf, hyperparameters, cv=10)
    clf = clf.fit(X_training,y_training)
    print('Best C:', clf.best_estimator_.get_params()['C'])
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

    hyperparameters = dict(hidden_layer_sizes = [(20, 10), (30, 20)])

    clf = MLPClassifier(max_iter=300)
    clf = GridSearchCV(clf, hyperparameters, cv=10)
    clf = clf.fit(X_training, y_training)
    print('Best hidden layer size:', clf.best_estimator_.get_params()['hidden_layer_sizes'])
    return clf


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def encode(y):
    """
    Encode the output result from y training and y testing set
    deep learning neural network model is a model that only accept numeric type
    on both input and output parameter.
    In order to put the output in model,We need to transfrom data to encoded
    lable.
    
    Parameters
    ----------
    y : Array of object
        unprocessed tumour result.

    Returns
    -------
    y : Array of int32
        tumour result of training set (encoded in M = 0,B = 1).
    """
    le = LabelEncoder()
    le.fit(y)
    print(list(le.classes_))
    y = le.transform(y)
    return y
    
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if __name__ == "__main__":
    pass
    # Write a main part that calls the different 
    # functions to perform the required tasks and repeat your experiments.
    # Call your functions here 

    print(my_team())

    X,y = prepare_dataset(r'/content/medical_records.data.txt')
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    #Display the Decision Tree prediction accuracy
    clf_DT = build_DecisionTree_classifier(x_train,y_train)
    output_DT = clf_DT.predict(x_test)
    score = clf_DT.score(x_test, y_test)
    print('The accuracy of decision tree classifier:' + str(score))
    
    #Display the KNN prediction accuracy
    clf_KNN = build_NearrestNeighbours_classifier(x_train,y_train)
    output_KNN = clf_KNN.predict(x_test)
    acc = accuracy_score(y_test, output_KNN)
    print('The accuracy of KNN:' + str(acc))
    
    #Display the SVM prediction accuracy
    clf_SVM = build_SupportVectorMachine_classifier(x_train,y_train)
    output_SVM = clf_SVM.predict(x_test)
    acc = accuracy_score(y_test, output_SVM)
    print('The accuracy of SVM:' + str(acc))
    
    #Display the NNC prediction accuracy
    clf = build_NeuralNetwork_classifier(x_train,y_train) 
    pred = clf.predict(x_test)
    cm = confusion_matrix(pred, y_test)
    diagonal_sum = cm.trace()
    sum_of_all_elements = cm.sum()
    accuracy = diagonal_sum / sum_of_all_elements
    print('The accuracy of Neural Network:' + str(accuracy))

    #Plot confusion matrices
    # Decision Tree CM
    disp_DT = confusion_matrix(y_test, output_DT)
    fig = plt.figure(figsize=[20, 6])
    plt.title("Confusion Matrix DT")
    ax = fig.add_subplot(1, 2, 1)
    c = ConfusionMatrixDisplay(disp_DT, display_labels=range(10))
    c.plot(ax = ax)

    # KNN CM
    disp_KNN = confusion_matrix(y_test, output_KNN)
    fig = plt.figure(figsize=[20, 6])
    plt.title("Confusion Matrix KNN")
    ax = fig.add_subplot(1, 2, 1)
    c = ConfusionMatrixDisplay(disp_KNN, display_labels=range(10))
    c.plot(ax = ax)

    # SVM CM
    disp_SVM = confusion_matrix(y_test, output_SVM)
    fig = plt.figure(figsize=[20, 6])
    plt.title("Confusion Matrix SVM")
    ax = fig.add_subplot(1, 2, 1)
    c = ConfusionMatrixDisplay(disp_SVM, display_labels=range(10))
    c.plot(ax = ax)

    # NNC CM
    fig = plt.figure(figsize=[20, 6])
    plt.title("Confusion Matrix Neural Network")
    ax = fig.add_subplot(1, 2, 1)
    c = ConfusionMatrixDisplay(cm, display_labels=range(10))
    c.plot(ax = ax)
    
    plt.show()
