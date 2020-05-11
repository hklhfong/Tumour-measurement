
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
from sklearn.metrics import plot_confusion_matrix

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
    history = clf.fit(X_training,y_training,epochs=20,validation_split=0.2)
    return clf,history


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
    clf,history = build_NeuralNetwork_classifier(x_train,y_train)  
    test_loss, test_acc = clf.evaluate(x_test,  y_test, verbose=2)
    print('The accuracy of NNC:', test_acc)
    # plot training history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    
    # disp = plot_confusion_matrix(history, x_test, y_test,
    #                              cmap=plt.cm.Blues)
    # disp.ax_.set_title('Confusion matrix')
    # print(disp.confusion_matrix)
    
    
    plt.show()

    
    
    
    


