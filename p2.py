import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from enum import Enum

ClassTask = Enum('Task', ['Binary', 'Multi'])
Model = Enum('Model', ['LogisticRegression', 'SVM', 'RandomForest', 'NeuralNetwork'])

"""
Prints a title with a line under it

args:
    title: The title to print
"""
def print_title(title):
    print(title)
    print('=' * len(title))

"""
Reads the data from the csv files

args:
    task: The type of classification task to perform

returns:
    x_train: The training data
    x_test: The test data
    y_train: The training labels
"""
def read_data(task):
    x_train = None
    x_test = None
    y_train = None
    if task == ClassTask.Binary:
        # TODO: Remember to change the path to lab path
        print('Reading binary classification data...')
        x_train = pd.read_csv('data/binary/X_train.csv', header=None)
        x_test = pd.read_csv('data/binary/X_test.csv', header=None)
        y_train = pd.read_csv('data/binary/Y_train.csv', header=None)
    elif task == ClassTask.Multi:
        print('Reading multi classification data...')
        x_train = pd.read_csv('data/multi/X_train.csv', header=None)
        x_test = pd.read_csv('data/multi/X_test.csv', header=None)
        y_train = pd.read_csv('data/multi/Y_train.csv', header=None)
    print('Done')
    return x_train, x_test, y_train

"""
Cleans the data

args:
    df: The dataframe to clean
    remove_outliers: Whether to remove outliers or not
"""
def clean_data(df, remove_outliers=False):
    df.dropna(inplace=True)
    if (df.duplicated().any()):
        print("Dropping duplicates...")
        df = df.drop_duplicates()
    if remove_outliers:
        print("Removing outliers...")
    print("Done")

"""
Cleans all the data for the given task

args:
    x_train: The training data
    x_test: The test data
    y_train: The training labels
"""
def clean_all_data(x_train, x_test, y_train):
    print("Cleaning data...")
    clean_data(x_train)
    clean_data(x_test)
    clean_data(y_train)
    print("Done")

"""
Describe the data

args:
    df: The dataframe to describe
"""
def describe_data(df):
    print("Describing data...")
    print(df.describe())
    print("Done")

"""
Preprocess the data
    
args:
    x_train: The training data
    x_test: The test data
"""
def preprocess_data(x_train, x_test):
    print("Preprocessing data...")
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    print("Done")
    return x_train, x_test

"""
Train the neural network model for binary classification

args:
    x_train: The training data
    y_train: The training labels

returns:
    clf: The trained model
"""
def train_binary_NN(x_train, y_train):
    print("Training Neural Network...")
    clf = MLPClassifier(hidden_layer_sizes=(100, 100, 100), max_iter=1000)
    clf.fit(x_train, y_train)
    print("Done")
    return clf

"""
Train the SVM model for binary classification

args:
    x_train: The training data
    y_train: The training labels

returns:
    clf: The trained model
"""
def train_binary_SVM(x_train, y_train):
    print("Training SVM...")
    clf = SVC()
    clf.fit(x_train, y_train)
    print("Done")
    return clf

if __name__ == "__main__":
    print_title('Binary Classification')
    x_train, x_test, y_train = read_data(ClassTask.Binary)
    clean_all_data(x_train, x_test, y_train)
    preprocess_data(x_train, x_test)
    print_title('Multiclass Classification')
