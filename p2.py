import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from enum import Enum

SEED = 1
TEST_SIZE = 0.2
HOG_BOUNDARY = 900
NORMAL_NOISE = 916

ClassTask = Enum('Task', ['Binary', 'Multi'])
Model = Enum('Model', ['SVM', 'NeuralNetwork'])

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
    return x_train, y_train, x_test

"""
Cleans the data

args:
    df: The dataframe to clean
    remove_outliers: Whether to remove outliers or not
"""
def clean_data(x_train, y_train, x_test_final):
    x_train.dropna(inplace=True)
    if (x_train.duplicated().any()):
        print("Dropping duplicates from training set...")
        x_train = x_train.drop_duplicates()
    if (x_test_final.duplicated().any()):
        print("Dropping duplicates from test set...")
        x_test_final = x_test_final.drop_duplicates()

    # y_train.dropna(inplace=True)
    # y_train = y_train[~y_train[0].isin(['seal', 'background'])]
    return x_train, y_train, x_test_final

"""
Describe the data

args:
    df: The dataframe to describe
"""
def describe_data(df):
    print("Describing data...")
    df.hist(bins=50, figsize=(20, 15))
    plt.show()
    print("Done")

"""
Preprocess the data
    
args:
    x_train: The training data
    x_test: The test data
"""
def preprocess_data(x_train, x_test):
    print("Preprocessing data...")
    min_max_scaler = MinMaxScaler()
    standard_scaler_hog = StandardScaler()
    standard_scaler_normal = StandardScaler()

    hog_featues_train = x_train.iloc[:, :HOG_BOUNDARY]
    normal_features_train = x_train.iloc[:, HOG_BOUNDARY:NORMAL_NOISE]
    color_features_train = x_train.iloc[:, NORMAL_NOISE:]

    hog_featues_test = x_test.iloc[:, :HOG_BOUNDARY]
    normal_features_test = x_test.iloc[:, HOG_BOUNDARY:NORMAL_NOISE]
    color_features_test = x_test.iloc[:, NORMAL_NOISE:]

    hog_featues_train = standard_scaler_hog.fit_transform(hog_featues_train)
    normal_features_train = standard_scaler_normal.fit_transform(normal_features_train)
    color_features_train = min_max_scaler.fit_transform(color_features_train)

    hog_featues_test = standard_scaler_hog.transform(hog_featues_test)
    normal_features_test = standard_scaler_normal.transform(normal_features_test)
    color_features_test = min_max_scaler.transform(color_features_test)

    x_train = pd.concat([pd.DataFrame(hog_featues_train), pd.DataFrame(normal_features_train), pd.DataFrame(color_features_train)], axis=1)
    x_test = pd.concat([pd.DataFrame(hog_featues_test), pd.DataFrame(normal_features_test), pd.DataFrame(color_features_test)], axis=1)
    print("Done")
    return x_train, x_test

"""
Apply PCA to the data

args:
    df: The dataframe to apply PCA to

returns:
    df: The dataframe after applying PCA
"""
def apply_PCA(x_train, x_test):
    print("Applying PCA...")
    pca = PCA(n_components=0.99, whiten=True)
    x_train = pd.DataFrame(pca.fit_transform(x_train))
    x_test = pd.DataFrame(pca.transform(x_test))
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
def train_binary_SVM(x_train, y_train, best_params):
    print("Training SVM...")
    clf = SVC(probability=True)
    clf.set_params(**best_params)
    clf.fit(x_train, y_train)
    print("Done")
    return clf

def evaluate_model(clf, x_test, y_test):
    print("Evaluating model...")
    y_pred = clf.predict(x_test)
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("F1 score: ", f1_score(y_test, y_pred, pos_label='seal'))
    print("Precision: ", precision_score(y_test, y_pred, pos_label="seal"))
    print("Recall: ", recall_score(y_test, y_pred, pos_label='seal'))
    print("Done")

def find_best_SVC_params(skf, df):
    print("Finding best SVC params...")
    param_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']}
    # Use a subset of the data to find the best params
    _, subset = train_test_split(df, test_size=0.05, random_state=SEED, stratify=df.iloc[:, -1])
    svc = SVC(probability=True)
    grid = GridSearchCV(svc, param_grid, cv=skf, scoring='accuracy', n_jobs=-1, verbose=True)
    grid.fit(subset.iloc[:, :-1], subset.iloc[:, -1])
    print("Done")
    print("Best params: ", grid.best_params_)
    return grid.best_params_

def run_binary_classification():
    print_title('Binary Classification')
    x_train, y_train, x_test_final = read_data(ClassTask.Binary)
    x_train, y_train, x_test_final = clean_data(x_train, y_train, x_test_final)
    x_train, x_test_final = preprocess_data(x_train, x_test_final)
    x_train, x_test_final = apply_PCA(x_train, x_test_final)

    df = pd.concat([x_train, y_train], axis=1)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    svc_best_params = find_best_SVC_params(skf, df)

    # folds = skf.split(x_train, y_train)
    # for i, (train_index, test_index) in enumerate(folds):
    #     print(f"Fold {i+1}:")
    #     print(f"Train size: {len(train_index)}")
    #     print(f"Test size: {len(test_index)}")
    #     x_train, x_test = df.iloc[train_index, :-1], df.iloc[test_index, :-1]
    #     y_train, y_test = df.iloc[train_index, -1], df.iloc[test_index, -1]
    #     clf = train_binary_SVM(x_train, y_train, svc_best_params)
    #     evaluate_model(clf, x_test, y_test)

if __name__ == "__main__":
    run_binary_classification()
