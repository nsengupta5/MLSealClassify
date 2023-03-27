import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sys import argv
from enum import Enum

SEED = 1
TEST_SIZE = 0.2
HOG_BOUNDARY = 900
NORMAL_NOISE = 916

ClassTask = Enum('Task', ['Binary', 'Multi'])
Model = Enum('Model', ['SVM', 'NN'])

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

    # Split the data into HOG, normal and color features
    hog_featues_train = x_train.iloc[:, :HOG_BOUNDARY]
    normal_features_train = x_train.iloc[:, HOG_BOUNDARY:NORMAL_NOISE]
    color_features_train = x_train.iloc[:, NORMAL_NOISE:]

    # Split the data into HOG, normal and color features
    hog_featues_test = x_test.iloc[:, :HOG_BOUNDARY]
    normal_features_test = x_test.iloc[:, HOG_BOUNDARY:NORMAL_NOISE]
    color_features_test = x_test.iloc[:, NORMAL_NOISE:]

    # Scale the data
    hog_featues_train = standard_scaler_hog.fit_transform(hog_featues_train)
    normal_features_train = standard_scaler_normal.fit_transform(normal_features_train)
    color_features_train = min_max_scaler.fit_transform(color_features_train)

    # Scale the data
    hog_featues_test = standard_scaler_hog.transform(hog_featues_test)
    normal_features_test = standard_scaler_normal.transform(normal_features_test)
    color_features_test = min_max_scaler.transform(color_features_test)

    # Concatenate the data
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
    # Account for 99% of the variance
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
def train_binary_model(df, folds, model, task, best_params=None):
    print(f"Training {model} model...")
    crs = []
    clf = None
    for i, (train_index, test_index) in enumerate(folds):
        print(f"Training fold {i+1}...")
        # Split the data
        x_train, x_test = df.iloc[train_index, :-1], df.iloc[test_index, :-1]
        y_train, y_test = df.iloc[train_index, -1], df.iloc[test_index, -1]
        # Get the model
        if model == Model.NN:
            clf = get_NN_model(x_train, y_train, task, best_params)
        elif model == Model.SVM:
            clf = get_SVM_model(x_train, y_train, task, best_params)
        # Evaluate the model
        cr = evaluate_model(clf, x_test, y_test, i+1)
        crs.append(cr)

    # Print the results
    print_training_results(crs)
    print("Done")
    return clf

"""
Get the neural network model

args:
    x_train: The training data
    y_train: The training labels

returns:
    clf: The trained neural network model
"""
def get_NN_model(x_train, y_train, task, best_params=None):
    if best_params is None:
        # Set the parameters to the best parameters found by grid search previously
        if task == ClassTask.Binary:
            clf = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000, alpha=0.1, solver='adam', random_state=1, learning_rate_init=0.001)
        elif task == ClassTask.Multi:
            # TODO
            pass
    else:
        clf = MLPClassifier(**best_params)
    clf.fit(x_train, y_train)
    return clf

"""
Get the SVM model

args:
    x_train: The training data
    y_train: The training labels

returns:
    clf: The trained SVM model
"""
def get_SVM_model(x_train, y_train, task, best_params=None):
    if best_params is None:
        # Set the parameters to the best parameters found by grid search previously
        if task == ClassTask.Binary:
            clf = SVC(C=10, gamma=0.001, probability=True, class_weight='balanced')
        elif task == ClassTask.Multi:
            # TODO
            pass
    else:
        clf = SVC(**best_params)
    clf.fit(x_train, y_train)
    return clf

"""
Find the best parameters for the SVM model

args:
    skf: The cross validation object
    df: The dataframe to use for finding the best params

returns:
    best_params: The best params for the SVM model
"""
def find_best_SVC_params(skf, df):
    print("Finding best SVC params...")
    # Set the parameters to search over
    param_grid = {'C': [0.1, 1, 10, 100, 1000], 
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
                  'kernel': ['rbf'],
                  'class_weight': ['balanced'],
                  'probability': [True],
                  'random_state': [SEED],
                 }

    # Use a validation set to find the best parameters
    _, subset = train_test_split(df, test_size=0.05, random_state=SEED, stratify=df.iloc[:, -1])
    svc = SVC(probability=True)

    # Find the best parameters
    grid = GridSearchCV(svc, param_grid, cv=skf, scoring='balanced_accuracy', n_jobs=-1, verbose=True)
    grid.fit(subset.iloc[:, :-1], subset.iloc[:, -1])
    print("Best params for SVC: ", grid.best_params_)
    print("Done")
    return grid.best_params_

"""
Find the best parameters for the neural network model

args:
    skf: The cross validation object
    df: The dataframe to use for finding the best params

returns:
    best_params: The best params for the neural network model
"""
def find_best_NN_params(skf, df):
    print("Finding best NN params...")
    # Set the parameters to search over
    param_grid = {'hidden_layer_sizes': [(50), (50, 50), (50, 50, 50), (100), (100, 100), (100, 100, 100)],
                  'max_iter': [1000, 2000, 3000],
                  'alpha': [0.0001, 0.001, 0.01, 0.1],
                  'learning_rate_init': [0.0001, 0.001, 0.01, 0.1],
                  'solver': ['adam'],
                  'random_state': [SEED],
                  }
    # Use a validation set to find the best parameters
    _, subset = train_test_split(df, test_size=0.05, random_state=SEED, stratify=df.iloc[:, -1])
    nn = MLPClassifier()

    # Find the best parameters
    grid = GridSearchCV(nn, param_grid, cv=skf, scoring='balanced_accuracy', n_jobs=-1, verbose=True)
    grid.fit(subset.iloc[:, :-1], subset.iloc[:, -1])
    print("Best params for NN: ", grid.best_params_)
    print("Done")
    return grid.best_params_

"""
Print the training results

args:
    crs: The list of classification reports
"""
def print_training_results(crs):
    print("Training Results:")
    print("Average Accuracy: ", np.mean([cr['accuracy'] for cr in crs]))

    print()
    cr_keys = list(crs[0].keys())

    # Print the results for each class
    for ck in cr_keys:
        if ck != 'accuracy':
            print(f"{ck.upper()}")
            print("=" * len(ck))
            precisions = []
            recalls = []
            f1_scores = []
            # Get the average precision, recall, and f1 score for each class
            for cr in crs:
                precisions.append(cr[ck]['precision'])
                recalls.append(cr[ck]['recall'])
                f1_scores.append(cr[ck]['f1-score'])
            print("Average Precision: ", np.mean(precisions))
            print("Average Recall: ", np.mean(recalls))
            print("Average F1 Score: ", np.mean(f1_scores))
            print()

"""
Evaluate the model on the test set

args:
    clf: The trained model
    x_test: The test set
    y_test: The test labels
    fold_idx: The index of the fold being evaluated

Returns
    The classification report as a dictionary
"""
def evaluate_model(clf, x_test, y_test, fold_idx):
    print(f"Evaluating model on fold {fold_idx}...")
    y_pred = clf.predict(x_test)
    cr_dict = classification_report(y_test, y_pred, output_dict=True)
    cr_str = classification_report(y_test, y_pred)
    print(cr_str)
    return cr_dict

"""
Output the predictions for the test set to a csv file

args::
    df: The training data
    x_test: The test data
    model: The model to use
    filename: The name of the file to output to
"""
def output_predictions(df, x_test, model, task, filename):
    print("Outputting predictions...")
    if model == Model.NN:
        clf = get_NN_model(df.iloc[:, :-1], df.iloc[:, -1], task)
    elif model == Model.SVM:
        clf = get_SVM_model(df.iloc[:, :-1], df.iloc[:, -1], task)
    # Train the model on the entire training set
    y_pred = clf.predict(x_test)
    y_pred = pd.DataFrame(y_pred)
    # Output the predictions to a csv file
    y_pred.to_csv(filename, index=False)
    print("Done")

if __name__ == "__main__":
    task = ClassTask.Binary
    model = Model.NN
    debug = False

    # Parse the command line arguments
    arg_len = len(argv)
    if arg_len == 1:
        # Set the task
        if argv[0] == 'binary':
            task = ClassTask.Binary
        elif argv[0] == 'multi':
            task = ClassTask.Multi
    if arg_len == 2:
        # Set the model
        if argv[1] == 'svm':
            model = Model.SVM
        elif argv[1] == 'nn':
            model = Model.NN
    if arg_len == 3:
        # Set the debug flag
        if argv[2] == 'debug':
            debug = True

    # Read the data
    x_train = None
    y_train = None
    x_test_final = None
    if task == ClassTask.Binary:
        print_title('Binary Classification')
        x_train, y_train, x_test_final = read_data(ClassTask.Binary)
    elif task == ClassTask.Multi:
        print_title('Multi-Class Classification')
        x_train, y_train, x_test_final = read_data(ClassTask.Multi)

    x_train, y_train, x_test_final = clean_data(x_train, y_train, x_test_final)
    x_train, x_test_final = preprocess_data(x_train, x_test_final)
    x_train, x_test_final = apply_PCA(x_train, x_test_final)

    # Use stratified k-fold cross validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = skf.split(x_train, y_train)

    # Concatenate the training data and labels
    df = pd.concat([x_train, y_train], axis=1)

    # Find the best parameters for the models
    if model == Model.NN:
        nn_best_params = None
        if debug:
            nn_best_params = find_best_NN_params(skf, df)
        # Train the NN and output the predictions
        train_binary_model(df, folds, model, task, nn_best_params)
        filename = ""
        if task == ClassTask.Binary:
            filename = 'data/binary/Y_test_NN.csv'
        elif task == ClassTask.Multi:
            filename = 'data/multi/Y_test_NN.csv'
        output_predictions(df, x_test_final, Model.NN, ClassTask.Binary, filename)

    elif model == Model.SVM:
        svc_best_params = None
        if debug:
            svc_best_params = find_best_SVC_params(skf, df)
        # Train the SVM and output the predictions
        train_binary_model(df, folds, model, task, svc_best_params)
        filename = ""
        if task == ClassTask.Binary:
            filename = 'data/binary/Y_test_SVM.csv'
        elif task == ClassTask.Multi:
            filename = 'data/multi/Y_test_SVM.csv'
        output_predictions(df, x_test_final, Model.SVM, ClassTask.Multi, filename)
