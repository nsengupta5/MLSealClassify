import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

"""
Show the countplot of the labels

args:
    df: dataframe
    col: column name
"""
def show_countplot(df, col):
    ax = sns.countplot(x=col, data=df)
    ax.set_title(f"Multi Classification Distribution of Labels")
    ax.set_xlabel("Labels")
    ax.set_ylabel("Count")
    plt.show()

"""
Show the confusion matrix

args:
    y_true: true labels
    y_pred: predicted labels
    labels: labels
"""
def show_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.show()

"""
Show the scree plot

args:
    pca: pca object
"""
def show_scree_plot(pca):
    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(explained_variance)
    plt.title("Scree Plot")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.show()

"""
Show the distribution plot

args:
    df: dataframe
"""
def show_distribution_plot(df):
    df.hist(bins=50, figsize=(20, 15))
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.show()
