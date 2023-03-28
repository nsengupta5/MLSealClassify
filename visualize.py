import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def show_countplot(df, col):
    ax = sns.countplot(x=col, data=df)
    ax.set_title(f"Multi Classification Distribution of Labels")
    ax.set_xlabel("Labels")
    ax.set_ylabel("Count")
    plt.show()

def show_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.show()
