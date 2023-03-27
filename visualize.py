import seaborn as sns
import matplotlib.pyplot as plt

def show_countplot(df, col):
    ax = sns.countplot(x=col, data=df)
    ax.set_title(f"Multi Classification Distribution of Labels")
    ax.set_xlabel("Labels")
    ax.set_ylabel("Count")
    plt.show()
