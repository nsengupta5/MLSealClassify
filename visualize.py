import seaborn as sns
import matplotlib.pyplot as plt

def show_countplot(df, col):
    sns.countplot(x=col, data=df)
    plt.show()
