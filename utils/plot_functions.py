import numpy as np
from matplotlib import pyplot as plt


def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots()
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    ax.set_xticklabels([''] + class_names)
    ax.set_yticklabels([''] + class_names)

    # Display the values on the confusion matrix
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, f'{val}', ha='center', va='center', color='red')

    plt.xlabel('Predicted')
    plt.ylabel('True')

    return fig
