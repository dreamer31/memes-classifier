import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import torch

from sklearn.metrics import confusion_matrix


def confusion_matrix_plot(y_true, y_predicted, classes):

    cf_matrix = confusion_matrix(y_true, y_predicted)
    
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) * 3,
                         index=[i for i in classes],
                         columns=[i for i in classes])

    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    plt.show()


def get_device():
    
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_images(images, labels, predicted, classes, rows=1, columns=5):
    

    axes = []
    fig = plt.figure(figsize=(10,10))
    for image in range(rows * columns):
        axes.append(fig.add_subplot(rows, columns, image + 1))
        subplot_tittle = f"{classes[labels[image]]} / {classes[predicted[image]]}"
        axes[image].set_title(subplot_tittle)
        
        img = images[image] / 2 + 0.5
        plt.imshow(np.transpose(img, (1, 2, 0)))
        
    fig.tight_layout()
    plt.show()