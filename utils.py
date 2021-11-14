import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import torch

from sklearn.metrics import confusion_matrix


def confusion_matrix_plot(y_true, y_predicted, classes) -> None:
    
    """
    Create a confusion matrix and plot it.
    
    :param y_true: True labels
    :param y_predicted: Predicted labels
    :param classes: List of class names
    
    :return: None
    
    """

    cf_matrix = confusion_matrix(y_true, y_predicted)
    
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) * 3,
                         index=[i for i in classes],
                         columns=[i for i in classes])

    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    plt.show()


def get_device() -> torch.device:
    
    """
    Return the device to use, gpu if is available.
    
    :return: torch.device
    """
    
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_images(images, labels, predicted, classes, rows=1, columns=5) -> None:
    
    """
    Plot images for a given set of labels and predicted labels.
    
    :param images: Images to plot
    :param labels: True labels
    :param predicted: Predicted labels
    :param classes: List of class names
    :rows: Number of rows
    :columns: Number of columns
    
    :return None
    
    """
    

    axes = []
    fig = plt.figure(figsize=(10,10))
    for image in range(rows * columns):
        axes.append(fig.add_subplot(rows, columns, image + 1))
        subplot_tittle = f"{classes[labels[image]]} / {classes[predicted[image]]}"
        axes[image].set_title(subplot_tittle)
        
        img = images[image] / 2 + 0.5
        plt.imshow((np.transpose(img, (1, 2, 0))))
        
    fig.tight_layout()
    plt.show()
    
    
    
def make_weights_for_balanced_classes(images, sub_images, nclasses) -> list:  
    
    """
    Make weights for balanced classes.
    
    :param images: Images
    :param sub_images: Sub images
    :param nclasses: Number of classes
    
    :return weights: Weights
    
    """
    
    count = [0] * nclasses                                                       
    for item in sub_images: 
        count[images[item]] += 1  
                     
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])                                 
    
    weight = [.0] * len(sub_images)
    for idx, item in enumerate(sub_images):
        weight[idx] = weight_per_class[images[item]]
                                 
    return weight  