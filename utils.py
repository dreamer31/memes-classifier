import pandas as pd
import numpy as np
import seaborn as sn
import torch
import matplotlib.pyplot as plt
import numpy as np
import itertools

from sklearn.metrics import confusion_matrix



def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):


    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="black" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="black" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


def confusion_matrix_plot(y_true, y_predicted, classes) -> None:
    
    """
    Create a confusion matrix and plot it.
    
    :param y_true: True labels
    :param y_predicted: Predicted labels
    :param classes: List of class names
    
    :return: None
    
    """

    label = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    cf_matrix = confusion_matrix(y_true, y_predicted, labels=label)
    cm_normalized = cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis]
    
    df_cm = pd.DataFrame(cm_normalized,
                         index=[i for i in classes],
                         columns=[i for i in classes])

    fig = plt.figure(figsize=(12, 7))

    sn.heatmap(df_cm, annot=True)
    plt.show()
    fig.savefig('confusion_matrix.png')
    


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


def weights_balanced(labels, nclasses):

    """
    Make weights for balanced classes.
    
    :param labels: Labels
    :param nclasses: Number of classes
    
    :return weights: Weights
    
    """
    
    count = [0] * nclasses                                                       
    for item in labels: 
        count[item] += 1  
                     
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])                                 
    
    weight = [.0] * len(labels)
    for idx, item in enumerate(labels):
        weight[idx] = weight_per_class[item]
                                 
    return weight