import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.metrics import confusion_matrix


def confusion_matrix_plot(y_true, y_predicted, classes):
    
    cf_matrix = confusion_matrix(y_true, y_predicted)
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) * 3, 
                         index = [i for i in classes], 
                         columns = [i for i in classes])
    
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.show()