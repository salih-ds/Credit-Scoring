## МЕТРИКИ КАЧЕСТВА МОДЕЛИ ##

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.metrics import plot_confusion_matrix, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score

# Построить график roc auc
def roc_auc_create(probs, y_valid):
    probs = probs[:,1]
    fpr, tpr, threshold = roc_curve(y_valid, probs)
    roc_auc = roc_auc_score(y_valid, probs)
    plt.figure()
    plt.plot([0, 1], label='Baseline', linestyle='--')
    plt.plot(fpr, tpr, label = 'Regression')
    plt.title('ROC AUC = %0.3f' % roc_auc)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc = 'lower right')
    plt.show()

# Построить график матрицы корреляции
def confusion_matrix_create(y_valid, y_pred):
    cm = confusion_matrix(y_valid, y_pred)
    cmd = ConfusionMatrixDisplay(cm, display_labels=['non_default','default'])
    cmd.plot()
    cmd.ax_.set(xlabel='Predicted', ylabel='True')

# Составить метрики качества классификатора
def model_metrics(y_valid,y_pred):
    print('precision_score:',precision_score(y_valid,y_pred))
    print('recall_score:',recall_score(y_valid,y_pred))
    print('f1_score:',f1_score(y_valid,y_pred))

