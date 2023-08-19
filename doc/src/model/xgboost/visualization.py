import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import roc_curve, auc

def plot_roc_curves_xgboost(predictions_df, ax = None):

    if ax is None:
        fig, ax =  plt.subplots(1, 1, figsize=(5, 5))

    fpr, tpr, _ = roc_curve(predictions_df['y_true'], predictions_df['y_pred']); roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, '-', color='darkorange', lw=1.5, label='ROC curve (area = %0.2f)' % roc_auc,)
    ax.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
    ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05]); ax.grid()
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    return ax