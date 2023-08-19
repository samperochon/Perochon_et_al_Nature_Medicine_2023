
import os
import sys
import json
from glob import glob
from copy import deepcopy
from tqdm import tqdm

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import  (confusion_matrix, roc_curve, fbeta_score, roc_auc_score, average_precision_score)
from metrics import f1score, average_precision, bestf1score, calc_auprg, create_prg_curve
import prg

# Import local packages
from const import *
from const_autism import *

sys.path.insert(0, '../../src')

def bootstrap_sensitivity_specificity(y_true, y_pred, optimal_threshold, verbose=False):
    
    def compute_sens_spec(y_true, y_pred, optimal_threshold):
    
        from sklearn.metrics import  (confusion_matrix, roc_curve)


        specificities_bar, sensitivities , thresholds = roc_curve(y_true, y_pred)

        specificities = 1 - specificities_bar

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred >= optimal_threshold).ravel()

        tpr =  tp / (tp+fn)
        tnr = tn / (tn+fp)
        return tpr, tnr


    n_pos  = np.sum(y_true == 1)
    n_neg  = np.sum(y_true == 0)

    idx_pos = np.argwhere(y_true == 1).flatten()
    idx_neg = np.argwhere(y_true == 0).flatten()

    specificities = []
    sensitivities = []

    for K in range(100):

        idx_resampled_pos = np.random.choice(idx_pos, size=n_pos)
        idx_resampled_neg = np.random.choice(idx_neg, size=n_neg)

        y_pred_resampled = np.concatenate([y_pred[idx_resampled_pos], y_pred[idx_resampled_neg]], axis=0)
        y_true_resampled = np.concatenate([y_true[idx_resampled_pos], y_true[idx_resampled_neg]], axis=0)

        sensitivity, specificity = compute_sens_spec(y_true=y_true_resampled, y_pred=y_pred_resampled, optimal_threshold=optimal_threshold)

        specificities.append(specificity); sensitivities.append(sensitivity)
    
    if verbose:
        print("Average Sensitivity: {:.3f} +/- {:.3f}".format(np.mean(sensitivities), np.std(sensitivities)))
        print("Average Specificity: {:.3f} +/- {:.3f}".format(np.mean(specificities), np.std(specificities)))
    
    return np.std(sensitivities), np.std(specificities)




def compute_performances(df, name=""):
    """
        df. is a dataframe with each rows representing one of the K exprriments. 
        
    """

    
    # 1) Compute Conclusiveness score 
    conslusiveness_score = []
    for i, row in df.iterrows():

        y_true = row['y_true'][0]
        y_pred = row['y_pred'][0]
        conslusiveness_score.append(list((y_pred >row['optimal_threshold']).astype(int)))
    conslusiveness_score = np.array(conslusiveness_score).mean(axis=0)
    
    y_pred = conslusiveness_score
    # 2) Compute Youden-optimal threshold
    
    specificities_bar, sensitivities , thresholds = roc_curve(y_true, y_pred)

    specificities = 1 - specificities_bar

    younden_indexes = sensitivities + specificities - 1

    max_youden, index_threshold = np.max(younden_indexes),  np.argmax(younden_indexes)
    
    optimal_threshold = thresholds[index_threshold]
    num_samples = row['num_samples']
    
    # 3) Compute performances
    y_pred = conslusiveness_score
    
    # Compute imbalance_ratio of our sample
    pi = y_true.mean()
    correction_factor = (pi*(1-REFERENCE_IMBALANCE_RATIO))/(REFERENCE_IMBALANCE_RATIO*(1-pi))
    
    # Compute first AUROC
    auroc = roc_auc_score(y_true, y_pred)

    # Compute the AUC-PR
    auc_pr = average_precision_score(y_true, y_pred)

    # Compute the AUC-PR Corrected
    auc_pr_corrected = average_precision(y_true, y_pred, pi0=REFERENCE_IMBALANCE_RATIO)

    # Compute the AUC-PR Gain
    auc_pr_g = prg.calc_auprg(prg.create_prg_curve(y_true, y_pred))

    # Compute the AUC-PR Gain corrected
    auc_pr_g_corrected = calc_auprg(create_prg_curve(y_true, y_pred, pi0=REFERENCE_IMBALANCE_RATIO))
    
    # Compute f1 and f2 scores
    f1 = fbeta_score(y_true, y_pred > optimal_threshold, beta=1) 
    f2 = fbeta_score(y_true, y_pred >= optimal_threshold, beta=2) 

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred >= optimal_threshold).ravel()

    # Compute corrected precision (ppv)
    ppv_corr = tp/(tp+correction_factor*fp)
    
    npv_corr = (correction_factor*tn)/(correction_factor*tn+fn)

    acc = (tp + tn) / (tp + tn + fp +  fn)
    mcc = (tp*tn - fp*fn) / np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    tpr =  tp / (tp+fn)
    tnr = tn / (tn+fp)
    ppv = tp / (tp+fp)
    npv = tn / (tn+fn)
    fnr = fn / (tp+fn)
    
    # Compute corrected F1 and F2
    f1_c = 2*(ppv_corr*tpr)/(ppv_corr+tpr)
    
    beta = 2
    f2_c = (1+beta**2)*(ppv_corr*tpr)/(beta**2 * ppv_corr + tpr)
    
    std_sens, std_spec = bootstrap_sensitivity_specificity(y_true, y_pred, optimal_threshold)

    performances_dict = {'name':name+'\n(N='+str(len(y_true))+')', 
                         'AUROC':round(auroc, 3),
                        'AUC-PR': round(auc_pr, 3),
                        'AUC-PR-Gain': round(auc_pr_g, 3),
                        'AUC-PR-Corrected': round(auc_pr_corrected, 3),
                        'AUC-PR-Gain-Corrected' :round(auc_pr_g_corrected, 3),
                        'F1 score (2 PPVxTPR/(PPV+TPR))': round(f1, 3),
                        'F1 score Corrected': round(f1_c, 3),
                        'F2': round(f2, 3),
                        'F2 Corrected': round(f2_c, 3),
                        'Accuracy' : round(acc, 3),
                        'Matthews correlation coefficient (MCC)': round(mcc, 3),
                        'Sensitivity, recall, hit rate, or true positive rate (TPR)': round(tpr, 3),
                        'Std - Sensitivity': round(std_sens, 3),  
                        'Specificity, selectivity or true negative rate (TNR)': round(tnr, 3),
                        'Std - Specificity': round(std_spec, 3), 
                        'Precision or positive predictive value (PPV)': round(ppv, 3),
                        'Corrected Precision or positive predictive value (PPV)': round(ppv_corr, 3),
                        'Corrected NPV': round(npv_corr, 3),
                        'Negative predictive value (NPV)': round(npv, 3),
                        'Miss rate or false negative rate (FNR)': round(fnr, 3),
                        'False discovery rate (FDR=1-PPV)': round(1-ppv, 3),
                        'False omission rate (FOR=1-NPV)': round(1-npv, 3),
                        'TP': tp,
                        'TN': tn,
                        'FP': fp,
                        'FN': fn, 
                        'optimal_threshold' :optimal_threshold,
                        'num_samples' : num_samples
                        }
    
    performances_df = pd.DataFrame(performances_dict, index=[name+'\n(N='+str(len(y_true))+')'])

    performances_df['TN'] = tn
    performances_df['TP'] = tp
    performances_df['FP'] = fp
    performances_df['FN'] = fn

    performances_df['TN_normalized'] = 100*tn/len(y_true)
    performances_df['TP_normalized'] = 100*tp/len(y_true)
    performances_df['FP_normalized'] =  100*fp/len(y_true)
    performances_df['FN_normalized'] =  100*fn/len(y_true)    
    performances_df['N'] = len(y_true) 
    performances_df['y_true'] = [y_true]
    performances_df['y_pred'] = [y_pred]
    performances_df['Hanley_CI'] = performances_df['AUROC'] .apply(lambda x: compute_SD(x, np.sum(y_true==1), np.sum(y_true==0)))
    
    return performances_df

def compute_performances_operating_points(y_true=None, y_pred=None):
    
    # Build a function that display the Table S2 showing all performances for diffferent threshold or operating points) 
    
    # Compute all the possible threshold
    specificities_bar, sensitivities , thresholds = roc_curve(y_true, y_pred)

    specificities = 1 - specificities_bar


    # Compute imbalance_ratio of our sample
    pi = y_true.mean()
    correction_factor = (pi*(1-REFERENCE_IMBALANCE_RATIO))/(REFERENCE_IMBALANCE_RATIO*(1-pi))

    ppv_list = []
    npv_list = []
    ppv_corr_list = []
    npv_corr_list = []

    for th in thresholds:

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred >= th).ravel()

        ppv = tp / (tp+fp)
        npv = tn / (tn+fn)

        # Compute corrected precision (ppv)
        ppv_corr = tp/(tp+correction_factor*fp)
        npv_corr = (correction_factor*tn)/(correction_factor*tn+fn)

        ppv_list.append(ppv)
        npv_list.append(npv)
        ppv_corr_list.append(ppv_corr)
        npv_corr_list.append(npv_corr)




    df_breakdown_results = pd.DataFrame({"Threshold index": np.arange(len(thresholds)), 
                                        "Threshold": thresholds, 
                                        "Sensitivity": sensitivities, 
                                        "Specificity": specificities, 
                                        "PPV": ppv_list, 
                                        "PPV_corr": ppv_corr_list, 
                                        "NPV": npv_list, 
                                        "NPV_corr": npv_corr_list, 
                                        })
    
    return df_breakdown_results



def compute_SD(AUC, N1, N2):
    """
            In the original paper of 1982, N1 is the number of "abnormal images", therefore here it is supposed to translate as the number of cases in the positive class.
    """
    Q1=AUC/(2-AUC)
    Q2 = 2*AUC*AUC/(1+AUC)
    return(np.sqrt((AUC*(1-AUC)+(N1-1)*(Q1-AUC*AUC) + (N2-1)*(Q2-AUC*AUC))/(N1*N2)))

def find_optimal_threshold_f(y_true, y_pred):

    pi = y_true.mean()
    correction_factor = (pi*(1-REFERENCE_IMBALANCE_RATIO))/(REFERENCE_IMBALANCE_RATIO*(1-pi))


    # Compute all the possible threshold
    _, _ , thresholds = roc_curve(y_true, y_pred)


    f1_c = []
    f2_c = []
    for th in thresholds[1:]:
        # For each threshold, compute the confusion matrix elements to be able to compute recall and precision 
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred >= th).ravel()    

        # Compute corrected precision (ppv)
        precision_corr = tp/(tp+correction_factor*fp)

        # Compute recall 
        recall =  tp / (tp+fn)

        f1_c.append(2*(precision_corr*recall)/(precision_corr+recall))

        beta = 2

        f2_c.append((1+beta**2)*(precision_corr*recall)/(beta**2 * precision_corr + recall))

    f2 = [fbeta_score(y_true, y_pred >= th, beta=2) for th in thresholds]
    best_f2, threshold_optimal_f2 = np.max(f2), thresholds[np.argmax(f2)-1]

    f1 = [fbeta_score(y_true, y_pred >= th, beta=1) for th in thresholds]
    best_f1, threshold_optimal_f1 = np.max(f2), thresholds[np.argmax(f1)-1]
    
    index_optimal_f1 = np.argmax(f1)-1
    index_optimal_f2 = np.argmax(f2)-1

    # Also return the f1c and f2c for the optimal f2 measure
    return best_f1, best_f2, f1_c[np.argmax(f2)-1], f2_c[np.argmax(f2)-1], index_optimal_f1, index_optimal_f2, threshold_optimal_f2

def find_optimal_threshold_youden(y_true, y_pred):
    
    # Compute the performances using Younden Index
    
    # Compute all the possible threshold
    specificities_bar, sensitivities , thresholds = roc_curve(y_true, y_pred)

    specificities = 1 - specificities_bar

    younden_indexes = sensitivities + specificities - 1

    max_youden, index_threshold = np.max(younden_indexes),  np.argmax(younden_indexes)
    
    optimal_threshold = thresholds[index_threshold]
    
    
    return max_youden, index_threshold, optimal_threshold

def corrected_f1_sklearn(clf, X, y):
    
    y_pred = clf.predict_proba(X)[:,1]
    
    from metrics import f1score, average_precision, bestf1score, calc_auprg, create_prg_curve
    
     # Compute the F1 score
    f1, optimal_threshold = bestf1score(y, y_pred, pi0=None)

    # Compute the corrected F1 score
    f1_corrected, _ = bestf1score(y, y_pred, pi0=REFERENCE_IMBALANCE_RATIO)

    return f1_corrected


def corrected_f1_xgboost(preds, dtrain):
    res = _corrected_f1_sklearn(preds, dtrain.get_label())
    print('yo')
    return 'f1_corrected', 1-res


def select(df, feat, value):
    return df[df[feat]==value]

def _corrected_f1_sklearn(y_pred, y):
        
    from metrics import f1score, average_precision, bestf1score, calc_auprg, create_prg_curve
    
     # Compute the F1 score
    f1, optimal_threshold = bestf1score(y, y_pred, pi0=None)

    # Compute the corrected F1 score
    f1_corrected, _ = bestf1score(y, y_pred, pi0=REFERENCE_IMBALANCE_RATIO)

    return f1_corrected


def fi(x=12, y=12):
    return plt.figure(figsize=(x, y))

def create_dataset(name, num_samples=10, ratio_of_missing_values=.5, imbalance_ratio=.5, provide_labels=True, only_one_missing=True, verbose=True):
    from sklearn import datasets
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils import shuffle 
    
    n = num_samples + 2000  # we generate num_samples for testing and 2k as ground truth 

    ################################
    # Generate the positive examples
    ################################
    if name=='moons':
        data = datasets.make_moons(n_samples=int(2*imbalance_ratio*n), noise=.05)
    elif name=='circles':
        data = datasets.make_circles(n_samples=int(2*imbalance_ratio*n), factor=.5, noise=.05)
    elif name=='blobs':
        data = datasets.make_blobs(n_samples=n, random_state=8)
    else:
        raise ValueError("Please use 'moons', 'circles', or 'blobs' datasets.") 
        
    X_all, labels = data  # keep the 2D samples 

    # normalize dataset for easier parameter selection
    X_all = StandardScaler().fit_transform(X_all)

    # Select the positive examples
    X_all = X_all[np.argwhere(labels==1).squeeze()]

    # Separate ground truth and training data
    X_pos = X_all[:int(num_samples*imbalance_ratio),:] 
    Xgt_pos = X_all[int(num_samples*imbalance_ratio):,:]
    labels_pos, labelsgt_pos = 1*np.ones((X_pos.shape[0], 1)), 1*np.ones((Xgt_pos.shape[0], 1))

    ################################
    # Generate the negative examples
    ################################
    if name=='moons':
        data = datasets.make_moons(n_samples=int(2*(1-imbalance_ratio)*n), noise=.05)
    elif name=='circles':
        data = datasets.make_circles(n_samples=int(2*(1-imbalance_ratio)*n), factor=.5, noise=.05)
    else:
        raise ValueError("Please use 'moons' or 'circles' datasets.") 
    
    X_all, labels = data  # keep the 2D samples 

    # normalize dataset for easier parameter selection
    X_all = StandardScaler().fit_transform(X_all)

    # Select the negative examples
    X_all = X_all[np.argwhere(labels==0).squeeze()]

    # Separate ground truth and training data
    X_neg = X_all[:int(num_samples*(1-imbalance_ratio)),:] 
    Xgt_neg = X_all[int(num_samples*(1-imbalance_ratio)):,:]
    labels_neg, labelsgt_neg = np.zeros((X_neg.shape[0], 1)), np.zeros((Xgt_neg.shape[0], 1))

    # Combine the positive and negative samples
    X, labels = np.concatenate([X_neg, X_pos], axis=0), np.concatenate([labels_neg, labels_pos], axis=0)
    Xgt, labelsgt = np.concatenate([Xgt_neg, Xgt_pos], axis=0), np.concatenate([labelsgt_neg, labelsgt_pos], axis=0)

    # Shuffle the data 
    X, labels = shuffle(X, labels, random_state=0)
    Xgt, labelsgt = shuffle(Xgt, labelsgt, random_state=0)
    
    
    if only_one_missing:
        # Simulate missing samples
        for i in range(X.shape[0]):  # randomtly remove features
            if np.random.random() < ratio_of_missing_values:
                if np.random.random() < .5:  # remove samples from x or y with 
                    # equal probability
                    X[i,0] = np.nan
                else:
                    X[i,1] = np.nan
    else:
        # Simulate missing samples
        for i in range(X.shape[0]):  # randomtly remove features
            if np.random.random() < ratio_of_missing_values:
                X[i,0] = np.nan

            if np.random.random() < ratio_of_missing_values:
                X[i,1] = np.nan

    if verbose:
        import seaborn as sns
        color = plt.get_cmap('tab20')(np.arange(0,2)); cmap = sns.color_palette(color)
        colors, colorsgt = [cmap[0] if l==1 else cmap[1] for l in labels], [cmap[0] if l==1 else cmap[1] for l in labelsgt]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
        ax1.scatter(Xgt[:,0], Xgt[:,1], c=colorsgt);ax1.axis('off');ax1.set_title("Ground Truth\n{}% imbalance ratio\n".format(int(imbalance_ratio*100)), weight='bold')
        ax2.scatter(X[:,0], X[:,1], c=colors);ax2.axis('off');ax2.set_title("Created samples\n{}% imbalance ratio\n{} % missing data".format( int(imbalance_ratio*100),int(ratio_of_missing_values*100)), weight='bold')

    if provide_labels: 
        return X, Xgt, labels.squeeze(), labelsgt.squeeze()
    else:
        return X, Xgt


def compare_imputation_methods(dataset='None', kernel_bandwidth=.2, num_samples=100, ratio_of_missing_values=.7, imbalance_ratio=.5, resolution=20, methods=None):
    
    h = kernel_bandwidth
    
    # (1) Create toy and ground truth data
    X, Xgt, _, _ = create_dataset(name=dataset, 
                                      num_samples=num_samples, 
                                      ratio_of_missing_values=ratio_of_missing_values, 
                                      imbalance_ratio=imbalance_ratio,
                                      provide_labels=True, 
                                      verbose=True)
    
    print('{} samples created'.format(X.shape[0]))
    plt.figure(figsize=[10,10]); plt.subplot(3,3,1); plt.scatter(X[:,0],X[:,1]); 
    plt.title('Toy data'); plt.xlim(-3, 3); plt.ylim(2.5, -2.5); 
    plt.xticks(()); plt.yticks(()); plt.axis('equal'); plt.axis('off')

    # Ground truth
    from sklearn.neighbors import KernelDensity
    kde = KernelDensity(kernel='gaussian', bandwidth=h).fit(Xgt)
    xygrid = np.meshgrid(np.linspace(-3, 3,resolution),np.linspace(-3, 3,resolution))
    H,W = xygrid[0].shape
    hat_f = np.zeros_like(xygrid[0])  # init. the pdf estimation
    for i in range(H):
        for j in range(W):
            x = xygrid[0][i,j]
            y = xygrid[1][i,j]
            hat_f[i,j] = np.exp(kde.score_samples([[x,y]]))
    plt.subplot(3,3,2); plt.imshow(hat_f); plt.axis('off'); plt.title('Ground truth')
    hat_fgt = hat_f
    hat_fgt /= hat_fgt.sum()
            
    if methods is None:
        methods = ['naive', 'mean', 'median', 'knn', 'mice', 'our', 'multi_distributions']
    for i,method in enumerate(methods):
        hat_f = estimate_pdf_TODO(X=X, method=method, resolution=resolution, bandwidth=h)  
        hat_f /= hat_f.sum()
        plt.subplot(3,3,i+3); plt.imshow(hat_f); plt.axis('off');
        l2diff = np.mean( (hat_fgt-hat_f)**2 ); 
        plt.title('{} error {:2.5f}'.format(method,1e6*l2diff))
    
    return


def estimate_pdf_TODO(X=None, method='multi_distributions', resolution=20, bandwidth=None):
    
    xygrid = np.meshgrid(np.linspace(-3, 3,resolution),np.linspace(-3, 3,resolution))
    H,W = xygrid[0].shape
    hat_f = np.zeros_like(xygrid[0])  # init. the pdf estimation
    h = bandwidth


    if method=='our':
        # See documentation
        from model.bayesian.stats import kernel_based_pdf_estimation
    
        for i in range(H):
            for j in range(W):
                x = xygrid[0][i,j]
                y = xygrid[1][i,j]
                hat_f[i,j] = kernel_based_pdf_estimation(X=X, x=[x,y], h=h)

    elif method=='naive':
        # Ignore missing values
        from model.bayesian.stats import kernel_based_pdf_estimation
        imp_X = X[~np.isnan(X[:,0]),:]
        imp_X = imp_X[~np.isnan(imp_X[:,1]),:]        
                
    elif method=='mean':
        from sklearn.impute import SimpleImputer
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp_X = imp.fit_transform(X)
    
    elif method=='median':
        from sklearn.impute import SimpleImputer
        imp = SimpleImputer(missing_values=np.nan, strategy='median')
        imp_X = imp.fit_transform(X)
    
    elif method=='knn':
        from sklearn.impute import KNNImputer
        knn_imputer = KNNImputer()
        imp_X = knn_imputer.fit_transform(X)
        
    elif method=='mice':
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        imp = IterativeImputer(n_nearest_features=None, imputation_order='ascending')
        imp_X = imp.fit_transform(X)

    elif method=='multi_distributions':


        #----------------------------------------------------------------------------------
        #  Estimation of f(X_1,X_2|Z_1=1, Z_2=1), f(X_2|Z_1=0,Z_2=1) and f(X_1|Z_1=1,Z_2=0)
        #----------------------------------------------------------------------------------

        from model.bayesian.stats import kernel_based_pdf_estimation_side_spaces

        hat_f_0 = np.zeros_like(xygrid[0])  # init. the pdf estimation
        hat_f_1 = np.zeros_like(xygrid[0])  # init. the pdf estimation
        hat_f_2 = np.zeros_like(xygrid[0])  # init. the pdf estimation

        for i in range(H):
            for j in range(W):
                x = xygrid[0][i,j]
                y = xygrid[1][i,j]
                # Computing contribution on coordinates i, j of hat_f, and coordinate i of hat_f_1 and coordinate j of hat_f_2
                hat_f[i,j], hat_f_0[i,j], hat_f_1[i,j], hat_f_2[i,j] =  kernel_based_pdf_estimation_side_spaces(X=X, x=[x, y], h=h)
                
        # Average the contribution of all i's and j's coordinate
        hat_f_0 = np.mean(hat_f_0)

        # Average the contribution of all j's coordinate on this horizontal line
        hat_f_1 = np.mean(hat_f_1, axis=0)
        
        # Average the contribution of all i's coordinate to form the vertical line
        hat_f_2 = np.mean(hat_f_2, axis=1) 

        # Normalization of the distributions
        hat_f /= (hat_f.sum()+EPSILON);hat_f_1 /= (hat_f_1.sum()+EPSILON);hat_f_2 /= (hat_f_2.sum()+EPSILON)

        
    
    if method in ['mice', 'knn', 'median', 'mean', 'naive']:
        from model.bayesian.stats import kernel_based_pdf_estimation
        for i in range(H):
            for j in range(W):
                x = xygrid[0][i,j]
                y = xygrid[1][i,j]
                hat_f[i,j] = kernel_based_pdf_estimation(imp_X, x=[x,y],h=h)


    return hat_f



def label_bar(rects,ax):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()*(2/4), .5*height,
                 "{:.2f}".format(height),
                 fontsize = 11,
                ha='center', va='bottom')

def check_experiment_already_done(df, verbose=False,return_df=False, **kwargs):
    
    narrowed_df=deepcopy(df)
    if verbose:
        print(len(narrowed_df)) 
    
    for key, value in kwargs.items():
        
        if key=='ratio_missing_per_class':
            if value is None:
                narrowed_df = narrowed_df[narrowed_df['ratio_missing_per_class_0'].isnull()]
                narrowed_df = narrowed_df[narrowed_df['ratio_missing_per_class_1'].isnull()]
            else:
                narrowed_df = narrowed_df[narrowed_df['ratio_missing_per_class_0']==value[0]]
                narrowed_df = narrowed_df[narrowed_df['ratio_missing_per_class_1']==value[1]]    
        elif key == 'use_missing_indicator_variables':
            if value is None:
                narrowed_df = narrowed_df[narrowed_df['use_missing_indicator_variables'].isnull()]
            else:
                narrowed_df = narrowed_df[narrowed_df['use_missing_indicator_variables']==value]
                
                
        elif key == 'ratio_of_missing_values':
            if value is None:
                narrowed_df = narrowed_df[narrowed_df['ratio_of_missing_values'].isnull()]
            else:
                narrowed_df = narrowed_df[narrowed_df['ratio_of_missing_values']==value]
                
                
        elif key == 'ratio_missing_per_class_0':
            if value is None:
                narrowed_df = narrowed_df[narrowed_df['ratio_missing_per_class_0'].isnull()]
            else:
                narrowed_df = narrowed_df[narrowed_df['ratio_missing_per_class_0']==value]
                
        elif key == 'ratio_missing_per_class_1':
            if value is None:
                narrowed_df = narrowed_df[narrowed_df['ratio_missing_per_class_1'].isnull()]
            else:
                narrowed_df = narrowed_df[narrowed_df['ratio_missing_per_class_1']==value]   
                
                
        else:
        
            narrowed_df = narrowed_df[narrowed_df[key]==value]
            
        print(len(narrowed_df), key, value) if verbose else None
        
    if not return_df:
        
        return len(narrowed_df) > 0
    
    else:
        
        return narrowed_df
        


def create_df(folder_names=EXPERIMENT_FOLDER_NAME):

    if not isinstance(folder_names, list):
        folder_names = list(folder_names)

    
    df = pd.DataFrame(columns = ['dataset_name','experiment_number', 'approach', 'missing_data_handling','imputation_method', 'use_missing_indicator_variables', 
                                'num_samples', 'imbalance_ratio', 'missingness_pattern', 'missingness_mechanism', 
                                'ratio_of_missing_values', 'missing_X1', 'missing_X2', 'missing_first_quarter','ratio_missing_per_class_0', 'ratio_missing_per_class_1','auc',
                                'Accuracy', 'F1', 'MCC', 'Sensitivity', 'Specificity', 'Precision', 'PPV', 'NPV', 'FNR', 'FDR', 'FOR', 
                                'resolution', 'bandwidth', 'estimation_time_0', 'estimation_time_1'])

    
    experiments_paths = []
    for folder_name in folder_names:
        experiments_paths.extend(glob(os.path.join(DATA_DIR, folder_name, "*", '*')))


    for experiment_path in experiments_paths:
        
        try:

            exp_path = os.path.join(experiment_path, 'experiment_log.json')
            dataset_path = os.path.join(experiment_path, 'dataset_log.json')

            dist_None_path = os.path.join(experiment_path, 'distributions_None_log.json')
            dist_1_path = os.path.join(experiment_path, 'distributions_1_log.json')
            dist_0_path = os.path.join(experiment_path, 'distributions_0_log.json')

            dist_0_data, dist_0_data, dist_None_data = None, None, None

            if os.path.isfile(exp_path):

                with open(exp_path) as experiment_json:

                    # Load experiment data
                    experiment_data = json.load(experiment_json)
            else:
                continue

            if os.path.isfile(dataset_path):
                with open(dataset_path) as data_json:

                    # Load experiment data
                    dataset_data = json.load(data_json)
            else:
                continue

            if os.path.isfile(dist_None_path):

                with open(dist_None_path) as dist_json:

                    # Load experiment data
                    dist_None_data = json.load(dist_json)

            if os.path.isfile(dist_1_path):

                with open(dist_1_path) as dist_json:

                    # Load experiment data
                    dist_1_data = json.load(dist_json)

            if os.path.isfile(dist_0_path):

                with open(dist_0_path) as dist_json:

                    # Load experiment data
                    dist_0_data = json.load(dist_json)

            # append rows to an empty DataFrame
            df = df.append({'dataset_name' : experiment_data['dataset_name'], 
                            'experiment_number' : experiment_data['experiment_number'],  
                            'approach' : experiment_data['approach'],  
                            'missing_data_handling' : dataset_data['missing_data_handling'],  
                            'imputation_method' : dataset_data['imputation_method'],  
                            'use_missing_indicator_variables': experiment_data['use_missing_indicator_variables'] if 'use_missing_indicator_variables' in experiment_data.keys() else dataset_data['use_missing_indicator_variables'],   # TODO 
                            'num_samples' : dataset_data['num_samples'],  
                            'imbalance_ratio' : dataset_data['imbalance_ratio'],  
                            'missingness_pattern' : int(dataset_data['missingness_pattern']),  
                            'missingness_mechanism' : dataset_data['missingness_parameters']['missingness_mechanism'],  
                            'ratio_of_missing_values' : dataset_data['missingness_parameters']['ratio_of_missing_values'],  
                            'missing_X1' : dataset_data['missingness_parameters']['missing_X1'],  
                            'missing_X2' : dataset_data['missingness_parameters']['missing_X2'],  
                            'missing_first_quarter' : dataset_data['missingness_parameters']['missing_first_quarter'],  
                            'ratio_missing_per_class_0' : dataset_data['missingness_parameters']['ratio_missing_per_class'][0] if dataset_data['missingness_parameters']['ratio_missing_per_class'] is not None else None,
                            'ratio_missing_per_class_1' : dataset_data['missingness_parameters']['ratio_missing_per_class'][1] if dataset_data['missingness_parameters']['ratio_missing_per_class'] is not None else None,
                            'resolution' : experiment_data['resolution'],
                            'bandwidth' : experiment_data['bandwidth'],
                            'auc' : experiment_data['performances_df']['Area Under the Curve (AUC)'][0] if 'Area Under the Curve (AUC)' in experiment_data['performances_df'].keys() else np.nan,
                            'Accuracy' : experiment_data['performances_df']['Accuracy'][0],  
                            'F1' : experiment_data['performances_df']['F1 score (2 PPVxTPR/(PPV+TPR))'][0],  
                            'MCC' : experiment_data['performances_df']['Matthews correlation coefficient (MCC)'][0],  
                            'Sensitivity' : experiment_data['performances_df']['Sensitivity, recall, hit rate, or true positive rate (TPR)'][0],  
                            'Specificity' : experiment_data['performances_df']['Specificity, selectivity or true negative rate (TNR)'][0],  
                            'Precision' : experiment_data['performances_df']['Precision or positive predictive value (PPV)'][0],  
                            'PPV' : experiment_data['performances_df']['Precision or positive predictive value (PPV)'][0],  
                            'NPV' : experiment_data['performances_df']['Negative predictive value (NPV)'][0],  
                            'FNR' : experiment_data['performances_df']['Miss rate or false negative rate (FNR)'][0],  
                            'FDR' : experiment_data['performances_df']['False discovery rate (FDR=1-PPV)'][0],  
                            'FOR' : experiment_data['performances_df']['False omission rate (FOR=1-NPV)'][0],  
                            }, 
                            ignore_index = True)
        except:
            pass

    df['ratio_missing_per_class_0'] = df['ratio_missing_per_class_0'].astype(float).round(2)
    df['ratio_missing_per_class_0'] = df['ratio_missing_per_class_0'].astype(float).round(2)
        
    df['ratio_missing_per_class_0'] = df['ratio_missing_per_class_0'].astype(float).round(2)
    df['ratio_missing_per_class_1'] = df['ratio_missing_per_class_1'].astype(float).round(2)
    df['ratio_of_missing_values'] = df['ratio_of_missing_values'].astype(float).round(2)
    df.loc[df['use_missing_indicator_variables'].isna(), 'use_missing_indicator_variables'] = False
    df.loc[df['use_missing_indicator_variables'].isnull(), 'use_missing_indicator_variables'] = False

    df.drop_duplicates(inplace=True)
    df = df.astype({"missingness_pattern": int, "experiment_number": int})
    
    return df

folder_names = ['autism_all']

  
      
def create_autism_df(folder_names):  
    
    if not isinstance(folder_names, list):
        folder_names = list(folder_names)
    df = pd.DataFrame(columns = ['dataset_name','experiment_number', 'experiment_name', 'approach', 'y_true', 'y_pred', 'missing_data_handling','imputation_method', 'features_name', 'n_features', 'use_missing_indicator_variables', 'scale_data', 'sampling_method','scenario','num_samples', 
    'max_depth', 'reg_lambda', 'gamma', 'learning_rate', 'n_estimators', 'imbalance_ratio', 'optimal_threshold', 'ratio_of_missing_values','ratio_missing_per_class_0', 'ratio_missing_per_class_1', 'resolution', 'bandwidth', 'estimation_time', 'num_cv', 'AUROC','AUC-PR', 'AUC-PR-Gain', 'AUC-PR-Corrected', 'AUC-PR-Gain-Corrected', 'F1', 'F1 score Corrected', 'F2', 'F2 score Corrected', 'Accuracy', 'MCC', 'Sensitivity', 'Specificity', 'Precision', 'PPV', 'PPV-Corr', 'NPV', 'FNR', 'FDR', 'FOR', 'TP', 'TN', 'FP', 'FN', 'tree_usage'])


    experiments_paths = []
    for folder_name in folder_names:
        experiments_paths.extend(glob(os.path.join(DATA_DIR, folder_name, "*", '*')))


    for experiment_path in tqdm(experiments_paths):
        
        try:
            
            exp_path = os.path.join(experiment_path, 'experiment_log.json')
            dataset_path = os.path.join(experiment_path, 'dataset_log.json')

            dist_None_path = os.path.join(experiment_path, 'distributions_None_log.json')
            dist_1_path = os.path.join(experiment_path, 'distributions_1_log.json')
            dist_0_path = os.path.join(experiment_path, 'distributions_0_log.json')

            dist_0_data, dist_0_data, dist_None_data = None, None, None

            if os.path.isfile(exp_path):

                with open(exp_path) as experiment_json:

                    # Load experiment data
                    experiment_data = json.load(experiment_json)
            else:
                continue

            if os.path.isfile(dataset_path):
                with open(dataset_path) as data_json:

                    # Load experiment data
                    dataset_data = json.load(data_json)
            else:
                continue

            if os.path.isfile(dist_None_path):

                with open(dist_None_path) as dist_json:

                    # Load experiment data
                    dist_None_data = json.load(dist_json)

            if os.path.isfile(dist_1_path):

                with open(dist_1_path) as dist_json:

                    # Load experiment data
                    dist_1_data = json.load(dist_json)

            if os.path.isfile(dist_0_path):

                with open(dist_0_path) as dist_json:

                    # Load experiment data
                    dist_0_data = json.load(dist_json)
        except:
            continue

        # append rows to an empty DataFrame
        df = df.append({'dataset_name' : experiment_path.split('/')[-3], 
                        'experiment_number' : experiment_data['experiment_number'],  
                        'experiment_name' : experiment_data['experiment_name'],  
                        'approach' : experiment_data['approach'],  
                        'y_true' : [pd.DataFrame(json.loads(experiment_data['predictions_df']))['y_true'].to_numpy()],  
                        'y_pred' : [pd.DataFrame(json.loads(experiment_data['predictions_df']))['y_pred'].to_numpy()],  
                        'missing_data_handling' : dataset_data['missing_data_handling'],  
                        'imputation_method' : dataset_data['imputation_method'],  
                        'features_name': str(dataset_data['_features_name']),
                        'n_features': len(dataset_data['_features_name']),
                        'use_missing_indicator_variables': dataset_data['use_missing_indicator_variables'],
                        'scale_data': dataset_data['scale_data'], 
                        'sampling_method': dataset_data['sampling_method'], 
                        'scenario':  dataset_data['scenario'], 
                        'num_samples' : dataset_data['num_samples'],  
                        'max_depth' : experiment_data['model_hyperparameters']['max_depth'],
                        'reg_lambda' : experiment_data['model_hyperparameters']['reg_lambda'],
                        'gamma' : experiment_data['model_hyperparameters']['gamma'],
                        'learning_rate': experiment_data['model_hyperparameters']['learning_rate'],
                        'n_estimators': experiment_data['model_hyperparameters']['n_estimators'] if 'n_estimators'
                         in experiment_data['model_hyperparameters'].keys() else 100,
                        'imbalance_ratio' : dataset_data['imbalance_ratio'],  
                        'ratio_of_missing_values' : dataset_data['ratio_of_missing_values'],  
                        'ratio_missing_per_class_0' : dataset_data['ratio_missing_per_class'][0],
                        'ratio_missing_per_class_1' : dataset_data['ratio_missing_per_class'][1],
                        'resolution' : experiment_data['resolution'],
                        'bandwidth' : experiment_data['bandwidth'],
                        'optimal_threshold': experiment_data['optimal_threshold'],
                        'estimation_time': experiment_data['estimation_time'],
                        'num_cv': experiment_data['num_cv'],
                        'AUROC' : experiment_data['performances_df']['Area Under the Curve (AUC)'][0] if 'Area Under the Curve (AUC)' in experiment_data['performances_df'].keys() else experiment_data['performances_df']['AUROC'][0] if 'AUROC' in experiment_data['performances_df'].keys() else np.nan,
                        'AUC-PR' : experiment_data['performances_df']['AUC-PR'][0],  
                        'AUC-PR-Gain' : experiment_data['performances_df']['AUC-PR-Gain'][0],  
                        'AUC-PR-Corrected' : experiment_data['performances_df']['AUC-PR-Corrected'][0],  
                        'AUC-PR-Gain-Corrected' : experiment_data['performances_df']['AUC-PR-Gain-Corrected'][0],  
                        'F1' : experiment_data['performances_df']['F1 score (2 PPVxTPR/(PPV+TPR))'][0],  
                        'F1 score Corrected' : experiment_data['performances_df']['F1 score Corrected'][0],   
                        'F2' : experiment_data['performances_df']['F2'][0],  
                        'F2 score Corrected' : experiment_data['performances_df']['F2 Corrected'][0],   
                        'Accuracy': experiment_data['performances_df']['Accuracy'][0],   
                        'MCC' : experiment_data['performances_df']['Matthews correlation coefficient (MCC)'][0],  
                        'Sensitivity' : experiment_data['performances_df']['Sensitivity, recall, hit rate, or true positive rate (TPR)'][0],  
                        'Specificity' : experiment_data['performances_df']['Specificity, selectivity or true negative rate (TNR)'][0],  
                        'Precision' : experiment_data['performances_df']['Precision or positive predictive value (PPV)'][0],  
                        'PPV-Corr': experiment_data['performances_df']['Corrected Precision or positive predictive value (PPV)'][0],
                        'PPV' : experiment_data['performances_df']['Precision or positive predictive value (PPV)'][0],  
                        'NPV' : experiment_data['performances_df']['Negative predictive value (NPV)'][0],  
                        'FNR' : experiment_data['performances_df']['Miss rate or false negative rate (FNR)'][0],  
                        'FDR' : experiment_data['performances_df']['False discovery rate (FDR=1-PPV)'][0],  
                        'FOR' : experiment_data['performances_df']['False omission rate (FOR=1-NPV)'][0],  
                        'TP' :  experiment_data['performances_df']['TP'][0], 
                        'TN' :  experiment_data['performances_df']['TN'][0], 
                        'FP' :  experiment_data['performances_df']['FP'][0], 
                        'FN' :  experiment_data['performances_df']['FN'][0], 
                        'tree_usage': experiment_data['tree_usage'] if 'tree_usage' in experiment_data.keys() else np.nan}, 
                        ignore_index = True)

    #df['ratio_missing_per_class_0'] = df['ratio_missing_per_class_0'].astype(float).round(2)
    #df['ratio_missing_per_class_1'] = df['ratio_missing_per_class_1'].astype(float).round(2)
    #df['ratio_of_missing_values'] = df['ratio_of_missing_values'].astype(float).round(2)
    df.loc[df['use_missing_indicator_variables'].isna(), 'use_missing_indicator_variables'] = False
    df.loc[df['use_missing_indicator_variables'].isnull(), 'use_missing_indicator_variables'] = False

    #df.drop_duplicates(inplace=True)
    df = df.astype({"experiment_number": int})
    df.loc[(df['missing_data_handling']=='encoding'), 'imputation_method'] = 'constant'
    
    return df
def repr(object_, indent=0):

    import seaborn as sns 
    import numpy as np
    
    if indent==0:
        
        print("{0:10}{1:30}\t {2:40}\t {3:150}".format("","Attribute Name", "type", "Value or first element"))
        print("{0:10}-----------------------------------------------------------------------------------------------------------------------------------------------------------------\n".format(""))
    
    for _ in range(indent):
        print("\t")
    
    if not isinstance(object_, dict):
    
        dict_ = object_.__dict__
    else:
        dict_ = object_
    
    for k, o in dict_.items():
        if type(o) == dict:
            print("{0:10}{1:30}\t {2:40}".format(indent*"\t" if indent > 0 else "", k, _print_correct_type(o)))
            repr(o, indent=indent+1)
        else:
            print("{0:10}{1:30}\t {2:40}\t {3:150}".format(indent*"\t" if indent > 0 else "", k, _print_correct_type(o), _print_correct_sample(o, indent=indent)))
    
    print("\n")
    return 

def _print_correct_sample(o, indent=0):
    """
        This helper function is associated with the show method, used to print properly classes object.
        This one output a string of the element o, taking the type into account.

    """


    
    if o is None:
        return "None"    
    
    elif isinstance(o, (int, float, np.float32, np.float64)):
        return str(o)
    
    elif isinstance(o, str):
        return o.replace('\n', '-') if len(o) < 80 else o.replace('\n', '-')[:80]+'...'
    
    elif isinstance(o, (list, tuple)) :#and not type(o) == sns.palettes._ColorPalette:
        return "{} len: {}".format(str(o[0]), len(o))
    
    elif isinstance(o, np.ndarray) :#and not type(o) == sns.palettes._ColorPalette:
        return "{} shape: {}".format(str(o[0]), str(o.shape))
    
    elif type(o) == dict : 
        return repr(o, indent=indent+1)

    elif type(o) == pd.core.frame.DataFrame:
        return 'dataframe'
    
    else:
        
        return str(o)
    
def _print_correct_type(o):
    """
        This helper function is associated with the show method, used to print properly classes object.
        This one output a string of the type of the element o.

    """
    if o is None:
        return "None"    
    
    elif isinstance(o, int):
        return "int"
    
    elif isinstance(o, float):
        return "float"
    
    elif isinstance(o, np.float32):
        return "np.float32"
    
    elif isinstance(o, np.float64):
        return "np.float64"    
    
    elif isinstance(o, list):
        return "list"
    
    elif isinstance(o, str):
        return "str"
    
    elif isinstance(o, tuple):
        return "tuple"
    
    elif isinstance(o, np.ndarray):
        return "np.ndarray"
    
    elif type(o) == dict : 
        return "dict"
    
    else:
        
        return str(type(o))
        
