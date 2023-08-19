"""
Machine learning algorithms, models, training and evaluating scripts. 
-----
"""

from operator import le
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
import matplotlib.pyplot as plt
import os
import plotly.express as px


class my_model_categorical(BaseEstimator, ClassifierMixin):
    """Wrap sklearn models into my own estimator, this allows my to handle some upsampling and custom operations in a more compact fashion. 
    """
    def __init__(self, method, name='ClassifierName'):
        self.name = name
        if method == 'cnb':
            from sklearn.naive_bayes import CategoricalNB
            self.model = CategoricalNB()

    def fit(self, X, y, balanced=True, batch_size=16, epochs=50, name = 'log_id'):
        """
        Train the model using X (n_samples, n_features) matrix, for the two classes
        problem. To work with imbalanced probels, balanced=True can be set, which 
        oversamples the minority class (assumed to be the positive class y=1), to 
        much the number of samples in the majority class (assumed to be the negative
        class, i.e., y=0). You can set "model_id" parameters to set where tensorboard
        logs are saved, which is useful to visualize training. 
        """
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        """
        Oversample the minority class
        """
        if balanced:
            X,y = upsample_minority(X,y,method='smote')

        """
        Fit
        """
        self.model.fit(X, y)        
        return self
    
    def predict_proba(self, X):
        # Predict the classification score (associated with the class==1)
        y_score = self.model.predict_proba(X)[:,1]  
        return y_score
    
    def predict(self, X, th=.5):
        # Check is fit had been called
        check_is_fitted(self)
        # Input validation
        X = check_array(X)
        # Normalize the sample to match the pre-processing done at training        
        y_score = self.predict_proba(X)  # predict 
        y_pred = [1 if yy>=th else 0 for yy in y_score]
        return y_pred


class my_model(BaseEstimator, ClassifierMixin):
    """Wrap sklearn models into my own estimator, this allows my to handle some upsampling and custom operations in a more compact fashion. 
    """
    def __init__(self, method, name='ClassifierName', **kwargs):
        
        self.name = name
        self.force_all_finite = True
        self.sampling_method = 'smote'

        # Depending on the method selected, init the proper sklearn model
        if method == "knn":  # use k-nearest neighbor
            from sklearn.neighbors import KNeighborsClassifier
            self.model =  KNeighborsClassifier(n_neighbors=40, weights='uniform', **kwargs)

        if method == "lg":  # use logistic regresion
            from sklearn.linear_model import LogisticRegression
            self.model = LogisticRegression(**kwargs)

        if method == 'nb': 
            from sklearn.naive_bayes import GaussianNB
            self.model = GaussianNB(**kwargs)
            
        if method == 'xgboost':
            from xgboost import XGBClassifier
            self.model = XGBClassifier(use_label_encoder=False,
                                      learning_rate=0.01,
                                      verbosity=1,
                                      objective='binary:logistic',
                                      eval_metric='auc',
                                      booster='gbtree',
                                      tree_method='exact',
                                      subsample=1,
                                      colsample_bylevel=.8,
                                      alpha=0, 
                                      **kwargs)
            self.force_all_finite = False
            self.sampling_method = 'vanilla'

    def fit(self, X, y, balanced=True, **kwargs):
        """
        Train the model using X (n_samples, n_features) matrix, for the two classes
        problem. To work with imbalanced probels, balanced=True can be set, which 
        oversamples the minority class (assumed to be the positive class y=1), to 
        much the number of samples in the majority class (assumed to be the negative
        class, i.e., y=0). You can set "model_id" parameters to set where tensorboard
        logs are saved, which is useful to visualize training. 
        """

        if self.force_all_finite:
            X_filled = X.copy()
            X_filled[np.isnan(X_filled)] = -1
        else:
            X_filled = X


        
        # Check that X and y have correct shape
        X, y = check_X_y(X_filled, y, force_all_finite=self.force_all_finite)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        """
        Normalize the data and store the normalization parameters. 
        """
        from sklearn import preprocessing
        self.scaler = preprocessing.StandardScaler()
        self.scaler.fit(X_filled)  # fit scaler
        X_filled = self.scaler.transform(X_filled) # normalize the training data. 

        """
        Oversample the minority class
        """
        if balanced:
            X_filled,y = upsample_minority(X_filled,y,method=self.sampling_method)
        #ratio_train = float(np.sum(y == 0)) / np.sum(y==1)
        #print("Ratio after upsampling minority class: {}".format(ratio_train))

        """
        Fit
        """
        self.model.fit(X_filled, y, **kwargs)        
        return self
    
    def predict_proba(self, X):
        # Predict the classification score (associated with the class==1)
        X = self.scaler.transform(X)  # normalized input

        # Fill the values with -1 in case of algorithm not robust to missing data
        if self.force_all_finite:
            X_filled = X.copy()
            X_filled[np.isnan(X_filled)] = -1
        else:
            X_filled = X
        y_score = self.model.predict_proba(X_filled)[:,1]  
        return y_score
    
    def predict(self, X, th=.5):
        # Check is fit had been called
        check_is_fitted(self)
        # Input validation
        X = check_array(X)
        # Normalize the sample to match the pre-processing done at training        
        y_score = self.predict_proba(X)  # predict 
        y_pred = [1 if yy>=th else 0 for yy in y_score]
        return y_pred
    

class nn_model(BaseEstimator, ClassifierMixin):
    """Wrap nn models into an sklearn estimator to leverage the flexibility of tf dnn and the convenience of sklearn pipelines. 
    """
    def __init__(self, name='ClassifierName'):
        self.name = name

    def fit(self, X, y, balanced=True, batch_size=32, epochs=100, name = 'log_id'):
        """
        Train the model using X (n_samples, n_features) matrix, for the two classes
        problem. To work with imbalanced probels, balanced=True can be set, which 
        oversamples the minority class (assumed to be the positive class y=1), to 
        much the number of samples in the majority class (assumed to be the negative
        class, i.e., y=0). You can set "model_id" parameters to set where tensorboard
        logs are saved, which is useful to visualize training. 
        """
        
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        """
        Normalize the data and store the normalization parameters. 
        """
        from sklearn import preprocessing
        self.scaler = preprocessing.StandardScaler()
        self.scaler.fit(X)  # fit scaler
        X = self.scaler.transform(X) # normalize the training data. 

        """
        Oversample the minority class
        """
        # Keep a copy of the raw data to be used for validation (specially useful when balancing the training samples)
        X_val = X.copy()
        y_val = y.copy()

        if balanced:
            X,y = upsample_minority(X,y,method='smote')
                
        """
        Convert the labels into one hot key (necessary for softmax prediction)
        """
        import tensorflow as tf
        labels = np.unique(y)
        y_one_hot = tf.one_hot(y, depth=len(labels))
        y_val_one_hot = tf.one_hot(y_val, depth=len(labels))

        """
        Initialize the network and compile it 
        """
        from tensorflow import keras
        from tensorflow.keras import layers
        from tensorflow.keras.initializers import RandomNormal as winit

        # Simple model
        def my_model1(num_feats=None):
            inputs = keras.Input(shape=(num_feats,))
            x = layers.Dense(3, activation=None)(inputs)  
            x = layers.Dense(2, activation='relu')(x)  
            outputs = layers.Dense(2, activation="softmax")(x)  
            model = keras.Model(inputs=inputs, outputs=outputs, name="diagnosis_prediction")
            return model
        
        # More complex model 
        def my_model2(num_feats=None):
            inputs = keras.Input(shape=(num_feats,))
            x = layers.Dense(5, activation=None)(inputs)  
            x = layers.Dense(5, activation='relu')(x)  
            x = layers.Dense(5, activation='relu')(x)  
            outputs = layers.Dense(2, activation="softmax")(x)  
            model = keras.Model(inputs=inputs, outputs=outputs, name="diagnosis_prediction")
            return model
        
        """
        Train the network
        """
        logdir = os.path.join('tblogs', self.name + '_' + name)  # Setup tensorboard output folder
        tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
        # Define the architecture and init the model.
        model = my_model2(num_feats = X.shape[1])
        # Set optimization parameters and compile model
        # optimizer = tf.keras.optimizers.SGD(learning_rate=0.2,name='SGD')
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.05)
        loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2)
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        # TRAIN:: 
        model.fit(x=X, y=y_one_hot, batch_size=batch_size, epochs=epochs, 
                  validation_data=(X_val,y_val_one_hot), callbacks=[tensorboard_callback], verbose=0)
        
        self.model = model  # Save the trained model. 
        return self
    
    def predict_proba(self, X):
        # Predict the classification score (associated with the class==1)
        return self.model.predict(X)[:,1]  
    
    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)
        # Input validation
        X = check_array(X)
        # Normalize the sample to match the pre-processing done at training
        X = self.scaler.transform(X)  # normalized input         
        y_score = self.predict_proba(X)  # predict 
        y_pred = np.around(y_score).astype('int')
        return y_pred


def upsample_minority(X,y,method='smote'):
    # Using smote:
    if method=='smote':
        from imblearn.over_sampling import SMOTE
        oversample = SMOTE()
        X, y = oversample.fit_resample(X, y)

    else:
        # Or using simple instance replication:
        # If balanced, upsamples the minority class to have a balanced training set. 
        X0 = X[y==0]
        y0 = y[y==0]
        X1 = X[y==1]
        y1 = y[y==1]

        # Upsample the minority class
        from sklearn import utils
        X1_upsample = utils.resample(X1, replace=True, n_samples=X0.shape[0])
        y1_upsample = utils.resample(y1, replace=True, n_samples=X0.shape[0])
        X = np.vstack((X0, X1_upsample))
        y = np.hstack((y0, y1_upsample))

    return X,y


def evaluate_model(model=None, X=None, y=None, threshold=.5, classes_names = ['0', '1'], num_cv=10, verbose=False, **kwargs):
    """
    Evaluate a sklearn model for the data X with ground truth labels y
    """
    from sklearn.metrics import classification_report
    from sklearn.metrics import plot_roc_curve
    from sklearn.model_selection import StratifiedKFold

    y_pred_score = -1*np.ones_like(y).astype('float32')  # init prediction scores 
    
    if num_cv>0:   # use cross-validation.
        cv = StratifiedKFold(n_splits=num_cv, shuffle=True, random_state=0)
        # print('Performing {} fold cross-validation.'.format(num_cv))
        for i, (train, test) in enumerate(cv.split(X, y)):
            if False and 'xgboost' in model.name:
                ratio_train = float(np.sum(y[train] == 0)) / np.sum(y[train]==1)
                model.model.scale_pos_weight = ratio_train

            if False and 'xgboost' in model.name:
                kwargs['eval_set'] = [(X[test], y[test])]

            # Fit classifier
            model.fit(X[train], y[train], balanced=True, **kwargs)
            # Predict samples on the test set
            y_pred_score[test] = model.predict_proba(X[test])

    else:  # just fit and predict the data (just for baseline, DON'T USE TO ASSES PERFORMANCE)
        model.fit(X, y, balanced= True)
        y_pred_score = model.predict_proba(X)

    """
    Plot PR and ROC curves.  
    """
    fig, recall, precision, ths_pr, fpr, tpr, ths_roc = plot_PR_and_ROC_curves(y_true=y, y_score=y_pred_score)
    
    """
    Classification performance for a operation point.  
    """
    if verbose:
        print('For a classification threshold of {:3.2f}, the performance is:'.format(threshold))
        y_pred = y_pred_score > threshold  # Predicted binary label for the given th.
        my_classification_report(y, y_pred)
    return fig, recall, precision, ths_pr, fpr, tpr, ths_roc


def my_classification_report(y_true, y_pred):
    """
    Print several performance metrics that are common in the context of screening and fraud detection.
    """    

    """
    First compute the TP, FP, TN and FN from which most metrics derive
    """
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    """
    Compute metrics of interest  
    """    
    print('Sample: {} positive and {} negative samples (#p/#n={:3.0f}%)'.format(tp+fn, tn+fp, 100*(tp+fn)/(tn+fp)))
    acc = (tp + tn) / (tp + tn + fp +  fn)
    print('Accuracy: {:3.1f}%'.format(100*acc))
    f1 = 2*tp / (2*tp + fp + fn)
    print('F1 score (2 PPVxTPR/(PPV+TPR)): {:3.1f}%'.format(100*f1))
    mcc = (tp*tn - fp*fn) / np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    print('Matthews correlation coefficient (MCC): {:3.1f}%'.format(100*mcc))
    tpr =  tp / (tp+fn)
    print('Sensitivity, recall, hit rate, or true positive rate (TPR): {:3.1f}%'.format(100*tpr))
    tnr = tn / (tn+fp)
    print('Specificity, selectivity or true negative rate (TNR): {:3.1f}%'.format(100*tnr))
    ppv = tp / (tp+fp)
    print('Precision or positive predictive value (PPV): {:3.1f}%'.format(100*ppv))
    npv = tn / (tn+fn)
    print('Negative predictive value (NPV): {:3.1f}%'.format(100*npv))
    fnr = fn / (tp+fn)
    print('Miss rate or false negative rate (FNR): {:3.1f}%'.format(100*fnr))
    print('False discovery rate (FDR=1-PPV): {:3.1f}%'.format(100*(1-ppv)))
    print('False omission rate (FOR=1-NPV): {:3.1f}%'.format(100*(1-npv)))
    

    
    return 


def plot_PR_and_ROC_curves(y_true=None, y_score=None):
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go    
    fig = make_subplots(rows=1, cols=2)
    
    ref_color = '#888E90'  # Color of reference lines
    
    # print('Model PR curve.')
    from sklearn.metrics import precision_recall_curve
    precision, recall, ths_pr = precision_recall_curve(y_true, y_score)
    
    fig.add_trace(go.Line(x=precision,y=recall, hovertext=ths_pr, name='Model PR curve'), row=1, col=1)

    # Add the iso-levels of f1
    f_scores = np.linspace(0.2, 0.8, num=4)
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        yy = f_score * x / (2 * x - f_score)
        fig.add_trace(go.Line(x=x[yy >= 0], y=yy[yy >= 0], line=dict(color=ref_color), 
                      name='f1={:2.1f}'.format(f_score)), row=1, col=1)
    fig.update_yaxes(title_text="Recall", scaleanchor = "x", scaleratio = 1, row=1, col=1)
    fig.update_xaxes(title_text="Precision", range=[0, 1], constrain='domain', row=1, col=1)
    fig.update_yaxes(range=(0, 1), constrain='domain', row=1, col=1)

    # Plot ROC curve
    from sklearn.metrics import roc_curve
    fpr, tpr, ths_roc = roc_curve(y_true, y_score)

    fig.add_trace(go.Line(x=fpr,y=tpr, hovertext=ths_roc, name='ROC curve'), row=1, col=2)
    fig.add_trace(go.Line(x=np.linspace(0,1,10), y=np.linspace(0,1,10), line=dict(color=ref_color)), row=1, col=2)
    fig.update_yaxes(title_text='TPR', scaleanchor = "x", scaleratio = 1, row=1, col=2)
    fig.update_xaxes(title_text='FPR', range=[0, 1], constrain='domain', row=1, col=2)
    fig.update_yaxes(range=(0, 1), constrain='domain', row=1, col=2)
    fig.update_layout(title="PR AND ROC curves")

    return fig, recall, precision, ths_pr, fpr, tpr, ths_roc


def visualize_classifier(model, xrange=[-1,1], yrange=[-1,1]):
    """This only works for 2D domains and will visualize the prediction of the 
    model in the intervals xrange yrange."""
    
    x0,x1,y0,y1 = xrange[0], xrange[1], yrange[0], yrange[1]
    xx = np.meshgrid(np.linspace(x0,x1,7), np.linspace(y0,y1,7))
    xs = xx[0]
    ys = xx[1]
    z = np.zeros_like(xs)
    h,w = z.shape
    for i in range(h):
        for j in range(w):
            batch = np.array([[xs[i,j],ys[i,j]]])
            pred = model.predict_proba(batch)
            z[i,j] = pred
            
    # Show the image 
    plt.imshow(z, origin=[x0,y0], extent=[x0,x1,y0,y1])
    plt.colorbar()
    return z
