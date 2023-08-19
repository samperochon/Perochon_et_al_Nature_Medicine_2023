"""
Tools for statistics data analysis. 
TODO: complete this description. 
----
Refs:
[1] Larry Wasserman, "All of Statistics, A Concise Course in Statistical Inference"
[2] Peter Bruce And Andrew Bruce, "Practical Statistics for Data Scientists"
[3] Alexander Gordon et al., "Control of the Mean Number of False Discoveries, Bonferroni and Stability of Multiple Testing".
[4] Jacob Cohen, "Things I have Learned (So Far)"
[5] Thomas Cover and Joy Thomas, "Elements of Information Theory"
[6] Richard Duda et al., "Pattern Classification"
[7] Judea Pearl et al., "Causal Inference in Statistics"
[8] Steven Kay, "Fundamentals of Statistical Signal Processing, Volume I: Estimation Theory"----
---- 
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from copy import deepcopy       
from const import EPSILON
#from numba.typed import List TODO: work on fighting against the deprecation of list in Numba, cf: https://numba.pydata.org/numba-doc/latest/reference/deprecation.html#deprecation-of-reflection-for-list-and-set-types


def feature_values_positive_to_negative_ratio(Xp=None, Xn=None, x_range=None, y_range=None, num_bins=50, verbose=1):
    """
    This functions computes the ratio 
     P(x|y=1)/P(x|y=0) and shows it in log scale. This ratio is equivalent to 
     P(x|y=1)/P(x|y=0) = (P(y=1|x)/P(y=0|x))/(P(y=1|x)/P(y=0|x)). This can be interpreted as: how more likely I expect to find a positive samples compared to a random sampling. 
     ----- 
     
    Keyword Arguments:
        Xp {1 or 2D array} -- 1 or 2D array, values of the feature for samples of the positive class. 
        Xn {1 or 2D array} -- 1 or 2D array, values of the feature for samples of the negative class. 
        verbose -- 1 display plots, 
        x_range -- [min_X, max_X] range of X feature. If none min(X) and max(X) is used. 
        y_range -- [min_Y, max_Y] range of Y feature. If none min(Y) and max(Y) is used. 
        num_bins -- Number of bins used when estimating the pdfs. 
    """
    
    if Xp.shape[1]==1:
        return feature_values_positive_to_negative_ratio_1D(Xp=Xp, Xn=Xn, verbose=verbose, x_range=x_range, num_bins=num_bins)
    elif Xp.shape[1]==2:
        return feature_values_positive_to_negative_ratio_2D(Xp=Xp, Xn=Xn, verbose=verbose, x_range=x_range, y_range=y_range, num_bins=num_bins)

def feature_values_positive_to_negative_ratio_1D(Xp=None, Xn=None, x_range=None, num_bins=50, verbose=1):
    """
    This functions computes the ratio 
     P(x|y=1)/P(x|y=0) and shows it in log scale. This ratio is equivalent to 
     P(x|y=1)/P(x|y=0) = (P(y=1|x)/P(y=0|x))/(P(y=1|x)/P(y=0|x)). This can be interpreted as: how more likely I expect to find a positive samples compared to a random sampling. 
     ----- 
     
    Keyword Arguments:
        Xp {1D array} -- 1D array, values of the feature for samples of the positive class. 
        Xn {1D array} -- 1D array, values of the feature for samples of the negative class. 
        verbose -- 1 display plots, 
        x_range -- [min_X, max_X] range of X feature. If none min(X) and max(X) is used. 
    """
    
    # 1) Estimate P(x|y) from the input data. To that end, we follows a frequentist approach, and approximate the pdf by a discrete function, with num_bins number of bins. The domain is set accordint to the x_range.  
    if x_range is None:
        X = np.hstack((Xp,Xn))
        xmin = np.quantile(X,0)  
        xmax = np.quantile(X,1)
        x_range = [xmin, xmax]
        # If the data has noise or outliers, you may want to consider useing 5% lower and upper quantiles to define x_range. Just replace "0" by "0.05" and "1" by "0.95".
        
    num_p = len(Xp)  # Number of positive samples in this set
    num_n = len(Xn)  # Number of negative samples in this set
    
    count_Xp, edges = np.histogram(Xp, bins=num_bins, range=x_range)  # Count samples per bin
    pdf_Xp = count_Xp / num_p  # Normalize to have an estimation of the prob. 
    count_Xn, _     = np.histogram(Xn, bins=num_bins, range=x_range)
    pdf_Xn = count_Xn / num_n

    eps = 1e-5  # A very small number just for numerical stability. 
    Q = np.log10( (pdf_Xp / (pdf_Xn + 1e-5) ) + 1e-5)  # P(x|y=1)/P(x|y=0) in logarithmic scale. 
    
    if verbose>0:  # Show plots 
        # Define a colormap function 
        min_val = -2; max_val = 1.5  # Recall these are in a log scale!
        colormap = define_colormap(min_value=min_val, max_value=max_val, zero=0., num_tones=20)
        plt.subplot(121)  # Plot raw histogram distribution per-class
        plt.bar(edges[:-1], pdf_Xn, width = edges[1]-edges[0], alpha=.5, color=colormap(-1)); 
        plt.bar(edges[:-1], pdf_Xp, width = edges[1]-edges[0], alpha=.5, color=colormap(.7))
        plt.xlabel('X'); plt.ylabel('Estimation of P(X|Y)'), plt.title('Positive and Negative empirical distribution of X')
                       
        plt.subplot(122);  # Plot Q
        colors = [colormap(q) for q in Q]
        plt.bar(edges[:-1], Q , width = edges[1]-edges[0], alpha=.5, color=colors)
        plt.plot(edges, 0*edges, '-k', linewidth=3)
        plt.ylim([min_val,max_val]); plt.xlim(x_range); ax = plt.gca(); plt.grid(axis='y')
        plt.xlabel('X'); plt.ylabel('Q'); plt.title('$Q = log_{10}( P(X|y=1)/P(X|y=-1) )$'); 
        
    return Q

def feature_values_positive_to_negative_ratio_2D(Xp=None, Xn=None, x_range=None, y_range=None, num_bins=50, verbose=1):
    """
    This functions computes the ratio 
     P(x|y=1)/P(x|y=0) and shows it in log scale. This ratio is equivalent to 
     P(x|y=1)/P(x|y=0) = (P(y=1|x)/P(y=0|x))/(P(y=1|x)/P(y=0|x)). This can be interpreted as: how more likely I expect to find a positive samples compared to a random sampling. 
     ----- 
     
    Keyword Arguments:
        Xp {2D array} -- 2D array, values of the feature for samples of the positive class. 
        Xn {2D array} -- 2D array, values of the feature for samples of the negative class. 
        verbose -- 1 display plots, 
        x_range -- [min_X, max_X] range of X feature. If none min(X) and max(X) is used. 
    """
    

    # 1) Estimate P(x|y) from the input data. To that end, we follows a frequentist approach, and approximate the pdf by a discrete function, with num_bins number of bins. The domain is set accordint to the x_range.  
    if x_range is None:
        X_combined = np.concatenate((Xp,Xn))
        xmin, xmax = np.quantile(X_combined[:,0],0), np.quantile(X_combined[:,0],1)
        x_range = [xmin, xmax]
    if y_range is None:
        X_combined = np.concatenate((Xp,Xn))
        ymin, ymax = np.quantile(X_combined[:,1],0), np.quantile(X_combined[:,1],1)
        y_range = [ymin, ymax]        
        # If the data has noise or outliers, you may want to consider useing 5% lower and upper quantiles to define x_range. Just replace "0" by "0.05" and "1" by "0.95".


    num_p = len(Xp)  # Number of positive samples in this set
    num_n = len(Xn)  # Number of negative samples in this set

    count_Xp, xedges, yedges = np.histogram2d(Xp[:,0], Xp[:,1], bins=num_bins, range=[x_range, y_range])  # Count samples per bin
    pdf_Xp = count_Xp / num_p  # Normalize to have an estimation of the prob. 
    count_Xn, _, _ = np.histogram2d(Xn[:,0], Xn[:,1], bins=num_bins, range=[x_range, y_range])  # Count samples per bin
    pdf_Xn = count_Xn / num_n

    eps = 1e-5  # A very small number just for numerical stability. 
    Q = np.log10( (pdf_Xp / (pdf_Xn + 1e-5) ) + 1e-5)  # P(x|y=1)/P(x|y=0) in logarithmic scale. 

    if verbose:
        from matplotlib.colors import LinearSegmentedColormap
        import seaborn as sns

        # This colormap has been designed using the awesome following cite: https://eltos.github.io/gradient/#1F77B4-FFFFFF-FF7F0E
        cmap = LinearSegmentedColormap.from_list('my_gradient', (
            # Edit this gradient at https://eltos.github.io/gradient/#1F77B4-FFFFFF-FF7F0E
            (0.000, (0.122, 0.467, 0.706)),
            (0.500, (1.000, 1.000, 1.000)),
            (1.000, (1.000, 0.498, 0.055))))     
        # Define a colormap function 
        min_val = -2; max_val = 1.5  # Recall these are in a log scale!
        colormap = define_colormap(min_value=min_val, max_value=max_val, zero=0., num_tones=20)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30,10))

        # Plot raw histogram distribution per-class
        ax1 = sns.histplot(x=Xp[:,0], y=Xp[:,1], bins=num_bins, color = colormap(.7), ax=ax1)
        sns.histplot(x=Xn[:,0], y=Xn[:,1], bins=num_bins, color = colormap(-1), ax=ax1)
        ax1.set_xlabel('x'); ax1.set_ylabel('y');ax1.set_title('Positive and Negative empirical distribution of X\nEstimation of P(X|Y)')

        # Plot Q
        xx, yy = np.mgrid[xmin:xmax:complex(0, num_bins), ymin:ymax:complex(0, num_bins)]
        ax2.set_xlim(xmin, xmax);ax2.set_ylim(ymin, ymax)
        # Contourf plot
        cfset = ax2.contourf(xx, yy, Q, cmap=cmap)
        ## Or kernel density estimate plot instead of the contourf plot
        #ax.imshow(np.rot90(Q), cmap='Blues', extent=[xmin, xmax, ymin, ymax])
        # Contour plot
        #cset = ax2.contour(xx, yy, Q, colors='k')
        # Label plot
        #ax.clabel(cset, inline=5, fontsize=20)
        cbar = plt.colorbar(cfset);ax2.set_xlabel('x'); ax2.set_ylabel('y'); ax2.set_title('$Q = log_{10}( P(X|y=1)/P(X|y=-1) )$'); 
        plt.show()
    return Q

def define_colormap(min_value=-1., max_value=1., zero=0., num_tones=10):
    """
    This is a shortcut to color quantities in tones of blue and orange. In this "ASD" screening examples, we associated orange tones with risk of ASD and blue tones with indications of TD. This function is just to simply this color mapping across different experiments. Zero is the "neutral value", and is mapped to the color white. Max value is the larges value an ASD risk factor can take, this value (and any value above this value) is mapped to the darkest orange. Min value is the lower value the risk indicator can take, and any value lower or equal is mapped to the darkest blue tone. This function returns a mapping function that you can use to compute the color of each new feature values. For example:
    colormap = define_colormap(min_value=-1, max_value=1, zero=0, num_tones=10)
    x = .5
    color_x = colormap(x)  --> returns the color "orange tone" [0.9, 0.5, 0.2, 1.0] and so on. 
    ------ 
    Arguments:
        min_value {float} -- min value associated to the lower asd risk
        max_value {float} -- max value associated to the highest asd risk
        zero {float} -- zero value, associated to the neutral value (no bias)
        num_tones {int} -- number of different tones (the more the more continuos)   
    Returns:
        colormap [function R->R^4] -- Map values to their color. 
    """
    from matplotlib import cm

    blue = cm.get_cmap('Blues', num_tones)   #  grab blue pallette 
    orange = cm.get_cmap('Oranges', num_tones)  # grab orange pallette
    
    def colormap(z):
        if z>zero:
            i = min(max_value-zero, z-zero)/(max_value-zero)  # [zero, max_value] --> [0,1] 
            color = orange(i)
        else:
            i = max(min_value-zero, z-zero)/(min_value-zero)  # [min_value,0] --> [1,0] 
            color = blue(i)
        return color
    
    return colormap


############## Weighted scheme data imputation            
#@jit(nopython=True, parallel=True)
def impute_missing_data(X_train, X_test, method='multi_dimensional_weighting', h=.2):
    """
    Imputation of the missing values of the different rows of X_test based on X_train. 
    X_prior contains the samples for which there is no missing values, which are used as prior when coputing the missing coordinate of a sample.
    """

    k = X_test.shape[1]  # dimension of the space of samples. 
    K = lambda u: 1/np.sqrt(2*np.pi) * np.exp(-u**2 / 2)  # Define the kernel

    if method == 'multi_dimensional_weighting':
        def W(X_1,X_2):
            """
            Weight between two samples X_1 and X_2 to measure the proximity of the hyperplane (in the case of missing values.)
            W(X_1,X_2) = e^( -1/2 1/h**2 sum_k(x_1k-x_2k)**2 )  for k s.t. x_1k and x_2k isn't nan. 
            """
            def dist(X_1,X_2):
                ks = [i for i in range(len(X_1)) if not np.isnan(X_1[i]) and not np.isnan(X_2[i])]
                if not ks:  # if ks is empty the distance can't be computed
                    return np.nan

                d = 0
                for k in ks:
                    x_1 = X_1[k]
                    x_2 = X_2[k]
                    d += (x_1-x_2)**2
                return np.sqrt(d)

            d = dist(X_1, X_2)
            if np.isnan(d):
                # Is the vectors don't share at least one common coordinate 
                return 0

            W = K( d/h )
            return W
            
        # Compute prior set.
        m = [not np.isnan(np.sum(X_train[i,:])) for i in range(X_train.shape[0])]
        X_prior = X_train[m,:]

        # Init. the imputed test set.
        imp_X_test = deepcopy(X_test)

        # TODO: KEep track of the weights ? As a measure of confidence for the imputation ?

        for i, X_i in enumerate(X_test):

            # Perform imputation if needed
            coords_missing = np.isnan(X_i)  # unknown coordinates of X_i
            for j in range(k):        
                if coords_missing[j]:  # we don't know the j-th coordinate, we need to impute it

                    # We use the term associate to the j-th coordinate for the 
                    # rest of the samples in the training set (for which the j-th component is know).
                    # The contribution of each term is weighted with the distance to the sample hyperplane.
                    hat_X_ij = 0
                    Ws = 1e-10  # eps
                    for X_p in X_prior:
                        w_p = W(X_i, X_p)
                        hat_X_ij += w_p * X_p[j]
                        Ws += w_p
                    hat_X_ij /= Ws

                    imp_X_test[i, j] = hat_X_ij
    elif method=='without':
        imp_X_test = X_test

    elif method=='naive':
        # Ignore missing values
        imp_X_test = X_test[~np.isnan(X_test[:,0]),:]
        imp_X_test = imp_X_test[~np.isnan(imp_X_test[:,1]),:]        
                

    elif method=='mean':
        from sklearn.impute import SimpleImputer
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp_X_test = imp.fit_transform(X_test)
      
    elif method=='median':
        from sklearn.impute import SimpleImputer
        imp = SimpleImputer(missing_values=np.nan, strategy='median')
        imp_X_test = imp.fit_transform(X_test)
      
    elif method=='knn':
        from sklearn.impute import KNNImputer
        knn_imputer = KNNImputer()
        imp_X_test = knn_imputer.fit_transform(X_test)
        
    elif method=='mice':
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        imp = IterativeImputer(n_nearest_features=None, imputation_order='ascending')
        imp_X_test = imp.fit_transform(X_test)

    return imp_X_test
