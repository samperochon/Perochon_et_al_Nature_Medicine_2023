import os
import sys 
import json
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from ast import literal_eval

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from scipy.stats import mannwhitneyu
from imblearn.over_sampling import SMOTE

# add tools path and import our own tools
sys.path.insert(0, '../src')

from const import *
from const_autism import *
from utils import repr, select


class Dataset(object):
    """
    This class aims at handle the S2K dataset. 


    Initialization step:
        1) Post process the dataframe.
            1) Add a study, remote (bool) field
            2) Add ASD diagnosis to SAESDM IMPACT and P3R
            3) Encode diagnosis, ethnicity, race, sex, StateOfTHeChild, Comments. 
            4) Add norder of the administration by using adm timing. 
            5) Compute the Aggregated CVA biomarkers (S/NS gaze, postural sway, etc.)
        2) Filter the dataframe using a pre-defined scenario (based on age, study, remote or not, etc.)
        3) Create the X and y array. Optional scaling and addition of missing indicator variabels are performed.
        4) Split the dataset into train and test sets. 
        5) Impute or encode the missing variables.
    """

    
    def __init__(self, 
                df,
                dataset_name='SenseToKnow',
                outcome_column='diagnosis',
                positive_class = None,
                features_name=DEFAULT_PREDICTORS, 
                scenario=None,
                missing_data_handling='encoding',
                imputation_method='without',
                scale_data=DEFAULT_SCALE_DATA,
                sampling_method=DEFAULT_SAMPLING_METHOD,
                proportion_train=PROPORTION_TRAIN,
                use_missing_indicator_variables=DEFAULT_USE_INDICATOR_VARIABLE,
                verbosity=4,
                debug=False, 
                random_state=RANDOM_STATE):
        
        self.dataset_name = dataset_name
        self.outcome_column = outcome_column
        self.positive_class = positive_class
        self.proportion_train = proportion_train
        self.scale_data = scale_data
        self.sampling_method = sampling_method
        
        self.verbosity = verbosity    
        self.debug = debug    
        
        # Imputed data (depend on the experiences/settings). If state is training, it contains imputation of 
        # the train set, otherwise the test set.
        self._imp_X_train, self._imp_X_test = None, None 
        self.train_index, self.test_index = None, None 
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        
        

        #`self.use_indicator_variable` can be a True or False, 
        #True: double the number of features by adding all missing variables
        #False: None are used
        #Dict: {high-level name of the missingness: [feat_1, feat2]}
        self.use_missing_indicator_variables = use_missing_indicator_variables
        
        # Init features name
        self.raw_features_name = deepcopy(features_name)
        self._features_name = self._init_features_name(deepcopy(features_name))
        
        # Init handling of missing data values
        self.missing_data_handling = missing_data_handling
        self.imputation_method = imputation_method 
        
        #
        # 1) Post-process the dataset
        # 
        self.df = self._post_process_df(df)
        self._raw_df = deepcopy(self.df)
        self.num_samples = len(self.df)
        
        #
        # 2) Filter the dataset according to pre-defined scenario
        # 
        self.scenario = self._init_scenario(scenario)
        

        #
        # 3) Initialize the raw X and y data.
        #         
        self._X, self._y = self._init_data()
        
        
        #
        # 4) Split the dataset into train and test set.
        #     
        self.split_test_train()
        
        #
        # 5) Impute or encode missing data. 
        #     
        self.impute_data()
        
        
        # Derive statistics etc. 
        
        
        self.imbalance_ratio = np.sum(self._y==1)/len(self._y)
        self.ratio_of_missing_values = np.isnan(self._X).sum()/(self._X.shape[0]*self._X.shape[1])
        self.ratio_missing_per_class = [np.isnan(self._X[(self._y==0).squeeze()]).sum()/(self._X[(self._y==0).squeeze()].shape[0]*self._X.shape[1]), 
                                        np.isnan(self._X[(self._y==1).squeeze()]).sum()/(self._X[(self._y==1).squeeze()].shape[0]*self._X.shape[1])]


        self.dataset_description = 'Number of samples: {}'.format(self.num_samples)
        self.cmap = sns.color_palette(plt.get_cmap('tab20')(np.arange(0,2)))

        self.random_state = random_state
        np.random.seed(random_state)

    def __call__(self):
        return repr(self)

    @property
    def features_name(self):
        return self._features_name

    @features_name.setter  
    def features_name(self, features_name):
        self._features_name = self._init_features_name(features_name)
        self._X, _ = self._init_data()
        self.ratio_of_missing_values = np.isnan(self._X).sum()/(self._X.shape[0]*self._X.shape[1])
        self.ratio_missing_per_class = [np.isnan(self._X[(self._y==0).squeeze()]).sum()/(self._X[(self._y==0).squeeze()].shape[0]*self._X.shape[1]), 
                                        np.isnan(self._X[(self._y==1).squeeze()]).sum()/(self._X[(self._y==1).squeeze()].shape[0]*self._X.shape[1])]

        self.split_test_train()

    def _reset(self):
        """
            Reset the df to the ost-processed df. 
        """
        self.df = deepcopy(self._raw_df)
        return 

    def filter(self, administration=None, features=None, validity=None, clinical=None, matching=None, demographics=None, other=None, verbose=True):
        """
            1) Reset the df to the post-processed one. 
            2) Filter based on age, demographics, clinical etc. 

            Example:
                self.filter(administration={'order': 'first', 
                                            'complete': True}, 
                            clinical={'diagnosis': [0, 1]}, 
                            demographics={'age':[18, 36], 
                                        'sex': 'Male'},
                            features={'having':['gaze', 'touch']},
                            other={'StateOfTheChild':['Slightly irritable', 'In a calm and/or good mood']}
                            )

        """
        self._reset()

        if administration is not None:

            if 'studies' in administration.keys():

                if not isinstance(administration['studies'], list):
                    administration['studies'] = [administration['studies']]

                indexes_to_drop = self.df[~self.df['study'].isin(administration['studies'])].index

                if self.verbosity>1 and verbose:
                    print("Removing {}/{} keeping only subject in studies: {}.".format(len(indexes_to_drop), len(self.df), str(administration['studies'])))
                    
                self.df.drop(index=indexes_to_drop, inplace=True)


            if 'complete' in administration.keys() or 'completed' in administration.keys():

                indexes_to_drop = self.df[(self.df['validity_available']==1) & (self.df['completed']==1) ].index

                if self.verbosity>1 and verbose:
                    print("Removing {}/{} incomplete administrations.".format(len(indexes_to_drop), len(self.df)))

                self.df.drop(index=indexes_to_drop, inplace=True)

            if 'order' in administration.keys():

                if administration['order'] == 'first':
                    
                    

                    indexes_to_drop = self.df[self.df['administration_number'] != 1].index
                    if self.verbosity>1 and verbose:
                        print("Removing {}/{} keeping first admin.".format(len(indexes_to_drop), len(self.df)))

                    self.df.drop(index=indexes_to_drop, inplace=True)
                
                elif administration['order'] == 'last':

                    indexes_to_drop = self.df[self.df.duplicated(subset=['id'], keep='last')].index

                    if self.verbosity>1 and verbose:
                        print("Removing {}/{} keeping last admin.".format(len(indexes_to_drop), len(self.df)))

                    self.df.drop(index=indexes_to_drop, inplace=True)
            
                elif administration['order'] == 'test-retest':
                    
                    indexes_to_drop = df[~df.duplicated(subset=['id'], keep=False)].index

                    if self.verbosity>1 and verbose:
                        print("Removing {}/{} keeping only subject with multiple administrations.".format(len(indexes_to_drop), len(self.df)))

                    self.df.drop(index=indexes_to_drop, inplace=True)

        if clinical is not None:

            if 'diagnosis' in clinical.keys():
                indexes_to_drop = self.df[~self.df['diagnosis'].isin(clinical['diagnosis'])].index

                if self.verbosity>1 and verbose:
                    print("Removing {}/{} keeping only subject with diagnosis: {}.".format(len(indexes_to_drop), len(self.df), clinical['diagnosis']))
                    
                self.df.drop(index=indexes_to_drop, inplace=True)

        if demographics is not None:

            if 'age' in demographics.keys():
                indexes_to_drop =  self.df[~((self.df['age'] >= demographics['age'][0]) & (self.df['age'] <= demographics['age'][1]))].index

                if self.verbosity>1 and verbose:
                    print("Removing {}/{} keeping only subject with age between {} and {} mo.".format(len(indexes_to_drop), len(self.df), demographics['age'][0], demographics['age'][1]))

                self.df.drop(index=indexes_to_drop, inplace=True)

            if 'sex' in demographics.keys():
                indexes_to_drop =  self.df[~(self.df['sex'] != demographics['sex'])].index

                if self.verbosity>1 and verbose:
                    print("Removing {}/{} keeping only subject with sex: {}.".format(len(indexes_to_drop), len(self.df), demographics['sex']))
                    
                self.df.drop(index=indexes_to_drop, inplace=True)
                
        if matching is not None:
            
            if 'age' in matching.keys():
                
                assert len(self.df['diagnosis'].unique())==2, "Make sure there are only two diagnosis group left in the dataset."
                
                indexes_to_drop = self._match_age(other_group=0, target_group=1)
                
                if self.verbosity>1 and verbose:
                    print("Removing {}/{} to match age. (removed diagnosis group : 0).".format(len(indexes_to_drop), len(self.df)))
                    
                self.df.drop(index=indexes_to_drop, inplace=True)
        if other is not None:

            for column, admissible_values in other.items():

                indexes_to_drop =  self.df[~self.df[column].isin(admissible_values)].index

                if self.verbosity>1 and verbose:
                    print("Removing {}/{} keeping only subject with column: {} in {}.".format(len(indexes_to_drop), len(self.df), column, admissible_values))
                    
                self.df.drop(index=indexes_to_drop, inplace=True)

        if self.verbosity > 1 and verbose:
            print("{} administrations left.".format(len(self.df)))
            
            display(self.df.groupby(self.outcome_column)[['id']].count())

        self.df.sort_values(by=['id', 'date'], inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        self.num_samples = len(self.df)
        
        # Reset all variables.
        self._X, self._y   = None, None
        self.imbalance_ratio = None 
        self.ratio_of_missing_values = None
        self.ratio_missing_per_class =None

        return

    def split_test_train(self):
        
        """
            Create the x_train, X_test, y_train, y_test 
            Split the dataset given some proportions, taking as input the see-able dataset with potentially missing data, having them either encoded of imputed.
        """
        if self.proportion_train not in [0, 1]:
            self.train_index, self.test_index = train_test_split(np.arange(self.num_samples), test_size = 1-self.proportion_train)

        elif self.proportion_train == 0:
            self.train_index, self.test_index = [], np.arange(self.num_samples)
            
        elif self.proportion_train == 1:
            self.train_index, self.test_index = np.arange(self.num_samples), []


        self._X_train = deepcopy(self._X[self.train_index])
        self._X_test = deepcopy(self._X[self.test_index])

        self._y_train = deepcopy(self._y[self.train_index])
        self._y_test = deepcopy(self._y[self.test_index])


        # By default once the split is done, no matter the state of the dataset, data are pusshed to the forefront.
        self.X_train = deepcopy(self._X_train)
        self.X_test = deepcopy(self._X_test)
        self.y_train = deepcopy(self._y_train)
        self.y_test = deepcopy(self._y_test)

        print("Splitting dataset into test and train set.") if (self.debug or self.verbosity > 1) else None

        return

    def impute_data(self):
        """
            Must be called just after the `split_test_train` method.
            Change the visible variable self.X_, depending on the class, and handling of missing data.

            If the approach are `bayesian` (computing distributions), at training time (state=='training'), self.X_train will contain 
            the the data from the training set, with potential missing value, for a certain class. 
            At inference time it will contain the test set, with potential missing data, for all classes of course. 
        """

    
        if self.missing_data_handling == 'imputation':
            
            if self.imputation_method != 'constant':

                self._imp_X_train, self._imp_X_test  = self._impute_missing_data()

                # Create X and y used for experiments
                self.X_train = deepcopy(self._imp_X_train)
                self.y_train = deepcopy(self._y_train)

                self.X_test = deepcopy(self._imp_X_test)
                self.y_test = deepcopy(self._y_test)
                
                #print("Imputed {} values (train) and {} (test) using method {}.".format(len(np.isnan(self.X_train)), len(np.isnan(self.X_test)), self.imputation_method)) if (self.debug or self.verbosity > 2)  else None   
            else:
                                
                self.X_train = deepcopy(self._X_train)
                self.X_train[np.isnan(self.X_train)] = DEFAULT_MISSING_VALUE
                self.y_train = deepcopy(self._y_train)

                self.X_test = deepcopy(self._X_test)
                self.X_test[np.isnan(self.X_test)] = DEFAULT_MISSING_VALUE
                self.y_test = deepcopy(self._y_test)

                print("Encoding {} (train) and {} (test) missing values with {}.".format(len(np.isnan(self._X_train)), len(np.isnan(self._X_test)), DEFAULT_MISSING_VALUE)) if (self.debug or self.verbosity > 2)  else None

        elif self.missing_data_handling == 'without':

            self.X_train = deepcopy(self._X_train)
            self.y_train = deepcopy(self._y_train)
            self.X_test = deepcopy(self._X_test)
            self.y_test = deepcopy(self._y_test)

        else:
            raise ValueError("Please use one of the following missing variables handling: imputation, encoding, or without")

        return 
    
    def upsample_minority(self, X, y):
        
        # Using smote:
        if self.sampling_method=='smote':
            
            oversample = SMOTE()
            X, y = oversample.fit_resample(X, y)
            print("Upampling minority class. Imbalance ratio of: {:.2f} to {:.2f}".format(self.imbalance_ratio, np.sum(y==1)/np.sum(y==0))) if self.verbosity > 1 else None

        elif self.sampling_method=='vanilla':
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

            print("Upampling minority class. Imbalance ratio of: {:.2f} to {:.2f}".format(self.imbalance_ratio, np.sum(y==1)/np.sum(y==0))) if self.verbosity > 1 else None
                  
        elif self.sampling_method in ['without', 'scale_pos_weight']:
            pass
        
        else:
            raise ValueError("Please use one of the following upsampling method: smote, vanilla, or without")
        
        return X,y

    def plot(self):

        self._plot_missing()

        return

    def _impute_missing_data(self, bandwidth=BANDWIDTH):
        """
            If state is training, we impute the missing data of the training set with itself.
            If state is inference, we impute the missing data of the test set with the training set.
            TODO: veriify that we are doing this once... (the imputation computation), because it's better to store than to recompute everytime.
        """
        from stats import impute_missing_data

        if self.proportion_train >0:
            _imp_X_train = impute_missing_data(X_train=self.X_train, X_test=self.X_train, method=self.imputation_method, h=bandwidth)
        else:
            _imp_X_train = np.array([])

        if self.proportion_train <1:
            _imp_X_test = impute_missing_data(X_train=self.X_train, X_test=self.X_test, method=self.imputation_method, h=bandwidth)
        else:
            _imp_X_test = np.array([])

        return _imp_X_train, _imp_X_test
    
    def save(self, experiment_path):

        # Unneccesary to save.

        #-------- Save dataset ----------#
        with open(os.path.join(experiment_path, 'dataset_log.json'), 'w') as outfile:
            json.dump(self, outfile, default=lambda o: o.astype(float) if type(o) == np.int64 else o.tolist() if type(o) == np.ndarray else o.to_json(orient='records') if type(o) == pd.core.frame.DataFrame else o.__dict__)
            
        # Reload the object that were unsaved 

        return
    
    def load(self, dataset_data):

        for key, value in dataset_data.items():
            if isinstance(value, list):
                setattr(self, key, np.array(value))
            else: 
                setattr(self, key, value)
        #self.features_name =  list(np.array(self.features_name).astype(str))
        #print(self.features_name)
        print('yo')
        
    def get_data(self):
        return self.X, self.Z, self.y

    def _post_process_df(self, df):
        """
            Post-process the raw dataframe. Note that some of these steps could be done at the API level when building the dataframe.
            1) Add a study, remote (bool) field
            2) Add ASD diagnosis to SAESDM IMPACT and P3R
            3) Add EHR data diagnosis
            3) Encode diagnosis, ethnicity, race, sex, StateOfTHeChild, Comments. 
            4) Add norder of the administration by using adm timing. 
            5) Compute the Aggregated CVA biomarkers (S/NS gaze, postural sway, etc.)
            6) Compute the confidence of each variables
        """

        if self.verbosity > 1:
            print("Post-processing inital df (removing columns with no cva features, encoding srings, compute administrations order, compute condensed S/NS variables)... ")

        # Delete rows wthout any features
        df.dropna(subset=MINIMAL_SET_OF_FEATURES, how='all', inplace=True)
        
        df['study'] = df['path'].apply(lambda x: x.split('/')[-3] if x.split('/')[-3] in S2K_STUDIES else x.split('/')[-4])
        df.loc[df['study'].isin(['SAESDM', 'IMPACT', 'P3R']), 'diagnosis'] = 'ASD'
        df['remote'] = df['study']
        df['remote'].replace({'SenseToKnowStudy':1, 
                    'P1':0,
                    'P2':0, 
                    'P3':0,
                    'IMPACT':0,
                    'SAESDM':0,
                    'ARC':0,
                    'P3R':1, 
                    'S2KP':0,
                    'P1R':1}, inplace = True)   
        
        # Add EHR diagnosis
        df = self._add_ehr_diagnosis(df, verbose=True if self.verbosity >=2 else False)     

        # encode categorical variables
        df['diagnosis'].replace({'TD':0., 
                                'ASD':1., 
                                'DDLD':2., 
                                'ADHD':3.,
                                 'Other':4., 
                                 np.nan: -1}, inplace = True)
    
        df['ethnicity'].replace({'Not Hispanic/Latino':0, 
                                'Hispanic/Latino':1, 
                                'Unknown or not reported':np.nan}, inplace = True)
        
        if True:
            df['race'].replace({'White':0., 
                        'White/Caucasian':0.,
                        'Black/African American':1., 
                        'More than one race':2.,
                        'American Indian/Alaskan Native':2.,
                        'Other':2.,
                        'Asian':2.,
                        'Unknown or not reported':np.nan,
                        'Unknown/Declined':np.nan,
                    }, inplace = True)

        df['sex'].replace({'M':0, 'F':1}, inplace=True)
        df['completed'].replace({'Complete (Do not readminister)':0, 'Partial (Do not readminister)':1, 'Incomplete (Readminister at next visit)':2}, inplace = True)

        df['StateOfTheChild'].replace({'In a calm and/or good mood':1, 'Slightly irritable':2, 'Somewhat distressed':3, 'Crying and/or tantrum':4}, inplace = True)

        df.replace('N.A', np.nan, inplace=True)
        df.loc[df['SiblingsInTheRoom']==9, 'SiblingsInTheRoom'] = np.nan

        df.loc[~df['Comments'].isnull(), 'Comments'] = 1
        df.loc[~df['AppTeamComment'].isnull(), 'AppTeamComment'] = 1
        
        df['valid_name_calls'] = df['valid_name_calls'].apply(literal_eval)
        
        # Add Blink data that came afterwards
        df = self._add_blink_data(df)

        # Merge time information and create `administration_number` column
        df = self._retrieve_administration_timing(df)

        # Merge postural sway variables into social and non-social 
        df = self._compute_cva_condensed_variables(df)
        
        # Compute linear mapping of the confidence on each variables
        df = self._compute_features_confidence(df)
        
        # Add Z_variables to predictors
        df = self._add_Z_variables(df)
        
        # Add Confidence
        df = self._compute_features_confidence(df)

        # Sort df
        df.sort_values(by=['id', 'date'], inplace=True)

        return df 
       
    def _init_data(self, verbose=None):
        """
            Initialize the X and y arrays. 
            
            1) Take from the df the original cva columns. 
            2) Scale the array (optional).
            3) Add potential indicator variable depending on the value of `self.use_indicator_variables`
        
        """

        # Init. the X array. 
        features_name = deepcopy(self.raw_features_name)
        
        if isinstance(self.use_missing_indicator_variables, dict):
            
            # Grab data
            
            X_raw = self.df[features_name].to_numpy().astype(float)
            
            if self.scale_data:
        
                scaler = StandardScaler()
                scaler.fit(X_raw)  # fit scaler
                X_raw = scaler.transform(X_raw)
                
                
            for feature_name_grouped, feats in self.use_missing_indicator_variables.items():
                self.df['Z_{}'.format(feature_name_grouped)] = 0
                self.df.loc[self.df.drop(index=self.df.dropna(subset=feats, how='any').index).index, 'Z_{}'.format(feature_name_grouped)] = 1
                X_raw = np.concatenate([X_raw, self.df["Z_{}".format(feature_name_grouped)].to_numpy().astype(int)[:, np.newaxis]], axis=1)  
        
        elif self.use_missing_indicator_variables:
            
            X_raw = self.df[features_name].to_numpy().astype(float)
            
            if self.scale_data:
        
                scaler = StandardScaler()
                scaler.fit(X_raw)  # fit scaler
                X_raw = scaler.transform(X_raw)

            X_raw = np.concatenate([X_raw, (~np.isnan(self.df[features_name].to_numpy().astype(float))).astype(int)], axis=1) 

        else:
            
            X_raw = self.df[features_name].to_numpy().astype(float)
            
            if self.scale_data:
        
                scaler = StandardScaler()
                scaler.fit(X_raw)  # fit scaler
                X_raw = scaler.transform(X_raw)
            
            
        # Init the y.
        if self.outcome_column in list(self.df.keys()):
            y = self.df[self.outcome_column].to_numpy().astype(float)
            
            if self.positive_class is not None:
                #marking the current class as 1 and all other classes as 0
                y = np.array([1 if x in self.positive_class else 0 for x in y])
    
        elif self.outcome_column[:2] == 'Z_':
            y = (~np.isnan(self.df[self.outcome_column[2:]].to_numpy().astype(float))).astype(int)

        

        if self.verbosity>1 and verbose != False:
            print("Predicting {} based on {} features".format(self.outcome_column, len(self.features_name)))
            
        return X_raw, y
    
    def _add_ehr_diagnosis(self, df, verbose=True):
        
            """
                Input: df with columns `id` as strings, and `diagnosis` 
                column with entries in ['ASD', 'DDLD', 'TD', 'Other', 'ADHD', nan].
                
            """
            

            # Load EHR data and reformat
            ehr_data = pd.read_csv(os.path.join(DATA_DIR, 'P1_EHR_FINAL.csv'))
            ehr_data.rename(columns={'ace_id':'id'}, inplace=True)
            ehr_data['id'] = ehr_data['id'].astype(str)
            del ehr_data['sex']

            # First we select the app data that are included in the EHR dataset
            df_merge = pd.merge(df, ehr_data, how='left', on='id')
            diagnosis_category = list(df_merge.diagnosis.unique())

            if verbose: 
                
                # Sanity check: Does all the ASD have a asd_dx to 1 ? 
                print("Subjects diagnosed with ASD as per our data but not ASD as per the EHR: {}".format(len(select(select(df_merge, 'diagnosis', 'ASD'), 'asd_dx', 0))))

                # Subjects with TD diagnosis actually having ASD:
                print("Subjects with TD diagnosis actually having ASD: {}".format(len(select(select(df_merge, 'diagnosis', 'TD'), 'asd_dx', 1))))

                # Subjects with Unknown diagnosis actually having ASD:
                print("Subjects with Unknown diagnosis actually having ASD: {}".format(len(select(df_merge[(df_merge['diagnosis'] == 'Other') | (df_merge['diagnosis'].isnull())], 'asd_dx', 1))))

                # Subjects with ADHD actually having ASD:
                print("Subjects with ADHD diagnosis actually having ASD: {}".format(len(select(select(df_merge, 'diagnosis', 'ADHD'), 'asd_dx', 1))))

                # Subjects with DDLD actually having ASD:
                print("Subjects with DDLD diagnosis actually having ASD: {}".format(len(select(select(df_merge, 'diagnosis', 'DDLD'), 'asd_dx', 1))))

                # Subjects with Unknown diagnosis actually having DDLD:
                print("Subjects with Unknown diagnosis actually having DDLD: {}".format(len(select(df_merge[(df_merge['diagnosis'] == 'Other') | (df_merge['diagnosis'].isnull())], 'ddld_dx', 1))))
            
                dict_maching = {'ASD':'asd_dx', 
                                'DDLD': 'ddld_dx',
                                'ADHD': 'adhd_dx'
                            }

                transition_matrix = np.zeros((len(diagnosis_category), 3))

                for i, initial_diag in enumerate(diagnosis_category):

                    if initial_diag not in ['ASD', 'DDLD', 'TD', 'Other', 'ADHD']:

                        for j, (new_diag, code) in enumerate(dict_maching.items()):                

                            transition_matrix[i][j] = len(select(df_merge[df_merge['diagnosis'].isnull()], code, 1))
                    else:

                        for j, (new_diag, code) in enumerate(dict_maching.items()):

                            transition_matrix[i][j] = len(select(select(df_merge, 'diagnosis', initial_diag), code, 1))

                transition_matrix = pd.DataFrame(transition_matrix, columns = ['ASD', 'DDLD', 'ADHD'], index=diagnosis_category)
                display(transition_matrix)


            # Add the updated_diagnosis column
            df['diagnosis'] = np.nan
            

            # Set the unknown diagnosis or TD having ASD as ASD
            df_merge.loc[((df_merge['diagnosis'] == 'Other') | (df_merge['diagnosis'].isnull()) | (df_merge['diagnosis'] == 'TD')) &  df_merge['asd_dx'] == 1, 'diagnosis'] = 'ASD'
            
            # Set the unknown diagnosis or TD having DDLD as DDLD
            df_merge.loc[((df_merge['diagnosis'] == 'Other') | (df_merge['diagnosis'].isnull()) | (df_merge['diagnosis'] == 'TD')) &  df_merge['ddld_dx'] == 1, 'diagnosis'] = 'DDLD'

            # Set the unknown diagnosis or TD having ADHD as ADHD
            df_merge.loc[((df_merge['diagnosis'] == 'Other') | (df_merge['diagnosis'].isnull()) | (df_merge['diagnosis'] == 'TD')) &  df_merge['adhd_dx'] == 1, 'diagnosis'] = 'ADHD'


            # Set the DDLD with ASD as ASD
            df_merge.loc[((df_merge['diagnosis'] == 'DDLD')) &  df_merge['asd_dx'] == 1, 'diagnosis'] = 'ASD'

            # Set the TD diagnosis having DDLD as DDLD
            df_merge.loc[((df_merge['diagnosis'] == 'TD')) &  df_merge['ddld_dx'] == 1, 'diagnosis'] = 'DDLD'

            # Set the TD diagnosis having ADHD as ADHD
            df_merge.loc[((df_merge['diagnosis'] == 'ADHD')) &  df_merge['adhd_dx'] == 1, 'diagnosis'] = 'ADHD'
            
            return df_merge

    def _retrieve_administration_timing(self, df):

        df.loc[df['time'].isna(), 'time'] = '00:00'
        df['date'] = pd.to_datetime(df["date"]+' '+df['time'])

        #del df['time']

        df['administration_number'] = np.nan

        for i, d in df.sort_values(by=['date']).groupby('id'):
            if len(d) == 1:
                df.loc[d.index, 'administration_number'] = 1
                
            else:
                df.loc[d.index, 'administration_number'] = np.arange(1, len(d)+1).astype(int)

        return df

    def _add_blink_data(self, df, blink_data_path=BLINK_DATA_PATH):
        
        blink_col = ['FB_blink_rate','DIGC_blink_rate','DIGRRL_blink_rate','ST_blink_rate','MP_blink_rate','PB_blink_rate','BB_blink_rate','RT_blink_rate','MML_blink_rate','PWB_blink_rate','FP_blink_rate']
        
        df_blink = pd.read_csv(blink_data_path, usecols=['participant_id', 'study', 'date'] + blink_col)
        df_blink.rename(columns={'participant_id':'id'}, inplace=True)
        df_blink['id'] = df_blink['id'].apply(lambda x: str(x))

        df = pd.merge(left=df, right=df_blink, how = 'outer', on=['id', 'date', 'study'])
        
        return df   
    
    def _compute_cva_condensed_variables(self, df):
        '''
            Merge postural sway variables into social and non-social 
        '''


        S_postural_sway = df[['ST_postural_sway', 'BB_postural_sway', 'MML_postural_sway', 'FP_postural_sway', 'PWB_postural_sway']].mean(axis=1) # 'RT_postural_sway',
        NS_postural_sway = df[['DIGC_postural_sway', 'DIGRRL_postural_sway', 'FB_postural_sway', 'MP_postural_sway']].mean(axis=1)
        df['S_postural_sway'] = S_postural_sway
        df['NS_postural_sway'] = NS_postural_sway
        
        S_postural_sway_derivative = df[['ST_postural_sway_derivative', 'BB_postural_sway_derivative', 'MML_postural_sway_derivative', 'FP_postural_sway_derivative', 'PWB_postural_sway_derivative']].mean(axis=1) # 'RT_postural_sway_derivative',
        NS_postural_sway_derivative = df[['DIGC_postural_sway_derivative', 'DIGRRL_postural_sway_derivative', 'FB_postural_sway_derivative', 'MP_postural_sway_derivative']].mean(axis=1)
        df['S_postural_sway_derivative'] = S_postural_sway_derivative
        df['NS_postural_sway_derivative'] = NS_postural_sway_derivative
        
        # Merge silhouette scores
        #gaze_silhouette_score = df[['BB_gaze_silhouette_score','S_gaze_silhouette_score' 'FP_gaze_silhouette_score']].mean(axis=1)
        gaze_silhouette_score = df[['BB_gaze_silhouette_score','S_gaze_silhouette_score']].mean(axis=1)
        df['gaze_silhouette_score'] = gaze_silhouette_score

        # Merge percent right 
        inv_S_gaze_percent_right = 1-df['S_gaze_percent_right']
        df['inv_S_gaze_percent_right'] = inv_S_gaze_percent_right
        mean_gaze_percent_right = df[['BB_gaze_percent_right', 'inv_S_gaze_percent_right']].mean(axis=1)
        df['mean_gaze_percent_right'] = 1-mean_gaze_percent_right
        
        df['S_postural_sway_complexity'] = df[['ST_head_movement_complexity', 'BB_head_movement_complexity', 'MML_head_movement_complexity', 'FP_head_movement_complexity', 'PWB_head_movement_complexity']].mean(axis=1)
        df['NS_postural_sway_complexity'] = df[['DIGC_head_movement_complexity', 'DIGRRL_head_movement_complexity', 'FB_head_movement_complexity', 'MP_head_movement_complexity']].mean(axis=1)

        df['S_facing_forward'] = df[['ST_facing_forward', 'BB_facing_forward', 'MML_facing_forward', 'FP_facing_forward', 'PWB_facing_forward']].mean(axis=1)
        df['NS_facing_forward'] = df[['DIGC_facing_forward', 'DIGRRL_facing_forward', 'FB_facing_forward', 'MP_facing_forward']].mean(axis=1)

        df['S_eyebrows_complexity'] = df[['ST_eyebrows_complexity', 'BB_eyebrows_complexity', 'MML_eyebrows_complexity', 'FP_eyebrows_complexity', 'PWB_eyebrows_complexity']].mean(axis=1)
        df['NS_eyebrows_complexity'] = df[['DIGC_eyebrows_complexity', 'DIGRRL_eyebrows_complexity', 'FB_eyebrows_complexity', 'MP_eyebrows_complexity']].mean(axis=1)

        df['S_mouth_complexity'] = df[['ST_mouth_complexity', 'BB_mouth_complexity', 'MML_mouth_complexity', 'FP_mouth_complexity', 'PWB_mouth_complexity']].mean(axis=1)
        df['NS_mouth_complexity'] = df[['DIGC_mouth_complexity', 'DIGRRL_mouth_complexity', 'FB_mouth_complexity', 'MP_mouth_complexity']].mean(axis=1)
        
        df['S_blink_rate']  = df[[ 'MML_blink_rate', 'FP_blink_rate', 'PWB_blink_rate']].mean(axis=1) # 'ST_blink_rate', 'BB_blink_rate',
        df['NS_blink_rate'] = df[['DIGC_blink_rate', 'DIGRRL_blink_rate', 'FB_blink_rate', 'MP_blink_rate']].mean(axis=1)
        
        return df

    def _compute_features_confidence(self, df):
        
        df['S_postural_sway_conf'] = (~df[['ST_postural_sway', 'BB_postural_sway', 'MML_postural_sway', 'FP_postural_sway']].isna()).sum(axis=1)/4
        df['NS_postural_sway_conf'] = (~df[['DIGC_postural_sway', 'DIGRRL_postural_sway', 'FB_postural_sway', 'MP_postural_sway']].isna()).sum(axis=1)/4
        df['S_postural_sway_derivative_conf'] = (~df[['ST_postural_sway_derivative', 'BB_postural_sway_derivative', 'MML_postural_sway_derivative', 'FP_postural_sway_derivative']].isna()).sum(axis=1)/4
        df['NS_postural_sway_derivative_conf'] = (~df[['DIGC_postural_sway_derivative', 'DIGRRL_postural_sway_derivative', 'FB_postural_sway_derivative', 'MP_postural_sway_derivative']].isna()).sum(axis=1)/4
        df['gaze_silhouette_score_conf'] = (~df[['BB_gaze_silhouette_score','S_gaze_silhouette_score']].isna()).sum(axis=1)/2
        df['mean_gaze_percent_right_conf'] = (~df[['S_gaze_percent_right','BB_gaze_percent_right']].isna()).sum(axis=1)/2
        df['FP_gaze_speech_correlation_conf'] = (~df[['FP_gaze_speech_correlation']].isna()).sum(axis=1)


        df['S_facing_forward_conf'] = (~df[['ST_facing_forward', 'BB_facing_forward', 'MML_facing_forward', 'FP_facing_forward']].isna()).sum(axis=1)/4
        df['NS_facing_forward_conf'] = (~df[['DIGC_facing_forward', 'DIGRRL_facing_forward', 'FB_facing_forward', 'MP_facing_forward']].isna()).sum(axis=1)/4

        df['S_eyebrows_complexity_conf'] = (~df[['ST_eyebrows_complexity', 'BB_eyebrows_complexity', 'MML_eyebrows_complexity', 'FP_eyebrows_complexity']].isna()).sum(axis=1)/4
        df['NS_eyebrows_complexity_conf'] = (~df[['DIGC_eyebrows_complexity', 'DIGRRL_eyebrows_complexity', 'FB_eyebrows_complexity', 'MP_eyebrows_complexity']].isna()).sum(axis=1)/4

        df['S_mouth_complexity_conf'] = (~df[['ST_mouth_complexity', 'BB_mouth_complexity', 'MML_mouth_complexity', 'FP_mouth_complexity']].isna()).sum(axis=1)/4
        df['NS_mouth_complexity_conf'] = (~df[['DIGC_mouth_complexity', 'DIGRRL_mouth_complexity', 'FB_mouth_complexity', 'MP_mouth_complexity']].isna()).sum(axis=1)/4

        df['S_postural_sway_complexity_conf'] = (~df[['ST_head_movement_complexity', 'BB_head_movement_complexity', 'MML_head_movement_complexity', 'FP_head_movement_complexity']].isna()).sum(axis=1)/4
        df['NS_postural_sway_complexity_conf'] = (~df[['DIGC_head_movement_complexity', 'DIGRRL_head_movement_complexity', 'FB_head_movement_complexity', 'MP_head_movement_complexity']].isna()).sum(axis=1)/4
        
        df['average_response_to_name_delay_conf'] = df['valid_name_calls'].apply(lambda x: np.sum(x))/3
        df['proportion_of_name_call_responses_conf'] = df['valid_name_calls'].apply(lambda x: np.sum(x))/3

        for f in TOUCH_VARIABLES:
            df['{}_conf'.format(f)] = df['number_of_touches'].apply(lambda x: 0 if np.isnan(x) else x/15  if x <=15 else 1. if x>= 16 else 0)
            
        df['RTN_conf'] = df['valid_name_calls'].apply(lambda x: np.sum(x))/3
        df['touch_conf'] = df['number_of_touches'].apply(lambda x: 0 if np.isnan(x) else x/15  if x <=15 else 1. if x>= 16 else 0)

        return df

    def _plot_missing(self):
        if self.df.isnull().sum().sum() != 0:
            na_df = (self.df.isnull().sum() / len(self.df)) * 100      
            na_df = na_df.drop(na_df[na_df == 0].index).sort_values(ascending=False)
            missing_data = pd.DataFrame({'Missing Ratio %' :na_df})
            missing_data.plot(kind = "bar", figsize=(30, 8))
            plt.title("Percentage of values missing (the higher the more missing)", weight='bold', fontsize=18)
            plt.show()
        else:
            
            print('No NAs found')
            
    def _init_features_name(self, features_name):
        """
            This function initialize the features name, so that it is composed of:
                1) The name of the raw features used 
                2) The possible missing indicator variables.
                    `self.use_indicator_variable` can be a True or False, 
                    True: double the number of features by adding all missing variables
                    False: None are used
                    Dict: {high-level name of the missingness: [feat_1, feat2]}
        """
        
        self.raw_features_name = deepcopy(features_name)
                
        if isinstance(self.use_missing_indicator_variables, dict):
                            
            for feature_name_grouped, feats in self.use_missing_indicator_variables.items():
                features_name.append("Z_{}".format(feature_name_grouped))
                
            return features_name
            

        elif self.use_missing_indicator_variables==True:

            return features_name + ['Z_' + feat for feat in features_name]

        elif self.use_missing_indicator_variables=='redundant':
    
            assert len(features_name)==2, "Only for toy dataset."

            return ['X_1', 'X_2', 'X_1_bis', 'X_2_bis']

        else:

            return features_name
        
    def _match_age(self, other_group=0, target_group=1):
        group_1_age = sorted(self.df.query(" `diagnosis` == @other_group").age.tolist())
        group_2_age = self.df.query("`diagnosis` == @target_group").age.tolist()

        if np.mean(group_2_age) < np.mean(group_1_age): # typically ASD older than TD; we need to remove young TD
            to_drop = 'older'
        else:
            to_drop = 'younger'

        drop = 0
        _, p = mannwhitneyu(group_2_age, group_1_age)
        while p < 0.05:
            drop += 1
            if to_drop=='younger':
                u, p = mannwhitneyu(group_1_age[drop:], group_2_age)
            elif to_drop=='older':
                u, p = mannwhitneyu(group_1_age[:-drop], group_2_age)

        drop_indices = self.df.query("`diagnosis` == @other_group").nsmallest(drop, 'age').index

        return drop_indices
    
    def _add_Z_variables(self, df):
        
        for feat in DEFAULT_PREDICTORS:
            df['Z_{}'.format(feat)] = np.isnan(df[feat])
        
        return df 
    
    def _init_scenario(self, scenario):
        
        self.scenario = scenario
    
        if scenario == 'multimodal_2023_regular':
            
            self.filter(administration={'studies':  ['ARC', 'P1', 'P2', 'P3'],
                                        'order': 'first',
                                        'completed': True}, 
                           demographics={'age':[17, 37]},
                            clinical={'diagnosis': [0, 1]},
                            verbose=True)
            
        elif scenario == 'multimodal_2023_regular_ddld':
            self.filter(administration={'studies':  ['ARC', 'P1', 'P2', 'P3'],
                                        'order': 'first',
                                        'completed': True}, 
                           demographics={'age':[17, 37]},
                            clinical={'diagnosis': [0, 1, 2]},
                            verbose=True)  
            
        elif scenario == 'multimodal_2023_regular_asd_ddld':
            self.filter(administration={'studies':  ['ARC', 'P1', 'P2', 'P3'],
                                        'order': 'first',
                                        'completed': True}, 
                           demographics={'age':[17, 37]},
                            clinical={'diagnosis': [1, 2]},
                            verbose=True)  
            
            
            
        elif scenario == 'multimodal_2023_regular_ddld_only':
            self.filter(administration={'studies':  ['ARC', 'P1', 'P2', 'P3'],
                                        'order': 'first',
                                        'completed': True}, 
                           demographics={'age':[17, 37]},
                            clinical={'diagnosis': [0, 2]},
                            verbose=True) 
            
        elif scenario == 'multimodal_2023_nt_ddld':
            self.filter(administration={'studies':  ['ARC', 'P1', 'P2', 'P3'],
                                        'order': 'first',
                                        'completed': True}, 
                           demographics={'age':[17, 37]},
                            clinical={'diagnosis': [0, 2]},
                            verbose=True)     
        elif scenario == 'multimodal_2023_extended':
            
            self.filter(administration={'studies':  ['ARC', 'P1', 'P2', 'P3'],
                                        'order': 'first',
                                        'completed': True}, 
                           demographics={'age':[17, 50]},
                            clinical={'diagnosis': [0, 1]},
                            verbose=True)
            
        elif scenario == 'multimodal_2023_extended_ddld':
            self.filter(administration={'studies':  ['ARC', 'P1', 'P2', 'P3'],
                                        'order': 'first',
                                        'completed': True}, 
                           demographics={'age':[17, 50]},
                            clinical={'diagnosis': [0, 1, 2]},
                            verbose=True)      
            
            
        elif scenario == 'multimodal_2023_all':
            
            self.filter(administration={'studies':  ['ARC', 'P1', 'P2', 'P3', 'SenseToKnowStudy'],
                                        'order': 'first',
                                        'completed': True}, 
                           demographics={'age':[17, 50]},
                            clinical={'diagnosis': [0, 1]},
                            verbose=True)
            

        elif scenario == 'asd_td_age_matched_n_balanced':
            
            self.filter(administration={'order': 'first', 
                                             'complete': True}, 
                            clinical={'diagnosis': [0, 1]}, 
                            demographics={'age':[10, 60]}, 
                            matching={'age':[0, 1]}, verbose=True)


        elif scenario == 'asd_td_age_matched_n_unbalanced':

            self.filter(administration={'order': 'first', 
                                     'complete': True}, 
                    clinical={'diagnosis': [0, 1]}, 
                    demographics={'age':[10, 36]}, 
                    matching={'age':[0, 1]}, verbose=True)

        elif scenario=='papers':

            self.filter(administration={'studies':  ['ARC', 'P1', 'P2', 'P3'],
                                        'order': 'first',
                                        'completed': True}, 
                           demographics={'age':[16, 46]},
                            clinical={'diagnosis': [0, 1]},
                            verbose=True)

        elif scenario=='papers_matched':
    
            self.filter(administration={'studies': ['ARC', 'P1'],
                            'order': 'first', 
                             'complete': True}, 
                            clinical={'diagnosis': [0, 1]}, 
                            matching={'age':[0, 1]}, verbose=True)
        elif scenario=='young_17_39':
            self.filter(administration={'studies': ['ARC', 'P1','SenseToKnowStudy'],
                            'order': 'first', 
                             'complete': True}, 
                            clinical={'diagnosis': [0, 1]}, 
                            demographics = {'age': [17, 39]}, 
                             verbose=True)  

        elif scenario=='papers_remote':
            self.filter(administration={'studies': ['SenseToKnowStudy'],
                            'order': 'first', 
                             'complete': True}, 
                            clinical={'diagnosis': [0, 1]}, 
                            demographics = {'age': [17, 39]}, 
                             verbose=True)  
        elif scenario=='all':
            self.filter(administration={
                            'order': 'first', 
                             'complete': True}, 
                            clinical={'diagnosis': [0, 1]},
                             verbose=True)      

              
        elif scenario is None:
            pass
        
        else:
            raise ValueError("Please use one of the following scenario: asd_td_age_matched_n_balanced or asd_td_age_matched_n_unbalanced or None")
            
        return scenario

