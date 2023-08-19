import os
import numpy as np

REFERENCE_IMBALANCE_RATIO = 0.0208#1/44
DEFAULT_POSITIVITY_THRESHOLD = 'Youden'
BLINK_DATA_PATH = 'HIDDEN'

S2K_STUDIES =  ['ARC','P1','P2','P3','IMPACT','SAESDM','SenseToKnowStudy','P1R','S2KP','P3R']


feature_name_mapping = {'S_facing_forward':"Facing Forward during Social movies", 
                        'NS_facing_forward':"Facing Forward during Non-social movies", 
                        'S_postural_sway': "Head Movements during Social movies", 
                        'NS_postural_sway': "Head Movements during Non-social movies", 
                        'S_postural_sway_complexity':  "Head Movements Complexity during Social movies", 
                        'NS_postural_sway_complexity':  "Head Movements Complexity during Non-social movies", 
                        
                        'mean_gaze_percent_right': "Gaze Percent Social",
                        'FP_gaze_speech_correlation': "Attention to Speech",
                        'gaze_silhouette_score': 'Gaze Silhouette Score',
                        'S_blink_rate': "Blink Rate during Social", 
                        'NS_blink_rate': "Blink Rate during Non-social", 

                        'average_response_to_name_delay': 'Response to Name Delay [s]',
                        'proportion_of_name_call_responses': 'Response to Name Proportion',
                        'S_eyebrows_complexity': 'Eyebrows Complexity during Social movies', 
                        'NS_eyebrows_complexity': 'Eyebrows Complexity during Non-social movies', 
                        'S_mouth_complexity': 'Mouth Complexity during Social movies', 
                        'NS_mouth_complexity': 'Mouth Complexity during Non-social movies', 
                        'pop_rate': 'Pop The Bubbles Popping Rate', 
                        'std_error': 'Pop The Bubbles Accuracy Variations', 
                        'average_force_applied':  'Pop The Bubbles Average Force Applied', 
                        'average_length':  'Pop The Bubbles Average Touch Length', 
                        'S_postural_sway_derivative': 'Head Movements Acceleration during Social movies', 
                        'NS_postural_sway_derivative': 'Head Movements Acceleration during Non-social movies'
                        }

CLINICAL_COLUMNS = [# DIAGNOSIS RELATED
                    'diagnosis',
                    # MULLEN RELATED
                    'mullen_el','mullen_fm','mullen_rl','mullen_vr','mullen_elc_std',
                    # ADOS RELATED
                    'ados_total','ados_rrb','ados_sa',
                    # SRS RELATED
                    'srs_total_tscore','srs_social_awareness_tscore','srs_social_motivation_tscore',
                    #CBCL RELATED
                    'cbcl_scaleIV_score','cbcl_asd_score',
                    # MCHAT RELATED
                    'mchat_total','mchat_final','mchat_result']

xgboost_hyperparameters = {'scale_pos_weight':True,#np.sum(data.y_train==0)/np.sum(data.y_train==1), 
                            'max_depth' : 3,
                            'learning_rate' : 0.15, 
                            'gamma': 0.1,
                            'n_estimators': 100,
                            'min_child_weight': 1,
                            'reg_lambda': 0.1}  

DEMOGRAPHIC_COLUMNS = ['age', 
                       'sex',
                     'ethnicity',
                     'race',
                     'primary_education']



APP_COLUMNS = ['id', 
                'language', 
                'app_version', 
                'features_extracted', 
                'face_tracking', 
                'date', 
                'path']

CVA_COLUMNS_old = [# GAZE RELATED
                'BB_gaze_percent_right',
                 'BB_gaze_silhouette_score',
                 'S_gaze_percent_right',
                 'S_gaze_silhouette_score',
                 'FP_gaze_speech_correlation',
                 'FP_gaze_silhouette_score',
                 'inv_S_gaze_percent_right',#aggregated
                 'mean_gaze_percent_right', #aggregated
                 'gaze_silhouette_score', #aggregated
   
                # NAME CALL RELATED
                 'proportion_of_name_call_responses',
                 'average_response_to_name_delay',
                  #'name_call_response_binary',

                # POSTURAL SWAY RELATED
                 'FB_postural_sway',
                 'FB_postural_sway_derivative',
                 'DIGC_postural_sway',
                 'DIGC_postural_sway_derivative',
                 'DIGRRL_postural_sway',
                 'DIGRRL_postural_sway_derivative',
                 'ST_postural_sway',
                 'ST_postural_sway_derivative',
                 'MP_postural_sway',
                 'MP_postural_sway_derivative',
                 'PB_postural_sway',
                 'PB_postural_sway_derivative',
                 'BB_postural_sway',
                 'BB_postural_sway_derivative',
                 'RT_postural_sway',
                 'RT_postural_sway_derivative',
                 'MML_postural_sway',
                 'MML_postural_sway_derivative',
                 'PWB_postural_sway',
                 'PWB_postural_sway_derivative',
                 'FP_postural_sway',
                 'FP_postural_sway_derivative',
                 'S_postural_sway',  #aggregated
                 'NS_postural_sway',  #aggregated
                 'S_postural_sway_derivative', #aggregated
                 'NS_postural_sway_derivative', #aggregated
                
                # TOUCH RELATED
                 'number_of_touches',
                 'average_length',
                 'std_length',
                 'average_error',
                 'std_error',
                 'number_of_target',
                 'pop_rate',
                 'average_touch_duration',
                 'std_touch_duration',
                 'average_delay_to_pop',
                 'std_delay_to_pop',
                 'average_force_applied',
                 'std_force_applied',
                 'average_accuracy_variation',
                 'accuracy_consistency',
                 'average_touches_per_target',
                 'std_touches_per_target',
                 'average_time_spent',
                 'std_time_spent',
                 'exploratory_percentage']

DEFAULT_PREDICTORS = [# GAZE RELATED
                 'mean_gaze_percent_right', #aggregated
                 'gaze_silhouette_score', #aggregated
                 'FP_gaze_speech_correlation',
    
                # NAME CALL RELATED
                 #'proportion_of_name_call_responses',
                 'average_response_to_name_delay',

    
                # POSTURAL SWAY RELATED
                 'S_postural_sway',  #aggregated
                 'NS_postural_sway',  #aggregated
                 'S_postural_sway_derivative',
                 'NS_postural_sway_derivative',
                 
                 'S_postural_sway_complexity', 
                 'NS_postural_sway_complexity',
                 
                 # Blink rate
                 'S_blink_rate', 
                 'NS_blink_rate',
                 
                 # Facing Forward
                 'S_facing_forward', 
                 'NS_facing_forward', 
                 
                 # Facial complexity
                 'S_eyebrows_complexity', 
                 'NS_eyebrows_complexity', 
                 'S_mouth_complexity', 
                 'NS_mouth_complexity',
                 
                # TOUCH RELATED
                'pop_rate', 'std_error', 'average_length', 'average_force_applied']#average_error
                #'std_error','number_of_touches','number_of_target','average_error']



        
                #'std_force_applied','average_delay_to_pop','std_length','number_of_target', 'average_error']

#                 'average_length',
#                 'std_length',
#                 'average_error',
#                 'pop_rate',
#                 'average_delay_to_pop',
#                 'average_time_spent']



USE_MISSING_INDICATOR_PREDICTORS = {'PlayingWithBlocks': ['PWB_postural_sway', 'PWB_postural_sway_derivative'],
                                     'FunAtThePark': ['FP_postural_sway',
                                      'FP_postural_sway_derivative',
                                      'FP_gaze_speech_correlation',
                                      'FP_gaze_silhouette_score'],
                                     'BlowingBubbles': ['BB_gaze_percent_right',
                                      'BB_gaze_percent_right',
                                      'BB_postural_sway',
                                      'BB_postural_sway_derivative'],
                                     'RhymesAndToys': ['RT_postural_sway', 'RT_postural_sway_derivative']
                                  }


DEFAULT_PREDICTORS_BY_TYPES = {'Gaze':['mean_gaze_percent_right', 'gaze_silhouette_score', 'FP_gaze_speech_correlation'],
                               'RTN':['proportion_of_name_call_responses', 'average_response_to_name_delay'],
                               'PosturalSway':['S_postural_sway', 'NS_postural_sway', 'S_postural_sway_derivative', 'NS_postural_sway_derivative',
                                'S_postural_sway_complexity', 'NS_postural_sway_complexity'],
                               'FacialComplexity': ['S_eyebrows_complexity', 'NS_eyebrows_complexity',
                                                    'S_mouth_complexity', 'NS_mouth_complexity'],
                               'BlinkRate': ['S_blink_rate', 'NS_blink_rate'],
                               'FacingForward': ['S_facing_forward', 'NS_facing_forward'],
                               'Touch': ['pop_rate', 'std_error', 'average_length', 'average_force_applied'],
                               'All': DEFAULT_PREDICTORS,
                               'All - Gaze': ['proportion_of_name_call_responses','average_response_to_name_delay', 
                                              'S_postural_sway','NS_postural_sway', 'S_postural_sway_derivative', 'NS_postural_sway_derivative',
                                              'S_blink_rate', 'NS_blink_rate',
                                              'S_postural_sway_complexity', 'NS_postural_sway_complexity',
                                              'S_eyebrows_complexity', 'NS_eyebrows_complexity',
                                                'S_mouth_complexity', 'NS_mouth_complexity',
                                                'S_facing_forward', 'NS_facing_forward',
                                               'pop_rate', 'std_error', 'average_length', 'average_force_applied'],
                               'All - RTN': ['mean_gaze_percent_right', 'gaze_silhouette_score',  'FP_gaze_speech_correlation',
                                              'S_postural_sway','NS_postural_sway', 'S_postural_sway_derivative', 'NS_postural_sway_derivative',
                                              'S_blink_rate', 'NS_blink_rate',
                                              'S_postural_sway_complexity', 'NS_postural_sway_complexity',
                                            'S_eyebrows_complexity', 'NS_eyebrows_complexity',
                                              'S_mouth_complexity', 'NS_mouth_complexity',
                                              'S_facing_forward', 'NS_facing_forward',
                                               'pop_rate', 'std_error', 'average_length', 'average_force_applied'],
                               'All - PosturalSway': ['mean_gaze_percent_right', 'gaze_silhouette_score', 'FP_gaze_speech_correlation','S_blink_rate', 'NS_blink_rate',
                                              'proportion_of_name_call_responses','average_response_to_name_delay',
                                               'pop_rate', 'std_error', 'average_length', 'average_force_applied',
                                              'S_eyebrows_complexity', 'NS_eyebrows_complexity',
                                                'S_mouth_complexity', 'NS_mouth_complexity',
                                                'S_facing_forward', 'NS_facing_forward'],
                               'All - Touch': ['mean_gaze_percent_right', 'gaze_silhouette_score',  'FP_gaze_speech_correlation',
                                              'proportion_of_name_call_responses','average_response_to_name_delay',
                                              'S_postural_sway','NS_postural_sway', 'S_postural_sway_derivative', 'NS_postural_sway_derivative','S_postural_sway_complexity', 'NS_postural_sway_complexity','S_blink_rate', 'NS_blink_rate',
                                              'S_eyebrows_complexity', 'NS_eyebrows_complexity',
                                                'S_mouth_complexity', 'NS_mouth_complexity',
                                                'S_facing_forward', 'NS_facing_forward'],
                               'All - FacialComplexity':['mean_gaze_percent_right', 'gaze_silhouette_score',  'FP_gaze_speech_correlation','S_blink_rate', 'NS_blink_rate',
                                              'S_postural_sway','NS_postural_sway', 'S_postural_sway_derivative', 'NS_postural_sway_derivative',
                                              'proportion_of_name_call_responses', 'average_response_to_name_delay',
                                              'S_postural_sway_complexity', 'NS_postural_sway_complexity',
                                              'S_facing_forward', 'NS_facing_forward',
                                               'std_error','number_of_touches','number_of_target','average_error'],
                               'All - FacingForward':['mean_gaze_percent_right', 'gaze_silhouette_score', 'FP_gaze_speech_correlation','S_blink_rate', 'NS_blink_rate',
                                              'S_postural_sway','NS_postural_sway', 'S_postural_sway_derivative', 'NS_postural_sway_derivative',
                                              'proportion_of_name_call_responses', 'average_response_to_name_delay',
                                              'S_postural_sway_complexity', 'NS_postural_sway_complexity',
                                            'S_eyebrows_complexity', 'NS_eyebrows_complexity',
                                              'S_mouth_complexity', 'NS_mouth_complexity',
                                               'std_error','number_of_touches','number_of_target','average_error'],
                               'All - BlinkRate': ['mchat_final', 'mean_gaze_percent_right', 
                                                  'FP_gaze_speech_correlation',
                                                  'gaze_silhouette_score',
                                                  'proportion_of_name_call_responses',
                                                  'average_response_to_name_delay',
                                                  'S_postural_sway',
                                                  'NS_postural_sway',
                                                  'S_postural_sway_derivative',
                                                  'NS_postural_sway_derivative',
                                                  'S_postural_sway_complexity',
                                                  'NS_postural_sway_complexity',
                                                  'S_facing_forward',
                                                  'NS_facing_forward',
                                                  'S_eyebrows_complexity',
                                                  'NS_eyebrows_complexity',
                                                  'S_mouth_complexity',
                                                  'NS_mouth_complexity',
                                                  'pop_rate', 'std_error', 'average_length', 'average_force_applied'], 
                               'All with MCHAT': ['mchat_final', 'mean_gaze_percent_right', 
                                                  'FP_gaze_speech_correlation',
                                                  'gaze_silhouette_score',
                                                  'proportion_of_name_call_responses',
                                                  'average_response_to_name_delay',
                                                  'S_postural_sway',
                                                  'NS_postural_sway',
                                                  'S_postural_sway_derivative',
                                                  'NS_postural_sway_derivative',
                                                  'S_postural_sway_complexity',
                                                  'NS_postural_sway_complexity',
                                                  'S_blink_rate', 'NS_blink_rate',
                                                  'S_facing_forward',
                                                  'NS_facing_forward',
                                                  'S_eyebrows_complexity',
                                                  'NS_eyebrows_complexity',
                                                  'S_mouth_complexity',
                                                  'NS_mouth_complexity',
                                                  'pop_rate', 'std_error', 'average_length', 'average_force_applied']
                              }
                                    
USE_MISSING_INDICATOR_PREDICTORS_BY_TYPES = {'Gaze':{'FunAtThePark': ['FP_gaze_speech_correlation','FP_gaze_silhouette_score'],
                                                     'BlowingBubbles': ['BB_gaze_percent_right', 'BB_gaze_percent_right']},
                                             
                                             'RTN':False,
                                             'PosturalSway':{'PlayingWithBlocks': ['PWB_postural_sway', 'PWB_postural_sway_derivative'],
                                                             'FunAtThePark': ['FP_postural_sway','FP_postural_sway_derivative'],
                                                     'BlowingBubbles': ['BB_postural_sway','BB_postural_sway_derivative'],
                                                     'RhymesAndToys': ['RT_postural_sway', 'RT_postural_sway_derivative']},
                                             
                                            'Touch':False,
                                             'All': USE_MISSING_INDICATOR_PREDICTORS,
                                             'All - Gaze': {'PlayingWithBlocks': ['PWB_postural_sway', 'PWB_postural_sway_derivative'],
                                                             'FunAtThePark': ['FP_postural_sway',
                                                              'FP_postural_sway_derivative'],
                                                             'BlowingBubbles': ['BB_postural_sway', 'BB_postural_sway_derivative'],
                                                             'RhymesAndToys': ['RT_postural_sway', 'RT_postural_sway_derivative']
                                                          },
                                             'All - RTN': {'PlayingWithBlocks': ['PWB_postural_sway', 'PWB_postural_sway_derivative'],
                                                             'FunAtThePark': ['FP_postural_sway',
                                                              'FP_postural_sway_derivative',
                                                              'FP_gaze_speech_correlation',
                                                              'FP_gaze_silhouette_score'],
                                                             'BlowingBubbles': ['BB_gaze_percent_right',
                                                              'BB_gaze_percent_right',
                                                              'BB_postural_sway',
                                                              'BB_postural_sway_derivative'],
                                                             'RhymesAndToys': ['RT_postural_sway', 'RT_postural_sway_derivative']
                                                          },
                                             'All - PosturalSway': {'FunAtThePark': ['FP_gaze_speech_correlation', 'FP_gaze_silhouette_score'],
                                                             'BlowingBubbles': ['BB_gaze_percent_right', 'BB_gaze_percent_right'],
                                                                  },
                                             'All - Touch': {'PlayingWithBlocks': ['PWB_postural_sway', 'PWB_postural_sway_derivative'],
                                                             'FunAtThePark': ['FP_postural_sway',
                                                              'FP_postural_sway_derivative',
                                                              'FP_gaze_speech_correlation',
                                                              'FP_gaze_silhouette_score'],
                                                             'BlowingBubbles': ['BB_gaze_percent_right',
                                                              'BB_gaze_percent_right',
                                                              'BB_postural_sway',
                                                              'BB_postural_sway_derivative'],
                                                             'RhymesAndToys': ['RT_postural_sway', 'RT_postural_sway_derivative']
                                                          }                                             
                                             
                                            }


grouped_missing_features = {'Gaze': ['mean_gaze_percent_right', 'gaze_silhouette_score', 'FP_gaze_speech_correlation'],
                            'Social' : ['S_postural_sway', 'S_postural_sway_derivative'],
                            'Non Social' : ['NS_postural_sway', 'NS_postural_sway_derivative'],
                             #'gaze_silhouette_score': ['gaze_silhouette_score'],
                             #'proportion_of_name_call_responses': ['proportion_of_name_call_responses'],
                             'average_response_to_name_delay': ['average_response_to_name_delay'],
                             #'S_postural_sway': ['S_postural_sway'],
                             #'NS_postural_sway': ['NS_postural_sway'],
                             #'S_postural_sway': ['S_postural_sway'],
                             #'S_postural_sway_derivative': ['S_postural_sway_derivative'],
                             #'NS_postural_sway_derivative': ['NS_postural_sway_derivative'],
                             'Game': ['average_length','std_length','average_error', 'pop_rate','average_delay_to_pop','average_time_spent']}

MINIMAL_SET_OF_FEATURES = ['BB_gaze_percent_right',
                           'FP_gaze_speech_correlation',
                            'BB_gaze_silhouette_score',
                            'S_gaze_percent_right',
                            'S_gaze_silhouette_score',
                            'FP_gaze_speech_correlation',
                            'FP_gaze_silhouette_score',
                            'proportion_of_name_call_responses',
                            'average_response_to_name_delay',
                            'FB_postural_sway',
                            'FB_postural_sway_derivative',
                            'DIGC_postural_sway',
                            'DIGC_postural_sway_derivative',
                            'DIGRRL_postural_sway',
                            'DIGRRL_postural_sway_derivative',
                            'ST_postural_sway',
                            'ST_postural_sway_derivative',
                            'MP_postural_sway',
                            'MP_postural_sway_derivative',
                            'PB_postural_sway',
                            'PB_postural_sway_derivative',
                            'BB_postural_sway',
                            'BB_postural_sway_derivative',
                            'RT_postural_sway',
                            'RT_postural_sway_derivative',
                            'MML_postural_sway',
                            'MML_postural_sway_derivative',
                            'PWB_postural_sway',
                            'PWB_postural_sway_derivative',
                            'FP_postural_sway',
                            'FP_postural_sway_derivative',
                            'number_of_touches',
                            'average_length',
                            'std_length',
                            'average_error',
                            'std_error',
                            'number_of_target',
                            'pop_rate',
                            'average_touch_duration',
                            'std_touch_duration',
                            'average_delay_to_pop',
                            'std_delay_to_pop',
                            'average_force_applied',
                            'std_force_applied',
                            'average_accuracy_variation',
                            'accuracy_consistency',
                            'average_touches_per_target',
                            'std_touches_per_target',
                            'average_time_spent',
                            'std_time_spent',
                            'exploratory_percentage']

dict_missing_stimulis = {'PopTheBubbles': 'number_of_touches',
                         'FloatingBubbles': ['FB_postural_sway', 'FB_postural_sway_derivative'],
                            'DogInGrassC': ['DIGC_postural_sway', 'DIGC_postural_sway_derivative'],
                            'DogInGrassRRL': ['DIGRRL_postural_sway', 'DIGRRL_postural_sway_derivative'],
                            'SpinningTop': ['ST_postural_sway', 'ST_postural_sway_derivative'],
                            'PlayingWithBlocks': ['PWB_postural_sway', 'PWB_postural_sway_derivative'],
                            'FunAtThePark': ['FP_postural_sway', 'FP_postural_sway_derivative',
                                                'FP_gaze_speech_correlation', 'FP_gaze_silhouette_score'],

                             'MechanicalPuppy': ['MP_postural_sway', 'MP_postural_sway_derivative'],
                             'BlowingBubbles': ['BB_postural_sway', 'BB_postural_sway_derivative',
                                                   'BB_gaze_percent_right', 'BB_gaze_percent_right'],
                            'RhymesAndToys': ['RT_postural_sway', 'RT_postural_sway_derivative'],
                             'MakeMeLaugh': ['MML_postural_sway', 'MML_postural_sway_derivative'],
                             'RTNDelay': ['average_response_to_name_delay']}

dict_missing_stimulis = {'PopTheBubbles': ['number_of_touches'],
                         'FloatingBubbles': ['FB_postural_sway', 'FB_postural_sway_derivative'],
                            'DogInGrassC': ['DIGC_postural_sway', 'DIGC_postural_sway_derivative'],
                            'DogInGrassRRL': ['DIGRRL_postural_sway', 'DIGRRL_postural_sway_derivative'],
                            'SpinningTop': ['ST_postural_sway', 'ST_postural_sway_derivative'],
                            'PlayingWithBlocks': ['PWB_postural_sway', 'PWB_postural_sway_derivative'],
                            'FunAtThePark': ['FP_postural_sway', 'FP_postural_sway_derivative',
                                                'FP_gaze_speech_correlation', 'FP_gaze_silhouette_score'],
                             'MechanicalPuppy': ['MP_postural_sway', 'MP_postural_sway_derivative'],
                             'BlowingBubbles': ['BB_postural_sway', 'BB_postural_sway_derivative',
                                                   'BB_gaze_percent_right', 'BB_gaze_percent_right'],
                            'RhymesAndToys': ['RT_postural_sway', 'RT_postural_sway_derivative'],
                             'MakeMeLaugh': ['MML_postural_sway', 'MML_postural_sway_derivative'],
                             'RTNDelay': ['average_response_to_name_delay']}


TOUCH_VARIABLES  =['number_of_touches',
                  'average_length',
                  'std_length',
                  'average_error',
                  'std_error',
                  'number_of_target',
                  'pop_rate',
                  'average_touch_duration',
                  'std_touch_duration',
                  'average_delay_to_pop',
                  'std_delay_to_pop',
                  'repeat_percentage',
                  'repeat_percentage_naive',
                  'double_tap',
                  'mean_velocity',
                  'std_velocity',
                  'average_force_applied',
                  'std_force_applied',
                  'average_accuracy_variation',
                  'accuracy_consistency',
                  'average_touches_per_target',
                  'std_touches_per_target',
                  'average_time_spent',
                  'std_time_spent',
                  'exploratory_percentage']

VALIDITY_COLUMNS = ['validity_available',
                    'completed', 
                    'StateOfTheChild', 
                    'SiblingsInTheRoom', 
                    'ShotsVaccines', 
                    'Distractions', 
                    'FamilyMemberDistract', 
                    'PetDistract', 
                    'PetNoiseDistract', 
                    'DoorbellPhoneDistract', 
                    'TVOnDistract', 
                    'OtherDistract', 
                    'SittingUp', 
                    'Hungry', 
                    'Diaper', 
                    'AppTeamComment',
                    'Comments']


GROUPED_FEATURES = {0: ['S_gaze_percent_right',
                      'S_gaze_silhouette_score',
                      'proportion_of_name_call_responses',
                        'name_call_response_binary',
                      'FB_postural_sway',
                      'FB_postural_sway_derivative',
                      'DIGC_postural_sway',
                      'DIGC_postural_sway_derivative',
                      'ST_postural_sway',
                      'ST_postural_sway_derivative',
                      'MP_postural_sway',
                      'MP_postural_sway_derivative'],
                    1: ['RT_postural_sway', 'RT_postural_sway_derivative'],
                    2: ['BB_gaze_percent_right',
                      'BB_gaze_silhouette_score',
                      'BB_postural_sway',
                      'BB_postural_sway_derivative'],
                    3: ['MML_postural_sway', 'MML_postural_sway_derivative'],
                    4: ['DIGRRL_postural_sway', 'DIGRRL_postural_sway_derivative'],
                    5: ['FP_gaze_speech_correlation',
                      'FP_gaze_silhouette_score',
                      'FP_postural_sway',
                      'FP_postural_sway_derivative'],
                    6: ['PB_postural_sway', 'PB_postural_sway_derivative'],
                    7: ['PWB_postural_sway', 'PWB_postural_sway_derivative'],
                    8: ['number_of_touches', 'number_of_target', 'exploratory_percentage'],
                    9: ['average_length',
                      'std_length',
                      'average_error',
                      'std_error',
                      'pop_rate',
                      'average_touch_duration',
                      'std_touch_duration',
                      'average_delay_to_pop',
                      'std_delay_to_pop',
                      'average_force_applied',
                      'std_force_applied',
                      'average_accuracy_variation',
                      'accuracy_consistency',
                      'average_touches_per_target',
                      'std_touches_per_target'],
                    10: ['average_time_spent', 'std_time_spent']}