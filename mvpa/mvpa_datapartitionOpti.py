from sklearn.model_selection import StratifiedKFold#, KFold
from scipy.stats import f_oneway
from scipy.stats import zscore
from mvpa.ArrGroupLabel import labeling_groupby_np
import numpy as np


def mvpa_datapartition(cfg: dict, prepdata: dict, random_state = None):

    '''
    This function takes the output from the mvpa_dataprep and partitions the data
    for further cross-validation procedure.
    For each partition fold a feature reduction step is performed:
    The feature Selection Step performs a One-way ANOVA for each feature.
    Only includes feature if p < 0.05
    When a fetuare is included is z-normalized!
    
    [data] = mvpa_datapartiation(k,prepdata)
    
    WHERE:
    PREPDATA is the output from the mvpa_dataprep
    CFG is a structure whit the following definitions:
    
    cfg.fold = number of partitions to be performed in the data.
    
    he OUTPUT will be a structure with the following fields:
    DATA.numclassifier	= is the number of timebins read from the data,
                          corresponds to the number of different classifiers that will be trained.
    DATA.fold			= is the number of partitions.
    DATA.classifier     = is a structure with as many fields as timebins.
                          In each DATA.classifier field there is a structure corresponding
                          to each partition fold with the following information stored:
    
                        selectedfeature_name    = a string vector containing the names of
                                                  the features that will be used for training the model
                        category_training       = a numerical vector contacting the category of each
                                                  observation included in the training data set
                        category_training_name  = a string vector containing the name of each category
                                                  included in the training data set
                        category_test           = a numerical vector contacting the category of each observation
                                                  included in the test data set
                        category_test_name      = a string vector contacting the name of each category
                                                  included in the test data set
                        feature_training        = a matrix (Obs X feature) with the features of each observation
                                                  in the training data set
                        feature_test            = matrix (Obs X feature) with the features of each observation
                                                  in the test data set
     
    ---------------------------------------------------------------------------------------------------
    '''

    category = prepdata['category']
    category_name = prepdata['category_name']
    feature_name = prepdata['feature_name']
    classifiernumber = len(prepdata['feature'])
    

    #for classyNo in range(classifiernumber):
    def classifierLoop(classyNo):

        #for train, test in cp.split(np.zeros(len(category)), category):
        def splitting(train, test):

            foldy = {'category_training': category[train],
                    'category_training_name': category_name[train],
                    'category_test': category[test],
                    'category_test_name': category_name[test],
                    'feature_training': np.array([]),
                    'feature_test': np.array([]),
                    'selectedfeature_name': np.array([[]])}
            feature_training_all = feature[train]
            feature_test_all = feature[test]
            _,size = np.shape(feature_training_all)
            for a in range(size):
                groups = labeling_groupby_np(feature_training_all[:,a], foldy['category_training'])
                _, p = f_oneway(*groups)
                if p < 0.05:
                    try:
                        foldy['feature_training'] = np.append(foldy['feature_training'], [zscore(feature_training_all[:,a])], axis=0)
                        foldy['feature_test'] = np.append(foldy['feature_test'], [zscore(feature_test_all[:,a])], axis=0)
                        foldy['selectedfeature_name'] = np.append(foldy['selectedfeature_name'], [feature_name[a]], axis=0)
                        
                    except:#Don't use np.ndarray or object dtype, immensly smaller and use less memory
                        foldy['feature_training'] = np.array([zscore(feature_training_all[:,a])]) #Center all data with mean 0 and std to 1, selecting one feature over categories
                        foldy['feature_test'] = np.array([zscore(feature_test_all[:,a])])
                        foldy['selectedfeature_name'] = np.array([feature_name[a]])


            foldy['feature_training'] = foldy['feature_training'].T # We want the features as columns when fitting
            foldy['feature_test'] = foldy['feature_test'].T

            return foldy

        #Create crossvalidation iterable object to create the different classifiers
        cp = StratifiedKFold(n_splits = cfg['fold'], shuffle = True, random_state = random_state)
        feature = prepdata['feature'][classyNo]

        #Train and test are the indices of the array.
        splittytemp = [splitting(train = train, test = test) for train, test in cp.split(np.zeros(len(category)), category)]

        #We know the returned objects are gonna be dicts, and want to make sure they are saved as.
        return {'fold': np.array(splittytemp, dtype=dict)}

    return {'classifier': np.array(list(map(classifierLoop, range(classifiernumber))), dtype=dict), 'numclassifier': classifiernumber, 'fold': cfg['fold']}


if __name__ == '__main__':

    '''
    Placeholder main function to tinker, profile and execute single functions.
    Testing can be done here, but should be done with unittests
    '''

    from scipy.io import loadmat, savemat
    import pickle
    import time
    matfile = loadmat('./Visual/Subj01/6-ClassificationData/Subj01_study_data', chars_as_strings = True, simplify_cells = True)['Subj01_study_data']
    cfg = {'classifiernumber': 20, 'fold': 10}
    t = time.time()
    datapart = mvpa_datapartition(cfg, matfile)
    print(time.time()-t)

    with open('dataparttest.pkl', 'wb') as file:
        pickle.dump({'Subj01_study_datapart': datapart}, file)

# ---------------------------------------------------------------------------------------------------
# Erickson copy of Bram�o, I. Octo. 2020
# ---------------------------------------------------------------------------------------------------