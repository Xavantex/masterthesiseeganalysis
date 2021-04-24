from sklearn.model_selection import StratifiedKFold
from scipy.stats import f_oneway
from scipy.stats import zscore
from rayModObjStore.remoteForking import dictAppend
#Function to seperate a list/array to the different groups depending on the mapped array, i.e. input: [1,2,3,4,5,6] with ['a','b','c','a','b','c'] -> output: [1,4], [2,5], [3,6]
from mvpa.ArrGroupLabel import labeling_groupby_np
import numpy as np
import ray

def mvpa_datapartition(cfg: dict, prepdata: dict):

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
    
    The OUTPUT will be a structure with the following fields:
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

    @ray.remote
    def featureLoop(feature, category, category_name, feature_name, foldNo, random_state = None):

        def splitting(train, test):

            foldData = {'category_training': category[train],
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
                groups = labeling_groupby_np(feature_training_all[:,a], foldData['category_training'])
                #Unpack the returned groups into seperate arguments.
                _, p = f_oneway(*groups)
                if p < 0.05:
                    try:
                        foldData['feature_training'] = np.append(foldData['feature_training'], [zscore(feature_training_all[:,a])], axis=0)
                        foldData['feature_test'] = np.append(foldData['feature_test'], [zscore(feature_test_all[:,a])], axis=0)
                        foldData['selectedfeature_name'] = np.append(foldData['selectedfeature_name'], [feature_name[a]], axis=0)
                    #Exception for first appendage, since array will be empty.
                    except:
                        foldData['feature_training'] = np.array([zscore(feature_training_all[:,a])]) #Center all data with mean 0 and std to 1, selecting one feature over categories
                        foldData['feature_test'] = np.array([zscore(feature_test_all[:,a])])
                        foldData['selectedfeature_name'] = np.array([feature_name[a]])


            foldData['feature_training'] = foldData['feature_training'].T # We want the features as columns when fitting
            foldData['feature_test'] = foldData['feature_test'].T

            return foldData

        #Create iterable for doing crossvalidation
        cp = StratifiedKFold(n_splits = foldNo, shuffle = True, random_state = random_state)

        #train and test are the indices of the data/what data to pick. And np.zeros is basically a placeholder, not necessarily the actual data
        tempfold = [splitting(train = train, test = test) for train, test in cp.split(np.zeros(len(category)), category)]

        return {'fold': np.array(tempfold, dtype = dict)}


    category = prepdata['category']
    category_name = prepdata['category_name']
    feature_name = prepdata['feature_name']
    classifiernumber = len(prepdata['feature'])

    #While ray.put is called each time for all variables, it is creating new references each time. by calling ti first it should be only put in the object store once.
    #One can also assign the reference as a variable and use as input.
    ray.put(category)
    ray.put(category_name)
    ray.put(feature_name)
    ray.put(cfg['fold'])

    data = {'numclassifier': classifiernumber, 'fold': cfg['fold']}

    tempdata = [featureLoop.remote(feature = feature, category = category, category_name = category_name, feature_name = feature_name, foldNo = cfg['fold']) for feature in prepdata['feature']]
    
    return dictAppend.remote(data, 'classifier', *tempdata)


if __name__ == '__main__':
    '''
    Placeholder main function, could used to quickly test function or to profile code.
    '''
    import pickle
    from scipy.io import loadmat
    import time
    ray.init(address='auto', _redis_password='5241590000000000')
    matfile = loadmat('./Visual/Subj01/6-ClassificationData/Subj01_study_data', chars_as_strings = True, simplify_cells = True)['Subj01_study_data']
    cfg = {'classifiernumber': 20, 'fold': 10}
    t = time.time()
    datapart = mvpa_datapartition(cfg, matfile)
    datapart = ray.get(datapart)
    print(time.time()-t)
    ray.shutdown()
    with open('dataparttest.pkl', 'wb') as file:
        pickle.dump({'Subj01_study_datapart': datapart}, file)

# ---------------------------------------------------------------------------------------------------
# Erickson copy of Bram�o, I. Octo. 2020
# ---------------------------------------------------------------------------------------------------