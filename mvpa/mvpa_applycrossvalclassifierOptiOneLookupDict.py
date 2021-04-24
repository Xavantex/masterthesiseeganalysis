from scipy.stats import zscore
from sklearn.metrics import confusion_matrix as conm
import numpy as np


def mvpa_applycrossvalclassifier(cfg:dict, model:dict, data:dict):

    '''
    cfg.fold = 10;
    cfg.classifier = 20; % How many classifiers? ex.3
    cfg.classifiernumber = 20; % Which classifier? [4 8 9]
    cfg.timebins = 115; % How many time bins?
    cfg.timebinsnumber = 3; % Which timebins? [4 8 9]
    cfg.category_predict = {'Face' 'Landmark' 'Object'};
    cfg.category_model = {'Face' 'Landmark' 'Object'};
    cfg.trials = 'selection';
    cfg.trialselection = 'trialinfo(:,3) == 1';

    selected the trails in the data that are going to be preditectd!!!!!!

    select observations based on cfg.trailinfo
    '''

    if cfg['trials'] == 'selection':#Check if trialselection exists for non selection
        indiceObs = data[cfg['trialselection']]
        cat_name = data['category_name'][indiceObs]
    else:
        cat_name = data['category_name']
        indiceObs = np.ones(len(cat_name))
    
    cat_name = cat_name[np.in1d(cat_name, cfg['category_predict'])]# Might be able to speed up? with already existing np function

    #Observation Indices for feature selection!
    indiceObs2 = np.in1d(data['category_name'], cfg['category_predict'])
    

    indiceObs = np.logical_and(indiceObs, indiceObs2)

    rc = len(cfg['category_model'])#num cats in the model!
    
    #Store the indices of the data['feature_name'] as dict, the lookup should be faster than to loop over in1d
    fLookup = dict((item, index) for index, item in enumerate(data['feature_name']))

    #Vectorize the lookup function to be able to retrieve multiple items at the same time, increasing time more
    arrLookup = np.vectorize(fLookup.__getitem__)

    def timebinFeatures(feature_tb):

        feature_tb = zscore(feature_tb[indiceObs])

        def classifier(classy):

            folds = classy['fold']

            confmatfinal = np.zeros((rc,rc))

            def folding(fold):

                #If we're gonna change the value of something out of scope we need to specify it by using nonlocal, this is however usually best to avoid. Even more so with the Global keyword.
                nonlocal confmatfinal

                if fold:
                    #extract feature for that cassifier and fold
                    #Should be faster to lookup the indices in this manner
                    indiceFeature = arrLookup(fold['feature'])
                    feature = feature_tb[:, indiceFeature]
                    #precit!!!!
                    label = fold['model'].predict(feature)
                    #confusion matrix
                    score = np.mean(label == cat_name)#Already predicted the labels, could just manually calculate the mean as we do here.

                    confmat = conm(y_true = cat_name, y_pred = label, labels = np.unique(cat_name))
                    
                    #If we use the fact that confusion matrices have equal dimension length then the total size must be larger as well if one dimension is
                    #and again np.size is faster than len, then we have the same results
                    if np.size(confmat) < np.size(confmatfinal):
                        id = np.flatnonzero(np.in1d(cfg['category_model'], label).T)
                        rw = np.flatnonzero(confmat)[0]
                        confmat_new = np.zeros((rc,rc))
                        
                        #vectorization of numpy arrays are faster if you can afford the memory because it's done in c instead of doing a slow python for loop
                        # the equivalent of the vectorization is shown below
                        confmat_new[rw, id[:]] = confmat[rw, :np.size(id)]
                        #Above is equivalent to
                        #for index, item in enumerate(id):
                        #    confmat_new[rw, item] = confmat[rw, index]

                        confmat = confmat_new

                    confmatfinal += confmat

                    return {'predict': label,
                            'truelabel': cat_name,
                            'feature': fold['feature'],
                            'likelihood': score,
                            'confmat': confmat}

                elif not fold:
                    pass#Return None if fold is empty
                else:
                    print('unkown error, model folds are not empty but couldnt execute')
                    print(fold)
                    exit()

            
            return {'fold': np.array(list(map(folding, folds)), dtype = dict)}, confmatfinal/10

        #zip repacks the returns: (1,2),(1,2),(1,2) becoms (1,1,1),(2,2,2) here
        predictedClass, predictedConfmatfinal = zip(*map(classifier, model['classifier']))

        return {'classifier': np.array(predictedClass, dtype = dict),
                'confmatfinal': np.array(predictedConfmatfinal)}#Usually best to let numpy decide datatype to decrease memory usage unless necessary with certain precision or certain datatype.


    return {'timebin': np.array(list(map(timebinFeatures, data['feature'])), dtype = dict)}


if __name__ == '__main__':
    '''
    Placeholder main function for fast profiling tests, or to see if working.
    Tests should however mainly be carried out using unittests. Please only use for executing a specific file or for profiling.
    '''
    import pickle
    import scipy.io as sio
    import time

    cdata = sio.loadmat(f'./Visual/Subj01/6-ClassificationData/Subj01_test_data_visual',
                        chars_as_strings = True, simplify_cells = True)[f'Subj01_test_data_visual']
    
    file = open('./data/Visual/Subj01/7-ClassifierTraining/Subj01_study_crossvalclass.pkl', 'rb')
    crossdata = pickle.load(file)[f'Subj01_study_crossvalclass']#give classifier
    file.close()

    cfg = {'fold': 10,
            'classifiernumber': 20,
            'timebinsnumber': 20,
            'category_predict': np.array(['Face', 'Landmark', 'Object']),
            'trials': 'all',
            'category_model': np.array(['Face', 'Landmark', 'Object'])}
    t = time.time()
    predtest_visual = mvpa_applycrossvalclassifier(cfg = cfg, model = crossdata, data = cdata)
    print(time.time()-t)
    print(f'Subj01\nDone')
    
    with open('test.pkl', 'wb') as file:
        pickle.dump({f'Subj01_study_predtest_visual': predtest_visual},file)