from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC as Linsvc
from sklearn.metrics import confusion_matrix as conm
from rayModObjStore.remoteForking import dictFork, dictAppend
import numpy as np
import ray


def mvpa_traincrossvalclassifier(cfg: dict, data: dict, random_state=None):

    '''
    ---------------------------------------------------------------------------------------------------
    
    [model] = mvpa_traincrossvalclassifier(cfg,data)
    
    WHERE:
    DATA is an output from mvpa_datapartition
    cfg.training_algorithm
    training_algorithm = 1 - SVM with three classes, onevsall strategy and linear kernel
    training_algorithm = 2 - SVM with two classes and linear kernel
    cfg.fold = 10;
    fg.classifiernumber = 20;
    Output model is structure with has many fields has classifiers trained
    ---------------------------------------------------------------------------------------------------
    '''

    rc = len(cfg['category_model'])

    def folding(foldData):

        def training():
            #Shouldn't have to use class_weight since it should already be evenly balanced from zscore, unecessary to do twice. Otherwise use argument class_weight = 'balanced'
            ClassifierSVM = Linsvc(loss = 'hinge', tol = 1e-4, multi_class = 'ovr', random_state=random_state, max_iter = 9000)
            parameters = {
                'C': [0.0001, 0.001, 0.01]
            }
            svc = GridSearchCV(estimator = ClassifierSVM, param_grid = parameters, refit = True)
            svc.fit(X = foldData['feature_training'],
                    y = foldData['category_training_name'])

            return svc

        #if classData['fold'][foldNo]['feature_training'].any():#Should it be any or size? Any is if any element is not 0 or false, size is that it's not just an empty array
        
        #if it is any then try is faster than doing lookups all the time,
        #instead just handle it with the except statement if something goes wrong,
        #the overhead from exceptions in THIS case is less than checking .any()
        try:
            svc = training()

            label = svc.predict(foldData['feature_test'])
            score = svc.score(foldData['feature_test'],
                            foldData['category_test_name'])
            confmatrix = conm(y_true = foldData['category_test_name'],
                y_pred = label,
                labels = np.unique(foldData['category_training_name']))#Could switch to 'category_test_name' since we should expect all categories to be in there as well,
                #saving on having to iterate over the larger 'category_training_name' list?


            return {'model': svc, 'predict': label,
                    'truelabel': foldData['category_test_name'],
                    'feature': foldData['selectedfeature_name'],
                    'likelihood': score, 'confmat': confmatrix}, confmatrix


        except ValueError as err:
            if not foldData['feature_training'].any():
                #Return zero array to np.sum won't complain.
                return None, np.zeros((rc,rc))
            else:
                print(err)
                print(foldData['feature_training'])
                print(label)
                print(score)
                print(confmatrix)
                print('something unexpected happened')
                print(foldData['feature_training'].any())
                print(foldData['category_training_name'])
                exit()


    #for classifyNo in range(cfg['classifiernumber']):
    @ray.remote(num_returns = 2)
    def classifying(classData):

        confmatfinal = np.zeros((rc,rc))
        #map returns map object which can be unpacked by zip and repacked so each return object is zipped in two seperate lists.
        fold, confmat = zip(*map(folding, classData['fold']))

        #Sum the confusionmatrices to receive confmatfinal
        np.sum(confmat, axis = 0, out = confmatfinal)

        return {'fold': np.array(fold)}, confmatfinal


    tempClassifier, tempConfmatfinal = zip(*map(classifying.remote, data['classifier']))

    model = dictFork.remote('confmatfinal', *tempConfmatfinal)
    model = dictAppend.remote(model, 'classifier', *tempClassifier)

    return model


if __name__ == "__main__":
    '''
    Placeholder main function to mess around with code, test and profile function.
    '''

    import pickle
    import time
    with open('./data/Visual/Subj02/7-ClassifierTraining/Subj02_study_datapart.pkl', 'rb') as file:
        matfile = pickle.load(file)['Subj02_study_datapart']

    cfg = {'training_algorithm': 1, 'fold': 10, 'classifiernumber': 20, 'category_model': ['Face', 'Landmark', 'Object']}
    t = time.time()
    crossval = mvpa_traincrossvalclassifier(cfg, matfile)

    print(time.time()-t)

    with open('crossvaltest.pkl', 'wb') as file:
        pickle.dump({'Subj02_study_crossvalclass': crossval}, file)