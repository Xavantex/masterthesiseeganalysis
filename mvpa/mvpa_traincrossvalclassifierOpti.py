import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OutputCodeClassifier as occ#Not something that was actually used in the original code, but seemed like on the surface. Keeping as reference
from sklearn.svm import LinearSVC as Linsvc
#from sklearn.svm import SVC as SVC#Option to LinearSVC, however it is slower in general
from sklearn.metrics import confusion_matrix as conm


def mvpa_traincrossvalclassifier(cfg: dict, data: dict, random_state = None):

    '''
    ---------------------------------------------------------------------------------------------------

    [model] = mvpa_traincrossvalclassifier(cfg,data)

    WHERE:
    DATA is an output from mvpa_datapartition
    cfg.training_algorithm
    training_algorithm = 1 - SVM with three classes, onevsall strategy and linear kernel
    training_algorithm = 2 - SVM with two classes and linear kernel
    cfg.fold = 10;
    cfg.classifiernumber = 20;
    Output model is structure with has many fields has classifiers trained
    ---------------------------------------------------------------------------------------------------
    '''

    def training_1(foldy):
        #Defining linear svc, with hinge loss function a stop tolerance of 10^-4, if using multiclass, it uses the strategy one vs rest/one vs all.
        ClassifierSVM = Linsvc(loss = 'hinge', tol = 1e-4, multi_class = 'ovr', random_state=random_state, max_iter = 9000)
        #set up error output codes, using linSVC model as 1vs1 classifier
        ecoc = occ(ClassifierSVM, code_size = 1, random_state=random_state, n_jobs=1)
        #Set up parameters for exhaustive search of best parameter choice
        parameters = {
            'estimator__C': [0.0001, 0.001, 0.01]
        }
        #setup exhaustive search object for optimal parameter choice, here only C is iterated over but can be multiple arguments
        svc = GridSearchCV(estimator = ecoc, param_grid = parameters, refit = True)

        #Fit to best model.
        svc.fit(foldy['feature_training'],
                foldy['category_training_name'])
        
        return svc


    def training_2(foldy):
        #Defining linear svc, with hinge loss function a stop tolerance of 10^-4, if using multiclass, it uses the strategy one vs rest/one vs all. 
        ClassifierSVM = Linsvc(loss = 'hinge', tol = 1e-4, multi_class = 'ovr', random_state=random_state, max_iter = 9000)
        #Set up parameters for exhaustive search of best parameter choice
        parameters = {
            'C': [0.0001, 0.001, 0.01]
        }
        #Setup exhaustive search object for optimal parameter choice, here only C is iterated over but can be multiple arguments
        svc = GridSearchCV(estimator = ClassifierSVM, param_grid = parameters, n_jobs=1, refit = True)

        #Fit to best model.
        svc.fit(X = foldy['feature_training'],
                y = foldy['category_training_name'])
        
        return svc

    #Switch statement in case we wanna keep multiple training modules, such as ECOC, this is probably also faster than a block of ifs.
    #This also allows us to remove the ifs inside the for loop, optimizing further.
    switch = {
        1: training_2,
        2: training_2
    }
    #Set which function to use
    training = switch[cfg['training_algorithm']]



    rc = len(cfg['category_model'])

    #Equivalent in for loop
    #for classyNo in range(cfg['classifiernumber']):
    def classifierLoop(classyNo):

        def foldLoop(foldNo):

            foldy = classy['fold'][foldNo]
            #While if statements are generally faster than try arguments if using fast conditions like boolean expressions,
            #However searching if lists are empty of True elements are quite expensive and therefore better to just test and see if it works and then take performance loss here.
            #if foldy['feature_training'].any():#Should it be any or size? Any is if any element is not 0 or false, size is that it's not just an empty array
            
            try: # Try is faster than doing lookups all the time, instead just handle it with the except statement if something goes wrong, since we don't care anyway with the wrongdoings
                svc = training(foldy)

                label = svc.predict(foldy['feature_test'])
                score = np.sum(np.equal(label, foldy['category_test_name']))/np.size(foldy['category_test_name'])
                confmatrix = conm(y_true = foldy['category_test_name'],
                    y_pred = label,
                    labels = np.unique(foldy['category_training_name']))#could just use 'category_test_name' here if we expect them to use the same categories and because it's less elements to iterate over.

                return {'model': svc,
                        'predict': label,
                        'truelabel': foldy['category_test_name'],
                        'feature': foldy['selectedfeature_name'],
                        'likelihood': score, 'confmat': confmatrix}, confmatrix

            #If we do get errors, we check that it was in fact empty and just return if it was. otherwise we print data structs to check if there's something we missed.
            except ValueError as err:
                if not foldy['feature_training'].any():
                    return None, np.zeros((rc,rc))
                else:
                    print(err)
                    print(f'{label}\n{score}\n{confmatrix}\n{confmatfinal}\n')
                    print('something unexpected happened')
                    print(foldy['feature_training'])
                    print(foldy['feature_training'].any())
                    print(foldy['category_training_name'])
                    exit()


        confmatfinal = np.zeros((rc,rc))

        classy = data['classifier'][classyNo]
        
        fold, confmat = zip(*map(foldLoop, range(cfg['fold'])))

        np.sum(confmat, axis = 0, out = confmatfinal)

        return {'fold': np.array(fold, dtype=dict)}, confmatfinal


    modelclassifier, modelconfmatfinal = zip(*map(classifierLoop, range(cfg['classifiernumber'])))

    return {'classifier': np.array(modelclassifier, dtype = dict),
            'confmatfinal': np.array(modelconfmatfinal)}


if __name__ == "__main__":
    '''
    Quick main function to test out function, this is better done in a unittest and is simply a placeholder,
    or if someone want to use this function for a single file.
    '''
    
    import pickle
    import time
    from scipy.io import loadmat
    with open('./data/Visual/Subj02/7-ClassifierTraining/Subj02_study_datapart.pkl', 'rb') as file:
        matfile = pickle.load(file)['Subj02_study_datapart']

    cfg = {'training_algorithm': 2, 'fold': 10, 'classifiernumber': 20, 'category_model': ['Face', 'Landmark', 'Object']}
    crossval = mvpa_traincrossvalclassifier(cfg, matfile)
    with open('crossvaltest.pkl', 'wb') as file:
        pickle.dump({'Subj02_study_crossvalclass': crossval}, file)