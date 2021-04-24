import numpy as np

def firstPerf(cfg, confmatfinal):
    '''
    cfg.performance = 1,2,...
    cfg.categories = {'Faces' 'Landmarks' 'Objects'};
    1 - classifier
    2 - timebins
    cfg.classifiernumber = 20;
    cfg.timebinsnumber = 20;
    '''
    def sumConf(classy):

        classyConfmatfinal = confmatfinal[classy]
        NumTrialTotal = np.sum(classyConfmatfinal, axis = (0,1))#along which axis do we sum

        #Have to check that we don't divide by zero and *BLOW UP THE UNIVERSE!* or just get div error
        if NumTrialTotal != 0:
            #Removed for loop for numpy functions, and dictcomprehensions.
            #for cat in range(cat_len):
            category = np.diag(classyConfmatfinal)
            tempSum = (category/NumTrialTotal)*100
            classyperf = {k: tempSum[i] for i, k in enumerate(cfg['category_model'])} #Replace data[confmat] with category cat?
            classyperf['Total'] = (np.sum(category)/NumTrialTotal)*100
        else:
            if NumTrialTotal == 0:
                #Return empty arrays if we don't have any hits. could be switched with 0 I guess
                #for cat in range(cat_len):
                classyperf = {k: np.array([]) for k in cfg['category_model']}
                classyperf['Total'] = np.array([])
            else:
                print('unknown error')
                exit()

        return classyperf

    performance = {'classifier': np.array(list(map(sumConf, range(cfg['classifiernumber']))), dtype = dict)}

    return performance

def secondPerf(cfg, timebin):

    def timebinLoop(tb):

        confmatfinal = timebin[tb]['confmatfinal']
        return firstPerf(cfg, confmatfinal)

    return {'timebin': np.array(list(map(timebinLoop, range(cfg['timebinsnumber']))), dtype = dict)}

def thirdPerf(cfg, timebin):

    def timebinLoop(tb):

        def classyLoop(classy):
            classyConfmatfinal = tbconfmatfinal[classy]
            NumTrialTotal = np.sum(classyConfmatfinal, axis = (0,1))
            classyperf = {}

            if NumTrialTotal !=0:
                if cfg['category_predict'] == 'Face':
                    classyperf = {'Face': (classyConfmatfinal[1, 1]/NumTrialTotal)*100,
                                'Landmark': (classyConfmatfinal[1, 2]/NumTrialTotal)*100,
                                'Object': (classyConfmatfinal[1, 3]/NumTrialTotal)*100}
                                            
                elif cfg['category_predict'] == 'Landmark':
                    classyperf = {'Face': (classyConfmatfinal[2, 1]/NumTrialTotal)*100,
                                'Landmark': (classyConfmatfinal[2, 2]/NumTrialTotal)*100,
                                'Object': (classyConfmatfinal[2, 3]/NumTrialTotal)*100}

                elif cfg['category_predict'] == 'Object':
                    classyperf = {'Face': (classyConfmatfinal[3, 1]/NumTrialTotal)*100,
                                'Landmark': (classyConfmatfinal[3, 2]/NumTrialTotal)*100,
                                'Object': (classyConfmatfinal[3, 3]/NumTrialTotal)*100}
            else:
                if NumTrialTotal == 0:
                    if cfg['category_predict'] == 'Face' or cfg['category_predict'] == 'Landmark' or cfg['category_predict'] == 'Object':
                        classyperf = {'Face': np.array([]),
                                    'Landmark': np.array([]),
                                    'Object': np.array([])}

                else:
                    print('unknown error')
                    exit()

            return classyperf

        tbconfmatfinal = timebin[tb]['confmatfinal']
        return {'classifier': np.array(list(map(classyLoop, range(cfg['classifiernumber']))), dtype = dict)}

    return {'timebin': np.array(list(map(timebinLoop, range(cfg['timebinsnumber']))), dtype = dict)}


def mvpa_classifierperf(cfg: dict, data: dict):

    #Switch statement in case we wanna keep multiple training modules, such as ECOC, this is probably also faster than a block of ifs.
    #This also allows us to remove the ifs inside the for loop, optimizing further.
    #Maybe move this outside the function for further optimization, removing a bunch of unecessary code execution
    switch = {
        1: firstPerf,
        2: secondPerf,
        3: thirdPerf
    }
    NameSwitch = {
        1: 'confmatfinal',
        2: 'timebin',
        3: 'timebin'
    }

    classPerf = switch[cfg['performance']] 

    return classPerf(cfg, data[NameSwitch[cfg['performance']]])



if __name__ == '__main__':

    '''
    Placeholder main function for tinkering, profiling, should do testing in unittests but could do here. Could execute single file if missing one or two.
    '''
    from scipy.io import loadmat, savemat

    matfile = loadmat('./Visual/Subj01/7-ClassifierTraining/Subj01_study_crossvalclass', chars_as_strings = True, simplify_cells = True)['Subj01_study_crossvalclass']
    cfg = {'classifiernumber': 20,
           'performance': 1,
           'category_model': ['Face', 'Landmark', 'Object']}
    classifierperf = mvpa_classifierperf(cfg, matfile)
    savemat(file_name = 'classifierperftest.mat', mdict = {'Subj01_study_crossvalclass_performance': classifierperf})