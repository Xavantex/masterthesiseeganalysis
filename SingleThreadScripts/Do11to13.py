import scipy.io as sio
import pickle
import numpy as np
import os
from mvpa.mvpa_traincrossvalclassifier import mvpa_traincrossvalclassifier
from mvpa.mvpa_datapartition import mvpa_datapartition
from mvpa.mvpa_classifierperf import mvpa_classifierperf
from mvpa.mvpa_applycrossvalclassifier import mvpa_applycrossvalclassifier
#from threading import Thread
import scipy.stats as pystats
import skimage.filters as filter
import warnings
warnings.filterwarnings("error")

def main(subject = np.array(['Subj01']), epoch = np.array(['study']), NOF_PERMUTATIONS = 5, size = [20, 20]):

    #--------------------------------------------------------------------------
    # Training the Classifier
    # Frequency
    #--------------------------------------------------------------------------

    def Do11TrainClassifierNonAmp(subject = 'Subj01', epoch = 'study'):
        response = np.array(['visual'])

        directory = f'./data/Visual/{subject}/7-ClassifierTraining/'#directory of the data
        os.makedirs(os.path.dirname(directory), exist_ok=True)

        temp = sio.loadmat(f'./Visual/{subject}/6-ClassificationData/{subject}_{epoch}_data', chars_as_strings = True, simplify_cells = True)
        cdata = temp[f'{subject}_{epoch}_data']
        #Partition of the data
        cfg = {'classifiernumber': 20, 'fold': 10}
        datapart = mvpa_datapartition(cfg, cdata)
        print('Data was partitioned and feature reduction for each partition is complete...')

        #Train the cross validated classifier
        cfg = {'training_algorithm': 1, 'fold': 10, 'classifiernumber': 20, 'category_model': ['Face', 'Landmark', 'Object']}
        crossvalclass = mvpa_traincrossvalclassifier(cfg, datapart)
        print('Cross Validated model successfully performed')
        #Performance
        cfg = {'performance': 1, 'category_model': ['Face', 'Landmark', 'Object'], 'classifiernumber': 20}
        crossvalclass_performance = mvpa_classifierperf(cfg, crossvalclass)
        print('Performance calculated')
        #Save
        file = open(directory + f'{subject}_{epoch}_crossvalclass.pkl', 'wb')
        pickle.dump({f'{subject}_{epoch}_crossvalclass': crossvalclass},file)
        file.close()
        
        sio.savemat(file_name = directory + f'{subject}_{epoch}_datapart.mat', mdict = {f'{subject}_{epoch}_datapart': datapart})
        sio.savemat(file_name = directory + f'{subject}_{epoch}_crossvalclass_performance.mat', mdict = {f'{subject}_{epoch}_crossvalclass_performance': crossvalclass_performance})
        
        return crossvalclass

    #--------------------------------------------------------------------------
    # Training the Classifier
    # Amplitude
    #--------------------------------------------------------------------------

    def Do11TrainClassifierAmp(subject = 'Subj01', epoch = 'study'):
        #response = np.array(['visual'])
        
        
        #for j, sub in enumerate(subject):
        print(subject)
        #    for e, epo in enumerate(epoch):
        directory = f'./data/Visual/{subject}/7-ClassifierTraining/'
        os.makedirs(os.path.dirname(directory), exist_ok=True)

        temp = sio.loadmat(f'./Visual/{subject}/6-ClassificationData/{subject}_{epoch}_data_amp', chars_as_strings = True, simplify_cells = True)
        cdata = temp[f'{subject}_{epoch}_data_amp']
        #partition of the data
        cfg = {'classifiernumber': 40, 'fold': 10}
        datapart_amp = mvpa_datapartition(cfg, cdata)
        print('Data was partitioned and feature reduction for each partition is complete')
        #Train the cross validated classifier
        cfg = {'training_algorithm': 1, 'fold': 10, 'classifiernumber': 40, 'category_model': ['Face', 'Landmark', 'Object']}
        crossvalclass_amp = mvpa_traincrossvalclassifier(cfg, datapart_amp)
        print('Cross Validated model successfully performed!')
        #Performance
        cfg = {'performance': 1, 'category_model': ['Face', 'Landmark', 'Object'], 'classifiernumber': 40}
        crossvalclass_performance_amp = mvpa_classifierperf(cfg, crossvalclass_amp)
        print('Performance calculated!')

        #save
        file = open(directory + f'{subject}_{epoch}_crossvalclass_amp.pkl', 'wb')
        pickle.dump({f'{subject}_{epoch}_crossvalclass_amp': crossvalclass_amp},file)
        file.close()

        #sio.savemat(file_name = f'./Visual/{sub}/7-ClassifierTraining/{sub}_{epo}_crossvalclass_amp.mat', mdict = {f'{sub}_{epo}_crossvalclass_amp': crossvalclass_amp})
        sio.savemat(file_name = directory + f'{subject}_{epoch}_datapart_amp.mat', mdict = {f'{subject}_{epoch}_datapart_amp': datapart_amp})
        sio.savemat(file_name = directory + f'{subject}_{epoch}_crossvalclass_performance_amp.mat', mdict = {f'{subject}_{epoch}_crossvalclass_performance_amp': crossvalclass_performance_amp})

        return crossvalclass_amp



    #------------------------------------------------------------------------------
    #  TEST THE CROSS VALIDATED CLASSIFIER ON THE RETRIEVAL DATA
    #------------------------------------------------------------------------------
    #  USE DATA FROM visual CLASSIFIED TRIALS - FREQUENCY
    #------------------------------------------------------------------------------

    def Do12TestClassifier(crossdata, cdata, subject = 'Subj01', epoch = 'study'):

        #print(j)

        directory = f'./data/Visual/{subject}/7-ClassifierTraining/'#directory of the data
        os.makedirs(os.path.dirname(directory), exist_ok=True) #Might not need these 2 lines aye?

        cfg = {'fold': 10,
                'classifiernumber': 20,
                'timebinsnumber': 20,
                'category_predict': np.array(['Face', 'Landmark', 'Object']),
                'trials': 'all',
                'category_model': np.array(['Face', 'Landmark', 'Object'])}
        
        #Apply calssifier into the correct visual/ correct visual trials
        predtest_visual = mvpa_applycrossvalclassifier(cfg, crossdata, cdata)
        
        print(f'{subject}')
        print('Done')

        directory = f'./data/Visual/{subject}/8-ClassifierTesting/'
        os.makedirs(os.path.dirname(directory), exist_ok=True)

        sio.savemat(file_name = directory + f'{subject}_{epoch}_predtest_visual.mat',
                    mdict = {f'{subject}_{epoch}_predtest_visual': predtest_visual})
        

        cfg = {'performance': 2,
                'category_model': np.array(['Face', 'Landmark', 'Object']),
                'category_predict': np.array(['Face', 'Landmark', 'Object']),
                'classifiernumber': 20,
                'timebinsnumber': 20}
        
        predtest_visual_performance = mvpa_classifierperf(cfg, predtest_visual)#Performance

        sio.savemat(file_name = directory + f'{subject}_{epoch}_predtest_visual',
                    mdict = {f'{subject}_{epoch}_predtest_visual_performance': predtest_visual_performance})


        print('Performance calculated')


    #------------------------------------------------------------------------------
    #  PERMUTATION FOR EACH PARTICIPANT!!!! FREQUENCY
    #------------------------------------------------------------------------------

    def Do13TestClassifier_ShuffledLabels_permutation(permutation, crossdata, cdata, subject = 'Subj01', epoch = 'study', size = [20, 20]):


        #print(subject)

        directory = f'./data/Visual/{subject}/7-ClassifierTraining/'
        os.makedirs(os.path.dirname(directory), exist_ok=True)

        R = np.random.permutation(len(cdata['category_name']))
        per = {'category_name': cdata['category_name'][R],
            'category': cdata['category'][R],
            'feature_name': cdata['feature_name'],
            'trial_info': cdata['trialinfo'],
            'feature': cdata['feature'],
            'numclassifiers': cdata['numclassifiers']}

        #Apply calssifier into the shuffled category trials
        cfg = {'fold': 10,
            'classifiernumber': 20,
            'timebinsnumber': 20,
            'category_predict': ['Face', 'Landmark', 'Object'],
            'trials': 'all',
            'category_model': ['Face', 'Landmark', 'Object']}
        per_predtest = mvpa_applycrossvalclassifier(cfg, crossdata, per)#PER%d_predtest

        #Construct and save the Permutation Map
        sub_P = np.zeros((size[0], size[1]))
        for col in range(size[0]):
            for row in range(size[1]):
                trials = sum(per_predtest['timebin'][col]['confmatfinal'][row], 2)
                sub_P[row,col] = ((per_predtest['timebin'][col]['confmatfinal'][row][0,0]/trials[0]*100) + 
                                (per_predtest['timebin'][col]['confmatfinal'][row][1,1]/trials[1]*100) +
                                (per_predtest['timebin'][col]['confmatfinal'][row][2,2]/trials[2]*100))
        
        directory = f'./data/Visual/{subject}/8-ClassifierTesting/PermutationStudyDecodeTestVisual/'
        os.makedirs(os.path.dirname(directory), exist_ok=True)
        sio.savemat(file_name = directory + f'{subject}_P{permutation+1}', mdict = {f'{subject}_P{permutation+1}': sub_P})

        return sub_P

    #------------------------------------------------------------------------------
    # For each permutation a T-Test is calculated
    # (classification accuracy is compared against chance (33.33))
    # Each T test is stored in a Struture called Random_Distribution
    # _____________________________________________________________________________
    # 
    # Dependent on Do13TestClassifier_ShuffledLabels_permutation
    #
    #------------------------------------------------------------------------------ 

    def Do13TestClassifier_ShuffledLabels_ttest(permutation, subject = ['Subj01'], epoch = ['study'], size = [20, 20]):

        #for permutation in range(NOF_PERMUTATIONS):
            #for e in epoch:
                #cdata = np.zeros((len(subject), size[0], size[1]))
                #for j, sub in enumerate(subject):
                #    directory = f'./data/Visual/{sub}/8-ClassifierTesting/PermutationStudyDecodeTestVisual/'
                #    temp = sio.loadmat(directory + f'{sub}_P{permutation+1}')
                #    cdataDo13[j] = temp[f'{sub}_P{permutation+1}']
                #    cdataDo13[j] = filter.gaussian(cdata[j], sigma = 1, truncate = 2, preserve_range=True)


        x = np.zeros(len(subject))
        for col in range(size[0]):
            for row in range(size[1]):
                for i, sub2 in enumerate(subject):
                    x[i] = cdataDo13ttest[i, row, col]
                #Does the same ttest in matlab with 'tail' 'both' settings, as it checks vs the alternative that it is not part of the mean 33.33
                stat, pvalue  = pystats.ttest_1samp(x, 33.33, nan_policy = 'propagate')
                T_P[row, col] = stat
                P_P[row, col] = pvalue
        
        print(f'permutation {permutation} .../n')
        Random_Distribution['Plevel'][permutation] = P_P
        Random_Distribution['Ttest'][permutation] = T_P





    size = [20, 20]
    T_P = np.zeros((size[0], size[1]))
    P_P = np.zeros((size[0], size[1]))
    Random_Distribution = {'Plevel': np.empty(NOF_PERMUTATIONS, dtype = np.ndarray),
                            'Ttest': np.empty(NOF_PERMUTATIONS, dtype = np.ndarray)}
    
    Do12Do13cdata = np.empty(len(subject), dtype = dict)
    Do12Do13Crossdata = np.empty(len(subject), dtype = dict)

    for e, epo in enumerate(epoch):
        for j, sub in enumerate(subject):

            print(sub)
            
            Do12Do13Crossdata[j] = Do11TrainClassifierNonAmp(subject = sub, epoch = epo)

            Do11TrainClassifierAmp(subject = sub, epoch = epo)

            Do12Do13cdata[j] = sio.loadmat(f'./Visual/{sub}/6-ClassificationData/{sub}_test_data_visual',
                            chars_as_strings = True, simplify_cells = True)[f'{sub}_test_data_visual']#give imput data

            Do12TestClassifier(crossdata = Do12Do13Crossdata[j], cdata = Do12Do13cdata[j], subject = sub, epoch = epo)

        #IDEA SO FAR, make a node for each crossvalclass and make it do all the permutations, once a chunk of it is done, send it back for ttest. Ttest waits for chunks from all nodes/subjects
        #And then performs the ttest, do this for all permutations and bom balam bam speedup is 1000x. Time to Read on Ray.io and dask for how synchronization etc. works.
        for permutation in range(NOF_PERMUTATIONS):
            cdataDo13ttest = np.zeros((len(subject), size[0], size[1]))
            for j, sub in enumerate(subject):
                temp = Do13TestClassifier_ShuffledLabels_permutation(permutation = permutation, crossdata = Do12Do13Crossdata[j], cdata = Do12Do13cdata[j], subject = sub, epoch = epo, size = size)
                cdataDo13ttest[j] = filter.gaussian(temp, sigma = 1, truncate = 2, preserve_range=True)
        
            Do13TestClassifier_ShuffledLabels_ttest(permutation = permutation, subject = subject, epoch = epo, size = size)



    directory = f'./data/Visual/{subject[-1]}/8-ClassifierTesting/PermutationStudyDecodeTestVisual/'
    tempdict = {'Random_Distribution': Random_Distribution}
    sio.savemat(file_name = directory + f'Random_Distribution.mat', mdict = tempdict)


if __name__ == "__main__":
    main(subject = np.array(['Subj01', 'Subj02', 'Subj03']), epoch = np.array(['study']), NOF_PERMUTATIONS = 5, size = [20, 20])
