from scipy.io import loadmat, savemat
from mvpa.mvpa_applycrossvalclassifier import mvpa_applycrossvalclassifier
from scipy.stats import ttest_1samp
from skimage.filters import gaussian
from pickle import load
import numpy as np
import os
import warnings
warnings.filterwarnings("error")


#Verbal
# directory  = '//psysrv004/psymemlab/Projects/TAPMVPA-LTH/Verbal/'; %directory of the data
# subject = {'Subj01' 'Subj02' 'Subj06' 'Subj07' 'Subj08' 'Subj09' 'Subj10' 'Subj11'...
#     'Subj12' 'Subj14' 'Subj15' 'Subj16' 'Subj17' 'Subj18' 'Subj19' 'Subj20' 'Subj21' 'Subj22'};

def Do13TestClassifier_ShuffledLabels(subject = ['Subj01'], epoch = ['study'], NOF_PERMUTATIONS = 5, size = [20, 20]):

    #Visual
    Do13TestClassifier_ShuffledLabels_permutation(subject = subject, epoch = epoch, NOF_PERMUTATIONS = NOF_PERMUTATIONS, size = size)
    Do13TestClassifier_ShuffledLabels_ttest(subject = subject, epoch = epoch, NOF_PERMUTATIONS = NOF_PERMUTATIONS, size = size)


#------------------------------------------------------------------------------
#  PERMUTATION FOR EACH PARTICIPANT!!!! FREQUENCY
#------------------------------------------------------------------------------

def Do13TestClassifier_ShuffledLabels_permutation(subject = ['Subj01'], epoch = ['study'], NOF_PERMUTATIONS = 5, size = [20, 20]):


    for j, sub in enumerate(subject):
        for e in epoch:
            print(sub)

            temp = loadmat(f'./Visual/{sub}/6-ClassificationData/{sub}_test_data_visual', chars_as_strings = True, simplify_cells = True)#directory of the data
            cdata = temp[f'{sub}_test_data_visual']

            directory = f'./data/Visual/{sub}/7-ClassifierTraining/'
            os.makedirs(os.path.dirname(directory), exist_ok=True)

            #temp = loadmat(f'./Visual/{sub}/7-ClassifierTraining/{sub}_{e}_crossvalclass')
            #crossdata = temp[f'{sub}_{e}_crossvalclass']

            with open(directory + f'{sub}_{e}_crossvalclass.pkl', 'rb') as file:
                crossdata = load(file)[f'{sub}_{e}_crossvalclass']#give imput data

            for permutation in range(NOF_PERMUTATIONS):#Number of Permutations
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
                        trials = np.sum(per_predtest['timebin'][col]['confmatfinal'][row], axis = 1)
                        sub_P[row,col] = ((per_predtest['timebin'][col]['confmatfinal'][row][0,0]/trials[0]*100) + 
                                        (per_predtest['timebin'][col]['confmatfinal'][row][1,1]/trials[1]*100) +
                                        (per_predtest['timebin'][col]['confmatfinal'][row][2,2]/trials[2]*100))
                
                directory = f'./data/Visual/{sub}/8-ClassifierTesting/PermutationStudyDecodeTestVisual/'
                os.makedirs(os.path.dirname(directory), exist_ok=True)
                savemat(file_name = directory + f'{sub}_P{permutation+1}', mdict = {f'{sub}_P{permutation+1}': sub_P})


#------------------------------------------------------------------------------
# For each permutation a T-Test is calculated
# (classification accuracy is compared against chance (33.33))
# Each T test is stored in a Struture called Random_Distribution
# _____________________________________________________________________________
# 
# Dependent on Do13TestClassifier_ShuffledLabels_permutation
#
#------------------------------------------------------------------------------ 

def Do13TestClassifier_ShuffledLabels_ttest(subject = ['Subj01'], epoch = ['study'], NOF_PERMUTATIONS = 5, size = [20, 20]):
    import time

    T_P = np.zeros((size[0], size[1]))
    P_P = np.zeros((size[0], size[1]))
    Random_Distribution = {'Plevel': np.empty(NOF_PERMUTATIONS, dtype = np.ndarray),
                            'Ttest': np.empty(NOF_PERMUTATIONS, dtype = np.ndarray)}

    #InfoSubjects_mvpa
    #subjects = fieldnames(Infosubjects);
    #Infosubjects = struct2cell(Infosubjects);
    for e in epoch:
        for permutation in range(NOF_PERMUTATIONS):
            cdata = np.zeros((len(subject), size[0], size[1]))
            for j, sub in enumerate(subject):
                directory = f'./data/Visual/{sub}/8-ClassifierTesting/PermutationStudyDecodeTestVisual/'
                temp = loadmat(directory + f'{sub}_P{permutation+1}')
                cdata[j] = temp[f'{sub}_P{permutation+1}']
                cdata[j] = gaussian(cdata[j], sigma = 1, truncate = 2, preserve_range=True)
                print(sub)

            #starttime = time.time()
            x = np.zeros(len(subject))
            for col in range(size[0]):
                for row in range(size[1]):
                    #starttime2 = time.time()
                    for i, sub2 in enumerate(subject):
                        x[i] = cdata[i, row, col]
                    #Does the same ttest in matlab with 'tail' 'both' settings, as it checks vs the alternative that it is not part of the mean 33.33
                    stat, pvalue  = ttest_1samp(x, 33.33, nan_policy = 'propagate')
                    T_P[row, col] = stat
                    P_P[row, col] = pvalue
                    #stoptime = time.time()-starttime2
                    #print(stoptime)
            #print('lul')
            #print(starttime-time.time())

            
            print(f'permutation {permutation} .../n')
            Random_Distribution['Plevel'][permutation] = P_P
            Random_Distribution['Ttest'][permutation] = T_P



    tempdict = {'Random_Distribution': Random_Distribution}

    savemat(file_name = directory + f'Random_Distribution.mat', mdict = tempdict)
            #print(pvalue)


    #print(cdata)


if __name__ == "__main__":

    subject = np.array(['Subj01', 'Subj02', 'Subj03',
                        'Subj04', 'Subj05', 'Subj06',
                        'Subj07', 'Subj08', 'Subj09',
                        'Subj11', 'Subj12', 'Subj13',
                        'Subj14', 'Subj15', 'Subj16',
                        'Subj17', 'Subj18', 'Subj19'])

    Do13TestClassifier_ShuffledLabels(subject = subject, epoch = ['study'], NOF_PERMUTATIONS = 5, size = [20, 20])