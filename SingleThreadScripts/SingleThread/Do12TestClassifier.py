import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from mvpa.mvpa_applycrossvalclassifierOptiOneLookupDict import mvpa_applycrossvalclassifier
from mvpa.mvpa_classifierperfOpti import mvpa_classifierperf
from scipy.io import loadmat
from pickle import load, dump

#------------------------------------------------------------------------------
#  TEST THE CROSS VALIDATED CLASSIFIER ON THE RETRIEVAL DATA
#------------------------------------------------------------------------------
#  USE DATA FROM visual CLASSIFIED TRIALS - FREQUENCY
#------------------------------------------------------------------------------

def Do12TestClassifier(subject = ['Subj01'], epoch = ['study']):

    for j, sub in enumerate(subject):
        for epo in epoch:#Should rename e, very hard to identify and change etc.

            directory = f'./data/Visual/{sub}/7-ClassifierTraining/'#directory of the data, should already exist otherwise can't load data

            cdata = loadmat(f'./Visual/{sub}/6-ClassificationData/{sub}_test_data_visual',
                                chars_as_strings = True,
                                simplify_cells = True)[f'{sub}_test_data_visual']#give imput data, skip first key since unecessary.
            
            with open(directory + f'{sub}_{epo}_crossvalclass.pkl', 'rb') as file:
                crossdata = load(file)[f'{sub}_{epo}_crossvalclass']#give classifier

            cfg = {'fold': 10,
                   'classifiernumber': 20,
                   'timebinsnumber': 20,
                   'category_predict': np.array(['Face', 'Landmark', 'Object']),
                   'trials': 'all',
                   'category_model': np.array(['Face', 'Landmark', 'Object'])}
            
            #Apply calssifier into the correct visual/ correct visual trials
            predtest_visual = mvpa_applycrossvalclassifier(cfg, crossdata, cdata)
            
            print(f'{sub} Done')

            directory = f'./data/Visual/{sub}/8-ClassifierTesting/'
            #Create new dir if not exist
            os.makedirs(os.path.dirname(directory), exist_ok=True)
            
            with open(directory + f'{sub}_{epo}_predtest_visual.pkl', 'wb') as file:
                dump({f'{sub}_{epo}_predtest_visual': predtest_visual},file)

            cfg = {'performance': 2,
                    'category_model': np.array(['Face', 'Landmark', 'Object']),
                    'category_predict': np.array(['Face', 'Landmark', 'Object']),
                    'classifiernumber': 20,
                    'timebinsnumber': 20}
            
            predtest_visual_performance = mvpa_classifierperf(cfg, predtest_visual)#Performance

            with open(directory + f'{sub}_{epo}_predtest_visual.pkl', 'wb') as file:
                dump({f'{sub}_{epo}_predtest_visual_performance': predtest_visual_performance},file)

            print('Performance calculated')


if __name__ == "__main__":

    subject = np.array(['Subj01', 'Subj02', 'Subj03',
                        'Subj04', 'Subj05', 'Subj06',
                        'Subj07', 'Subj08', 'Subj09',
                        'Subj11', 'Subj12', 'Subj13',
                        'Subj14', 'Subj15', 'Subj16',
                        'Subj17', 'Subj18', 'Subj19'])
                        
    Do12TestClassifier(subject = subject, epoch = ['study'])