#Explicitly importing functions improves performance slightly, if one uses the functions multiple times(like thousands of times)
import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from scipy.io import loadmat
from pickle import dump, load
from mvpa.mvpa_traincrossvalclassifierOpti import mvpa_traincrossvalclassifier
from mvpa.mvpa_datapartitionOpti import mvpa_datapartition
from mvpa.mvpa_classifierperfOpti import mvpa_classifierperf


def Do11TrainClassifier(subject = np.array(['Subj01']), epoch = np.array(['study']), random_state = None):

    Do11TrainClassifierNonAmp(subject = subject, epoch = epoch, random_state = None)
    Do11TrainClassifierAmp(subject = subject, epoch = epoch, random_state = None)


#--------------------------------------------------------------------------
# Training the Classifier
# Frequency
#--------------------------------------------------------------------------

def Do11TrainClassifierNonAmp(subject = np.array(['Subj01']), epoch = np.array(['study']), random_state = None):
    #response = np.array(['visual'])

    def oClass(sub):
        
        def iClass(epo):
        
            directory = f'./data/Visual/{sub}/7-ClassifierTraining/'#directory for saving of the data
            #Create the new directory in case it doesn't already exist
            os.makedirs(os.path.dirname(directory), exist_ok=True)

            #Python handles loading .mat files almost better than matlab itself
            cdata = loadmat(f'./Visual/{sub}/6-ClassificationData/{sub}_{epo}_data',
                            chars_as_strings = True,
                            simplify_cells = True)[f'{sub}_{epo}_data']#Skip first key since it's unecessary.

            #Partition of the data
            cfg = {'classifiernumber': 20, 'fold': 10}
            datapart = mvpa_datapartition(cfg, cdata, random_state = random_state)
            print('Data was partitioned and feature reduction for each partition is complete...')

            #While it is possible to save the files as .mat files with scipy.io I didn't get it to work without giving severe penalaties on performance
            with open(directory + f'{sub}_{epo}_datapart.pkl', 'wb') as file:
                dump({f'{sub}_{epo}_datapart': datapart}, file)

            #Train the cross validated classifier
            cfg = {'training_algorithm': 1, 'fold': 10, 'classifiernumber': 20, 'category_model': ['Face', 'Landmark', 'Object']}
            crossvalclass = mvpa_traincrossvalclassifier(cfg, datapart)
            del datapart#Deleting now redundant variable to free memory
            print('Cross Validated model successfully performed')
            
            #Save
            with open(directory + f'{sub}_{epo}_crossvalclass.pkl', 'wb') as file:
                dump({f'{sub}_{epo}_crossvalclass': crossvalclass}, file)
            
            #Performance
            cfg = {'performance': 1, 'category_model': ['Face', 'Landmark', 'Object'], 'classifiernumber': 20}
            crossvalclass_performance = mvpa_classifierperf(cfg, crossvalclass)
            print('Performance calculated')
            #Save
            with open(directory + f'{sub}_{epo}_crossvalclass_performance.pkl', 'wb') as file:
                dump({f'{sub}_{epo}_crossvalclass_performance': crossvalclass_performance}, file)
        

        print(sub)
        #Lists are much faster by compiling the function, skipping reinterpretation of the function multiple times as you do with for loops.
        list(map(iClass, epoch))#Works like a for loop by iterating over epoch, using each element in epoch as input to iClass here
        

    list(map(oClass, subject))
            

#--------------------------------------------------------------------------
# Training the Classifier
# Amplitude
#--------------------------------------------------------------------------

def Do11TrainClassifierAmp(subject = np.array(['Subj01']), epoch = np.array(['study']), random_state = None):
    response = np.array(['visual'])
    

    def oClass(sub):
    
        def iClass(epo):

            directory = f'./data/Visual/{sub}/7-ClassifierTraining/'
            #Create dir, if not exist
            os.makedirs(os.path.dirname(directory), exist_ok=True)

            cdata = loadmat(f'./Visual/{sub}/6-ClassificationData/{sub}_{epo}_data_amp',
                            chars_as_strings = True,
                            simplify_cells = True)[f'{sub}_{epo}_data_amp']

            #partition of the data
            cfg = {'classifiernumber': 40, 'fold': 10}
            datapart_amp = mvpa_datapartition(cfg, cdata, random_state = random_state)
            print('Data was partitioned and feature reduction for each partition is complete')

            #Again saving data.
            with open(directory + f'{sub}_{epo}_datapart_amp.pkl', 'wb') as file:
                dump({f'{sub}_{epo}_datapart_amp': datapart_amp}, file)
            
            #Train the cross validated classifier
            cfg = {'training_algorithm': 1, 'fold': 10, 'classifiernumber': 40, 'category_model': ['Face', 'Landmark', 'Object']}
            crossvalclass_amp = mvpa_traincrossvalclassifier(cfg, datapart_amp)
            del datapart_amp#Free up memory
            print('Cross Validated model successfully performed!')
            
            with open(directory + f'{sub}_{epo}_crossvalclass_amp.pkl', 'wb') as file:
                dump({f'{sub}_{epo}_crossvalclass_amp': crossvalclass_amp}, file)

            #Performance
            cfg = {'performance': 1, 'category_model': ['Face', 'Landmark', 'Object'], 'classifiernumber': 40}
            crossvalclass_performance_amp = mvpa_classifierperf(cfg, crossvalclass_amp)
            print('Performance calculated!')

            #save
            with open(directory + f'{sub}_{epo}_crossvalclass_performance_amp.pkl', 'wb') as file:
                dump({f'{sub}_{epo}_crossvalclass_performance_amp': crossvalclass_performance_amp}, file)




        print(sub)
        list(map(iClass, epoch))

    list(map(oClass, subject))


if __name__ == "__main__":

    subject = np.array(['Subj01', 'Subj02', 'Subj03',
                        'Subj04', 'Subj05', 'Subj06',
                        'Subj07', 'Subj08', 'Subj09',
                        'Subj11', 'Subj12', 'Subj13',
                        'Subj14', 'Subj15', 'Subj16',
                        'Subj17', 'Subj18', 'Subj19'])

    Do11TrainClassifier(subject = subject, epoch = ['study'])