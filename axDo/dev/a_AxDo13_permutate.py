import os
os.environ["OMP_NUM_THREADS"] = "1"
from scipy.io import loadmat
from numpy import array_split
from mvpa import mvpa_applycrossvalclassifierOptiOneLookupDict as acc
import numpy as np


depend_extra = (acc,)

description = r'''
Accelerating Do11 on single PC
'''
#Input parameters
options = dict(
	subjects=list,
    epoch=list,
    size=list,
    NOF_PERMUTATIONS=int,
)
#Input job reference
jobs = ('do11Job',)

#Something serial before analyis
#def prepare()

#job:current job, sliceno: current "processor", slices:total amount of process with this function
def analysis(job, sliceno, slices):
    #Divide the subject list so that each sliceno processes their own subjects.
    sublist = array_split(options.subjects, slices)[sliceno]

    size = options.size
    NOF_PERMUTATIONS = options.NOF_PERMUTATIONS
    jobbig = jobs.do11Job

    for epo in options.epoch:
        for sub in sublist:
            print(sub)

            fname = job.input_filename(f'./{sub}/6-ClassificationData/{sub}_test_data_visual.mat')
            cdata = loadmat(fname, chars_as_strings = True, simplify_cells = True)[f'{sub}_test_data_visual']

            directory = f'./data/Visual/{sub}/7-ClassifierTraining/'#directory of the data

            crossdata = jobbig.load(directory + f'{sub}_{epo}_crossvalclass.pickle')[f'{sub}_{epo}_crossvalclass']#give imput data

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

                per_predtest = acc.mvpa_applycrossvalclassifier(cfg, crossdata, per)#PER%d_predtest

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

                job.save({f'{sub}_P{permutation+1}': sub_P}, directory + f'{sub}_P{permutation+1}.pickle')