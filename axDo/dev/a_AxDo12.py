import os
os.environ["OMP_NUM_THREADS"] = "1"
from scipy.io import loadmat
from numpy import array_split
from mvpa import mvpa_applycrossvalclassifierOptiOneLookupDict as acc
from mvpa import mvpa_classifierperfOpti as cp
import numpy as np


depend_extra = (acc, cp,)

description = r'''
Accelerating Do11 on single PC
'''

#Input parameters
options = dict(
	subjects=list,
    epoch=list,
)
#input job reference argument.
jobs = ('do11Job',)

#serial stuff before analysis
#def prepare()

#Parallel execution, job:current job, sliceno:Current "process", slices:Total amount of process
def analysis(job, sliceno, slices):
    #Divide the subject list so that each sliceno processes their own subjects.
    sublist = array_split(options.subjects, slices)[sliceno]

    jobbig = jobs.do11Job

    for epo in options.epoch:
        for sub in sublist:

            directory = f'./data/Visual/{sub}/7-ClassifierTraining/'#directory of the data

            fname = job.input_filename(f'./{sub}/6-ClassificationData/{sub}_test_data_visual.mat')
            cdata = loadmat(fname, chars_as_strings = True, simplify_cells = True)[f'{sub}_test_data_visual']#give imput data
            
            crossdata = jobbig.load(directory + f'{sub}_{epo}_crossvalclass.pickle')[f'{sub}_{epo}_crossvalclass']#give classifier

            cfg = {'fold': 10,
                   'classifiernumber': 20,
                   'timebinsnumber': 20,
                   'category_predict': np.array(['Face', 'Landmark', 'Object']),
                   'trials': 'all',
                   'category_model': np.array(['Face', 'Landmark', 'Object'])}
            
            #Apply calssifier into the correct visual/ correct visual trials
            predtest_visual = acc.mvpa_applycrossvalclassifier(cfg, crossdata, cdata)
            
            print(f'{sub} Done')

            directory = f'./data/Visual/{sub}/8-ClassifierTesting/'

            os.makedirs(os.path.dirname(directory), exist_ok=True)

            job.save({f'{sub}_{epo}_predtest_visual': predtest_visual}, directory + f'{sub}_{epo}_predtest_visual.pickle')

            cfg = {'performance': 2,
                    'category_model': np.array(['Face', 'Landmark', 'Object']),
                    'category_predict': np.array(['Face', 'Landmark', 'Object']),
                    'classifiernumber': 20,
                    'timebinsnumber': 20}

            predtest_visual_performance = cp.mvpa_classifierperf(cfg, predtest_visual)#Performance

            print('Performance calculated')

            job.save({f'{sub}_{epo}_predtest_visual_performance': predtest_visual_performance}, directory + f'{sub}_{epo}_predtest_visual.pickle')

#something serial after prepare and analysis
#def synthesis()