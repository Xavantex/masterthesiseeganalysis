import os
os.environ["OMP_NUM_THREADS"] = "1"
from scipy.io import loadmat
from numpy import array_split
from mvpa import mvpa_traincrossvalclassifierOpti as tcc
from mvpa import mvpa_datapartitionOpti as dp
from mvpa import mvpa_classifierperfOpti as cp

#This processes all amplification data.
#Extra dependendencies, can add normal data files here as well and code or modules as I have.
depend_extra = (tcc, dp, cp,)

description = r'''
Accelerating Do11 on single PC
'''
#input parameters
options = dict(
	subjects=list,
    epoch=list,
)
#Potential input job references.
#jobs = ('source',)

#Serial things to do before parallel stuff
#def prepare()

#parallel function, job:current job, sliceno: the current "process/processor", slices: Total number of "processors/processes"
def analysis(job, sliceno, slices):
    #Divide the subject list so that each sliceno processes their own subjects.
    sublist = array_split(options.subjects, slices)[sliceno]


    for epo in options.epoch:
        for sub in sublist:

            fname = job.input_filename(f'./{sub}/6-ClassificationData/{sub}_{epo}_data_amp.mat')
            cdata = loadmat(fname, chars_as_strings = True, simplify_cells = True)[f'{sub}_{epo}_data_amp']

            directory = f'./data/Visual/{sub}/7-ClassifierTraining/'

            os.makedirs(os.path.dirname(directory), exist_ok=True)

            #Partition of the data
            cfg = {'classifiernumber': 40, 'fold': 10}
            datapart = dp.mvpa_datapartition(cfg, cdata)

            #Partition of the data
            print('Data was partitioned and feature reduction for each partition is complete...')
            
            #Commented out because datapartition is by far the largest dataobject and takes ridicolous space, of course no problem if one has more than like 50 Gb
            #job.save({f'{sub}_{epo}_datapart_amp': datapart}, directory + f'{sub}_{epo}_datapart_amp.pickle')
            
            #Train the cross validated classifier
            cfg = {'training_algorithm': 1, 'fold': 10, 'classifiernumber': 40, 'category_model': ['Face', 'Landmark', 'Object']}
            crossval = tcc.mvpa_traincrossvalclassifier(cfg, datapart)

            print('Cross Validated model successfully performed')

            job.save({f'{sub}_{epo}_crossvalclass_amp': crossval}, directory + f'{sub}_{epo}_crossvalclass_amp.pickle')

            #Performance
            cfg = {'performance': 1, 'category_model': ['Face', 'Landmark', 'Object'], 'classifiernumber': 40}
            crossval_perf = cp.mvpa_classifierperf(cfg, crossval)
            print('Performance calculated')
    
            job.save({f'{sub}_{epo}_crossvalclass_performance_amp': crossval_perf}, directory + f'{sub}_{epo}_crossvalclass_performance_amp.pickle')


#Serial code after analysis to aggregate results, or do something which must be serial.
#def synthesis(analysis_res):
#    pass