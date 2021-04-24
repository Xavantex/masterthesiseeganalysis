import os
os.environ["OMP_NUM_THREADS"] = "1"
from skimage.filters import gaussian
from scipy.stats import ttest_1samp
import numpy as np

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
#input job reference
jobs = ('do13_permutateJob',)

#If we wanna prepare something serial before analysis.
#def prepare()

#Do something in parallel with a designated no:sliceno
#job:Current job, sliceno:current "Processor", slices:Total amount of processes working on this function.
def analysis(job, sliceno, slices):
    #Own function to divide the processors into seperate permutations, since we need all subjects to generate.
    #Divides the number of permutation into as equal subranges for each core to do t-test over,
    #for example with 54 permutations with 18 cores, core #1 does permuation 1-3, core#2 does 4-6 core#3 does 7-9 etc up to permutation 54.
    size = options.size
    NOF_PERMUTATIONS = options.NOF_PERMUTATIONS
    subjects = options.subjects

    jobbig = jobs.do13_permutateJob

    T_P = np.zeros((size[0], size[1]))
    P_P = np.zeros((size[0], size[1]))

    subrange, modPerm = divmod(NOF_PERMUTATIONS, slices)

    if sliceno < modPerm:
        stop = (sliceno+1) * (subrange + 1)
        start = (subrange + 1) * sliceno
    else:
        stop = subrange*(sliceno+1) + modPerm
        start = subrange * sliceno + modPerm
    
    del subrange
    del modPerm

    arrSize = stop-start

    Plevel = np.empty(arrSize, dtype = np.ndarray)
    Ttest = np.empty(arrSize, dtype = np.ndarray)
    del arrSize

    for e in options.epoch:
        for permutation in range(start, stop):
            cdata = np.zeros((len(subjects), size[0], size[1]))
            for j, sub in enumerate(subjects):
                directory = f'./data/Visual/{sub}/8-ClassifierTesting/PermutationStudyDecodeTestVisual/'
                temp = jobbig.load(directory + f'{sub}_P{permutation+1}.pickle')[f'{sub}_P{permutation+1}']
                cdata[j] = gaussian(temp, sigma = 1, truncate = 2, preserve_range=True)


            x = np.zeros(len(subjects))
            for col in range(size[0]):
                for row in range(size[1]):
                    #Vectorization of numpy is always faster than forloop if you can spare the memory, since it's done in C++
                    x[:] = cdata[:, row, col]
                    #Above is equivalent of doing the below commented forloop.
                    #for i, sub2 in enumerate(subject):
                    #    x[i] = cdata[i, row, col]
                    #Does the same ttest in matlab with 'tail' 'both' settings, as it checks vs the alternative that it is not part of the mean 33.33
                    stat, pvalue  = ttest_1samp(x, 33.33, nan_policy = 'propagate')
                    T_P[row, col] = stat
                    P_P[row, col] = pvalue

            
            print(f'permutation {permutation} .../n')
            Plevel[permutation - start] = P_P
            Ttest[permutation - start] = T_P

    return Plevel, Ttest

def synthesis(analysis_res, job):

    #Aggregate results and save as a pickle file.
    Plevel, Ttest = zip(*analysis_res)
    
    Random_Distribution = {'Plevel': np.concatenate(Plevel, axis = 0),
                            'Ttest': np.concatenate(Ttest, axis = 0)}

    directory = f'./data/Visual/Random_Distribution/'

    os.makedirs(os.path.dirname(directory), exist_ok=True)

    job.save({'Random_Distribution': Random_Distribution}, directory + f'Random_Distribution.pickle')