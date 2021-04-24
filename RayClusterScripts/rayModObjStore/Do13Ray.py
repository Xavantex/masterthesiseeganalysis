from mvpa.mvpa_applycrossvalclassifierOptiOneLookupDict import mvpa_applycrossvalclassifier
from openstack import connection
from rayModObjStore.remoteForking import arrayFuse
import ray
import scipy.io as sio
import scipy.stats as pystats
import skimage.filters as filter
import numpy as np
import pickle
import os

#------------------------------------------------------------------------------
#  PERMUTATION FOR EACH PARTICIPANT!!!! FREQUENCY
#------------------------------------------------------------------------------
@ray.remote
def Do13TestClassifier_ShuffledLabels_permutation(sub, epo, perm, size, cdata, crossdata):
    """Do13 shuffling labels, create a permutation from old data
    sub -> current subject to process
    epo -> current epoch
    perm -> current permutation number
    size -> list with dimension of size ie. [20,20]
    cdata -> subject test data, classificationdata from 6-ClassificationData
    crossdata -> classifier from Do11
    """

    #Connection object to talk with openstack
    conn = connection.Connection(cloud='openstack')

    #Openstack container name
    ContainerName = 'DroneContainer'

    #Create permutation by reshuffling old data randomly
    R = np.random.permutation(len(cdata['category_name']))
    per = {'category_name': cdata['category_name'][R],
        'category': cdata['category'][R],
        'feature_name': cdata['feature_name'],
        'trial_info': cdata['trialinfo'],
        'feature': cdata['feature'],
        'numclassifiers': cdata['numclassifiers']}

    #Apply classifier into the shuffled category trials
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

    #Create mock name and save to objectstore in openstack
    filename = f'data/Visual/{sub}/8-ClassifierTesting/PermutationStudyDecodeTestVisual/{sub}_P{perm+1}'
    conn.create_object(ContainerName, filename, data = pickle.dumps(sub_P))

    #Clean upp connectionobject, not necessary but good practice and potentially less problems
    conn.close()

    #Apply gaussian filter before returning sub_P
    return filter.gaussian(sub_P, sigma = 1, truncate = 2, preserve_range=True)


#------------------------------------------------------------------------------
# For each permutation a T-Test is calculated
# (classification accuracy is compared against chance (33.33))
# Each T test is stored in a Struture called Random_Distribution
# _____________________________________________________________________________
# 
# Dependent on Do13TestClassifier_ShuffledLabels_permutation
#
#------------------------------------------------------------------------------ 
@ray.remote
def Do13TestClassifier_ShuffledLabels_ttest(perm, cdata, size):
    """Ttest of new data,
    perm -> Current permutation number
    cdata -> the newly created data, permutation
    size -> list with dimension of size ie. [20,20]
    """
    x = np.zeros(len(cdata))

    T_P = np.zeros((size[0], size[1]))
    P_P = np.zeros((size[0], size[1]))
    
    for col in range(size[0]):
        for row in range(size[1]):
            #Vectorization of numpy is always faster than forloop if you can spare the memory, since it's done in C++
            x[:] = cdata[:, row, col]

            stat, pvalue  = pystats.ttest_1samp(x, 33.33, nan_policy = 'propagate')
            T_P[row, col] = stat
            P_P[row, col] = pvalue

    
    print(f'permutation {perm} .../n')
    return P_P, T_P



def Do13(NOF_PERMUTATIONS, subjects, epo, size, permcdata, crossval):
    """Executes Do13,
    NOF_PERMUTATIONS -> the number of randomly generated permutations to make
    subjects -> the subject list to iterate over
    epo -> the current epoch
    size -> list with dimension of size ie. [20,20]
    permcdata -> subject test data, classificationdata from 6-ClassificationData
    crossval -> classifier from Do11
    """
    tempRandomDist = []

    #Puts the individual subject references to the redis object store, this the methods using them will use the one in the redis server.
    #Instead of creating a new references each time a method calls it. Same reasoning for epo, in case in future they wanna do more than one. Same Size.
    [ray.put(sub) for sub in subjects]
    ray.put(epo)
    ray.put(size)

    def permutationLoop(perm):
        ray.put(perm)
        def subjectsLoop(sub):
            #Return permutations of the data generated in Do12TestClassifier
            return Do13TestClassifier_ShuffledLabels_permutation.remote(sub = sub, epo = epo, perm = perm, size = size,
                                                                        cdata = permcdata[sub], crossdata = crossval[sub])

        #We save the permutations from Do13TestClassifier_ShuffledLabels_permutations.remote() in ttestcdata, and use them finally in Do13TestClassifier....ttest.remote()
        ttestcdata = arrayFuse.remote(*list(map(subjectsLoop, subjects)))
        tempRandomDist.append(Do13TestClassifier_ShuffledLabels_ttest.remote(perm = perm, cdata = ttestcdata, size = size))

    #The loop over the number of permutations are called here
    list(map(permutationLoop, range(NOF_PERMUTATIONS)))

    return tempRandomDist