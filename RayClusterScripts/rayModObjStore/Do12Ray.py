from mvpa.mvpa_applycrossvalclassifierOptiOneLookupDict import mvpa_applycrossvalclassifier
from mvpa.mvpa_classifierperfOpti import mvpa_classifierperf
from openstack import connection
from io import BytesIO
from pickle import dumps
import ray
import scipy.io as sio
#import os
import numpy as np

#------------------------------------------------------------------------------
#  TEST THE CROSS VALIDATED CLASSIFIER ON THE RETRIEVAL DATA
#------------------------------------------------------------------------------
#  USE DATA FROM visual CLASSIFIED TRIALS - FREQUENCY
#------------------------------------------------------------------------------
@ray.remote
def Do12TestClassifier(sub, epo, crossdata):
    """Do12 Clsasifier testing
    sub -> the subject to be tested
    epo -> the epoch the subject's in
    crossdata -> data structure with classifier of Do11
    """
    print(sub)
    #Connection object to talk with openstack
    conn = connection.Connection(cloud='openstack')

    ContainerName = 'DroneContainer'

    #Mock Directory to load from object store.
    directory = f'data/Visual/{sub}/8-ClassifierTesting/'


    #Retrieve data as a raw binary data.
    cFile = BytesIO()
    #Retrieve data from objectstore
    conn.get_object(ContainerName, f'Visual/{sub}/6-ClassificationData/{sub}_test_data_visual.mat', outfile = cFile)

    cdata = sio.loadmat(cFile, chars_as_strings = True, simplify_cells = True)[f'{sub}_test_data_visual']

    #Clean up after ourselves so we don't destroy stuff and free memory
    cFile.close()

    cfg = {'fold': 10,
            'classifiernumber': 20,
            'timebinsnumber': 20,
            'category_predict': np.array(['Face', 'Landmark', 'Object']),
            'trials': 'all',
            'category_model': np.array(['Face', 'Landmark', 'Object'])}

    #Apply calssifier into the correct visual/ correct visual trials
    predtest_visual = mvpa_applycrossvalclassifier(cfg = cfg, model = crossdata, data = cdata)

    #Store data in object store
    conn.create_object(ContainerName, directory + f'{sub}_{epo}_predtest_visual.pkl', data = dumps({f'{sub}_{epo}_predtest_visual': predtest_visual}))

    cfg = {'performance': 2,
            'category_model': np.array(['Face', 'Landmark', 'Object']),
            'category_predict': np.array(['Face', 'Landmark', 'Object']),
            'classifiernumber': 20,
            'timebinsnumber': 20}

    predtest_visual_performance = mvpa_classifierperf(cfg, predtest_visual)#Performance

    #Store data in object store
    conn.create_object(ContainerName, directory + f'{sub}_{epo}_predtest_visual.pkl', data = dumps({f'{sub}_{epo}_predtest_visual_performance': predtest_visual_performance}))

    #Cleanup Crew reporting for duty!
    conn.close()

    print('Performance calculated')

    return cdata


def Do12(subjects, epo, crossval):
    """Executes Do12, iterates over a list of subjects and takes a list of classifiers from Do11.
    subjects -> the subjects to iterate over
    epo -> the epoch currently processing
    crossval -> a dictionary of the classifiers from do11 mapped to each subject in subjects
    """
    cdata = {}

    #Puts the individual subject references to the redis object store, this the methods using them will use the one in the redis server.
    #Instead of creating a new references each time a method calls it. Same reasoning for epo, in case in future they wanna do more than one. Same Size.
    [ray.put(sub) for sub in subjects]
    ray.put(epo)

    def testClassifierLoop(sub):
        cdata[sub] = Do12TestClassifier.remote(sub = sub, epo = epo, crossdata = crossval[sub])


    list(map(testClassifierLoop, subjects))

    return cdata