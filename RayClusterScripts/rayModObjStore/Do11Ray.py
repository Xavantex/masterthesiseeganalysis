from mvpa.mvpa_traincrossvalclassifierOneVsRest import mvpa_traincrossvalclassifier
from mvpa.mvpa_datapartitionOptiDistri import mvpa_datapartition
from mvpa.mvpa_classifierperfOpti import mvpa_classifierperf
from openstack import connection
from io import BytesIO
from pickle import dumps
import ray
import scipy.io as sio
import numpy as np
import os

#--------------------------------------------------------------------------
# Training the Classifier
# Frequency
#--------------------------------------------------------------------------

@ray.remote
def Do11DatapartNonAmp(sub, epo):
    """Datapartition NonAmp part, 
    sub -> a subject string or similiar to process
    epo -> the epoch for the subject
    """
    #Connection Object to connect to object store
    conn = connection.Connection(cloud='openstack')
    #The container's name in the objectstore
    ContainerName = 'DroneContainer'

    #To retrieve and read the data as a binary object to read with scipy.io loadmat
    cFile = BytesIO()
    #Retrieving data from object store
    conn.get_object(ContainerName, f'Visual/{sub}/6-ClassificationData/{sub}_{epo}_data.mat', outfile = cFile)

    cdata = sio.loadmat(cFile, chars_as_strings = True, simplify_cells = True)[f'{sub}_{epo}_data']

    #Clean up our connection and close them, unecessary but good practices
    cFile.close()
    conn.close()

    #Partition of the data
    cfg = {'classifiernumber': 20, 'fold': 10}
    return mvpa_datapartition(cfg, cdata)


@ray.remote
def Do11TrainClassifierNonAmp(sub, epo, datapart):
    """Classifier Amp part, 
    sub -> a subject string or similiar to process
    epo -> the epoch for the subject
    datapart -> a datastructure with the datapartition from a datapart function
    """
    #Connection Object to connect to object store
    conn = connection.Connection(cloud='openstack')
    #The container's name in the objectstore
    ContainerName = 'DroneContainer'
    
    #Mock directory to store the data in the object store
    NonAmpDirectory = f'data/Visual/{sub}/7-ClassifierTraining/'#directory of the data
    
    #Partition of the data
    print('Data was partitioned and feature reduction for each partition is complete...')

    #Saving the data to the object store
    conn.create_object(ContainerName, NonAmpDirectory + f'{sub}_{epo}_datapart.pkl', data = dumps({f'{sub}_{epo}_datapart': datapart}))

    #Train the cross validated classifier
    cfg = {'training_algorithm': 1, 'fold': 10, 'classifiernumber': 20, 'category_model': ['Face', 'Landmark', 'Object']}
    return mvpa_traincrossvalclassifier(cfg, datapart)


@ray.remote
def Do11ClassifierPerfNonAmp(sub, epo, crossvalclass):
    """Classifier performance NonAmp part, 
    sub -> a subject string or similiar to process
    epo -> the epoch for the subject
    crossvalclass -> data structure with results from a train classifier function
    """
    #Connection Object to connect to object store
    conn = connection.Connection(cloud='openstack')
    #The container's name in the objectstore
    ContainerName = 'DroneContainer'

    #Mock directory to store the data in the object store
    NonAmpDirectory = f'data/Visual/{sub}/7-ClassifierTraining/'#directory of the data

    print('Cross Validated model successfully performed')

    #Saving the data to the object store
    conn.create_object(ContainerName, NonAmpDirectory + f'{sub}_{epo}_crossvalclass.pkl', data = dumps({f'{sub}_{epo}_crossvalclass': crossvalclass}))

    #Performance
    cfg = {'performance': 1, 'category_model': ['Face', 'Landmark', 'Object'], 'classifiernumber': 20}
    crossvalclass_performance = mvpa_classifierperf(cfg, crossvalclass)
    print('Performance calculated')

    #Saving the data to the object store
    conn.create_object(ContainerName, NonAmpDirectory + f'{sub}_{epo}_crossvalclass_performance.pkl', data = dumps({f'{sub}_{epo}_crossvalclass_performance': crossvalclass_performance}))

    #Outcommented because falls out of scope, but still good practice
    #conn.close()

    return True
            

#--------------------------------------------------------------------------
# Training the Classifier
# Amplitude
#--------------------------------------------------------------------------

#Amp Variables

@ray.remote
def Do11DatapartAmp(sub, epo):
    """Datapartition Amp part, 
    sub -> a subject string or similiar to process
    epo -> the epoch for the subject
    """
    #Connection Object to connect to object store
    conn = connection.Connection(cloud='openstack')
    #The container's name in the objectstore
    ContainerName = 'DroneContainer'

    #Creating bytesIO object to read data raw or specifically as a binary data to load in scipy.io
    cFile = BytesIO()
    #Retrieving data from object store
    conn.get_object(ContainerName, f'Visual/{sub}/6-ClassificationData/{sub}_{epo}_data_amp.mat', outfile = cFile)

    cdata = sio.loadmat(cFile, chars_as_strings = True, simplify_cells = True)[f'{sub}_{epo}_data_amp']

    #Cleanup Crew, uncessary but nice
    cFile.close()
    conn.close()

    #partition of the data
    cfg = {'classifiernumber': 40, 'fold': 10}
    return mvpa_datapartition(cfg, cdata)


@ray.remote
def Do11TrainClassifierAmp(sub, epo, datapart_amp):
    """Classifier Amp part, 
    sub -> a subject string or similiar to process
    epo -> the epoch for the subject
    datapart -> a datastructure with the datapartition from a datapart function
    """
    #Connection Object to connect to object store
    conn = connection.Connection(cloud='openstack')
    #The container's name in the objectstore
    ContainerName = 'DroneContainer'

    #Mock directory to store the data in the object store
    AmpDirectory = f'data/Visual/{sub}/7-ClassifierTraining/'

    print('Data was partitioned and feature reduction for each partition is complete')

    #Saving the data to the object store
    conn.create_object(ContainerName, AmpDirectory + f'{sub}_{epo}_datapart_amp.pkl', data = dumps({f'{sub}_{epo}_datapart_amp': datapart_amp}))

    #Train the cross validated classifier
    cfg = {'training_algorithm': 1, 'fold': 10, 'classifiernumber': 40, 'category_model': ['Face', 'Landmark', 'Object']}
    return mvpa_traincrossvalclassifier(cfg, datapart_amp)


@ray.remote
def Do11ClassifierPerfAmp(sub, epo, crossvalclass_amp):
    """Classifier performance NonAmp part, 
    sub -> a subject string or similiar to process
    epo -> the epoch for the subject
    crossvalclass -> data structure with results from a train classifier function
    """
    #Connection Object to connect to object store
    conn = connection.Connection(cloud='openstack')
    #The container's name in the objectstore
    ContainerName = 'DroneContainer'

    #Mock directory to store the data in the object store
    AmpDirectory = f'data/Visual/{sub}/7-ClassifierTraining/'

    print('Cross Validated model successfully performed!')

    #Saving the data to the object store
    conn.create_object(ContainerName, AmpDirectory + f'{sub}_{epo}_crossvalclass_amp.pkl', data = dumps({f'{sub}_{epo}_crossvalclass_amp': crossvalclass_amp}))

    #Performance
    cfg = {'performance': 1, 'category_model': ['Face', 'Landmark', 'Object'], 'classifiernumber': 40}
    crossvalclass_performance_amp = mvpa_classifierperf(cfg, crossvalclass_amp)
    print('Performance calculated!')

    #Saving the data to the object store
    conn.create_object(ContainerName, AmpDirectory + f'{sub}_{epo}_crossvalclass_performance_amp.pkl', data = dumps({f'{sub}_{epo}_crossvalclass_performance_amp': crossvalclass_performance_amp}))

    #Falls out of scope, but good practice
    #conn.close()

    return True


#Execute all of Do11
def Do11(subjects, epo):

    crossval_amp = {}
    crossval = {}
    datapart_amp = {}
    datapart = {}
    
    #Puts the individual subject references to the redis object store, this the methods using them will use the one in the redis server.
    #Instead of creating a new references each time a method calls it. Same reasoning for epo, in case in future they wanna do more than one. Same Size.
    [ray.put(sub) for sub in subjects]
    ray.put(epo)

    #Amp function definitions
    def datapartAmpLoop(sub):
        datapart_amp[sub] = Do11DatapartAmp.remote(sub = sub, epo = epo)
    
    def trainClassifierAmpLoop(sub):
        #This will call Do11Train, the amplified part for all subjects and return a ray object which is essentially a reference to an eventual result from the method.
        #The .remote is used when a function is decorated with the @ray.remote which schedules the function call to be executed on any available core on the cluster.
        crossval_amp[sub] = Do11TrainClassifierAmp.remote(sub = sub, epo = epo, datapart_amp = ray.get(datapart_amp.pop(sub)))#Could move this last, but since wanna calculate stuff we can just as well get rid of it and free up resources.

    def classifierPerfAmpLoop(sub):
        return Do11ClassifierPerfAmp.remote(sub = sub, epo = epo, crossvalclass_amp = ray.get(crossval_amp.pop(sub)))


    #No Amp function definitions
    def datapartNonAmpLoop(sub):
        datapart[sub] = Do11DatapartNonAmp.remote(sub = sub, epo = epo)

    def trainClassifierNonAmpLoop(sub):
        #returns the Do11Train non Amplifying part and saves the reference for use later another methods.
        crossval[sub] = Do11TrainClassifierNonAmp.remote(sub = sub, epo = epo, datapart = ray.get(datapart.pop(sub)))

    def classifierPerfNonAmpLoop(sub):
        crossval[sub] = ray.get(crossval[sub])#Might be clog up Ray, could try without this and the one below
        return Do11ClassifierPerfNonAmp.remote(sub = sub, epo = epo, crossvalclass = crossval[sub])


    #Amp part of code

    #map is very much like a for loop, it takes a function and uses the second argument's elements as input.
    #The reason why one would use this is because then the interpreter only have to interpret the method once and not for each loop,
    # speeding up computations a small bit. The analogue of below is "for sub in subjects: datapartAmpLoop(sub)"
    
    list(map(datapartAmpLoop, subjects))

    list(map(trainClassifierAmpLoop, subjects))
    del datapart_amp

    ampList = list(map(classifierPerfAmpLoop, subjects))
    del crossval_amp

    ray.get(ampList)
    del ampList
    
    #NoAmp part of code
    list(map(datapartNonAmpLoop, subjects))

    list(map(trainClassifierNonAmpLoop, subjects))
    del datapart

    nonAmpList = list(map(classifierPerfNonAmpLoop, subjects))

    ray.get(nonAmpList)#Might be unecessary to have this, could bottleneck the ray network

    return crossval