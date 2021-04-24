'''
These are basic unittests I cooked up, they are by no means good. They are intended to check that the data I give out give similiar results between revisions.
They aren't built for the thought core functionality of the functions or they are supposed to behave, while the pythonf functions mimic the behaviour of the matlab scripts
there will always be small differences and since as I'm not always 100% sure what the thoughts behind the functions are.
'''


from mvpa.ArrGroupLabel import labeling_groupby_np
from mvpa.mvpa_datapartitionOpti import mvpa_datapartition
from mvpa.mvpa_traincrossvalclassifierOpti import mvpa_traincrossvalclassifier
from mvpa.mvpa_classifierperfOpti import mvpa_classifierperf
from mvpa.mvpa_applycrossvalclassifierOptiOneLookupDict import mvpa_applycrossvalclassifier
from SingleThread.Do11TrainClassifier import Do11TrainClassifierNonAmp
from scipy.io import loadmat
from pickle import load
import numpy as np
import pytest

#Variables
datapart = None
crossdata = None

cfg = {'classifiernumber': 2, 'fold': 5}
testdata = loadmat('./testData/TestData_study_data.mat', chars_as_strings = True, simplify_cells = True)['testData_study_data']
datapart = mvpa_datapartition(cfg, testdata, random_state = 0)

cfg = {'training_algorithm': 1, 'fold': 5, 'classifiernumber': 2, 'category_model': ['Face', 'Landmark', 'Object']}
crossdata = mvpa_traincrossvalclassifier(cfg, datapart)

cfg = {'performance': 1, 'category_model': ['Face', 'Landmark', 'Object'], 'classifiernumber': 2}
crossPerf = mvpa_classifierperf(cfg, crossdata)


cdata = loadmat(f'./testData/TestData_test_data_visual.mat',
                chars_as_strings = True, simplify_cells = True)[f'Subj01_test_data_visual']
cfg = {'fold': 5,
    'classifiernumber': 2,
    'timebinsnumber': 3,
    'category_predict': np.array(['Face', 'Landmark', 'Object']),
    'trials': 'all',
    'category_model': np.array(['Face', 'Landmark', 'Object'])}

applyCross = mvpa_applycrossvalclassifier(cfg = cfg, model = crossdata, data = cdata)

cfg = {'performance': 2,
        'category_model': np.array(['Face', 'Landmark', 'Object']),
        'category_predict': np.array(['Face', 'Landmark', 'Object']),
        'classifiernumber': 2,
        'timebinsnumber': 3}

crossPerf2 = mvpa_classifierperf(cfg, applyCross)

def asserting_dataparts(datapart, TrueDatapart):
    assert datapart.keys() == TrueDatapart.keys()
    assert datapart['fold'] == TrueDatapart['fold']
    assert datapart['numclassifier'] == TrueDatapart['numclassifier']
    assert np.shape(datapart['classifier']) == np.shape(TrueDatapart['classifier'])
    for idx, classy in enumerate(datapart['classifier']):
        assert np.shape(classy['fold']) == np.shape(TrueDatapart['classifier'][idx]['fold'])
        for idx2, fold in enumerate(classy['fold']):
            print(fold['selectedfeature_name'])
            print(TrueDatapart['classifier'][idx]['fold'][idx2]['selectedfeature_name'])
            assert np.all(fold['selectedfeature_name'] == TrueDatapart['classifier'][idx]['fold'][idx2]['selectedfeature_name'])
            assert np.all(fold['category_training'] == TrueDatapart['classifier'][idx]['fold'][idx2]['category_training'])
            assert np.all(fold['category_training_name'] == TrueDatapart['classifier'][idx]['fold'][idx2]['category_training_name'])
            assert np.all(fold['category_test'] == TrueDatapart['classifier'][idx]['fold'][idx2]['category_test'])
            assert np.all(fold['category_test_name'] == TrueDatapart['classifier'][idx]['fold'][idx2]['category_test_name'])
            assert np.all(fold['feature_training'] == TrueDatapart['classifier'][idx]['fold'][idx2]['feature_training'])
            assert np.all(fold['feature_test'] == TrueDatapart['classifier'][idx]['fold'][idx2]['feature_test'])
            print(fold['feature_test'].dtype)
            print(TrueDatapart['classifier'][idx]['fold'][idx2]['feature_test'].dtype)
            print(fold['feature_test'].__sizeof__())
            print(TrueDatapart['classifier'][idx]['fold'][idx2]['feature_test'].__sizeof__())

def asserting_random_dataparts(datapart, TrueDatapart):
    assert datapart.keys() == TrueDatapart.keys()
    assert datapart['fold'] == TrueDatapart['fold']
    assert datapart['numclassifier'] == TrueDatapart['numclassifier']
    assert np.shape(datapart['classifier']) == np.shape(TrueDatapart['classifier'])
    for idx, classy in enumerate(datapart['classifier']):
        assert np.shape(classy['fold']) == np.shape(TrueDatapart['classifier'][idx]['fold'])
        for idx2, fold in enumerate(classy['fold']):
            print(fold['selectedfeature_name'])
            print(TrueDatapart['classifier'][idx]['fold'][idx2]['selectedfeature_name'])
            if len(fold['selectedfeature_name']) == len(TrueDatapart['classifier'][idx]['fold'][idx2]['selectedfeature_name']):
                assert np.all(fold['selectedfeature_name'] != TrueDatapart['classifier'][idx]['fold'][idx2]['selectedfeature_name'])
                
            if len(fold['category_training']) == len(TrueDatapart['classifier'][idx]['fold'][idx2]['category_training']):
                assert np.all(fold['category_training'] != TrueDatapart['classifier'][idx]['fold'][idx2]['category_training'])
                assert np.all(fold['category_training_name'] != TrueDatapart['classifier'][idx]['fold'][idx2]['category_training_name'])
                assert np.all(fold['category_test'] != TrueDatapart['classifier'][idx]['fold'][idx2]['category_test'])
                assert np.all(fold['category_test_name'] != TrueDatapart['classifier'][idx]['fold'][idx2]['category_test_name'])
                
            if len(fold['feature_training']) == len(TrueDatapart['classifier'][idx]['fold'][idx2]['feature_training']):
                assert np.all(fold['feature_training'] != TrueDatapart['classifier'][idx]['fold'][idx2]['feature_training'])
                assert np.all(fold['feature_test'] != TrueDatapart['classifier'][idx]['fold'][idx2]['feature_test'])

def asserting_crossvals(crossdata, trueCrossdata):
    assert crossdata.keys() == trueCrossdata.keys()
    for idx, conf in enumerate(crossdata['confmatfinal']):
        assert np.array_equal(conf, trueCrossdata['confmatfinal'][idx])
    assert np.shape(crossdata['classifier']) == np.shape(trueCrossdata['classifier'])
    for idx, classy in enumerate(crossdata['classifier']):
        assert classy.keys() == trueCrossdata['classifier'][idx].keys()
        for idx2, fold in enumerate(classy['fold']):
            assert fold.keys() == trueCrossdata['classifier'][idx]['fold'][idx2].keys()
            assert np.array_equal(fold['predict'], trueCrossdata['classifier'][idx]['fold'][idx2]['predict'])
            assert np.array_equal(fold['truelabel'], trueCrossdata['classifier'][idx]['fold'][idx2]['truelabel'])
            assert np.array_equal(fold['feature'], trueCrossdata['classifier'][idx]['fold'][idx2]['feature'])
            assert np.array_equal(fold['likelihood'], trueCrossdata['classifier'][idx]['fold'][idx2]['likelihood'])
            assert np.array_equal(fold['confmat'], trueCrossdata['classifier'][idx]['fold'][idx2]['confmat'])

def asserting_classifierPerf(crossPerf, trueCrossPerf):
    assert crossPerf.keys() == trueCrossPerf.keys()
    for idx, classy in enumerate(crossPerf['classifier']):
        assert classy == trueCrossPerf['classifier'][idx]

def asserting_applycrossval(applyCross, trueApplyCross):
    assert trueApplyCross.keys() == applyCross.keys() 
    for idx, bin in enumerate(trueApplyCross['timebin']):
        trueBin = applyCross['timebin'][idx]['classifier']
        assert np.shape(bin['classifier']) == np.shape(trueBin)
        for idxconf, conf in enumerate(bin['confmatfinal']):
            assert np.array_equal(conf, applyCross['timebin'][idx]['confmatfinal'][idxconf])
        assert bin.keys() == applyCross['timebin'][idx].keys()
        for idx2, classy in enumerate(bin['classifier']):
            trueClass = trueBin[idx2]
            assert classy.keys() == trueClass.keys()
            for idx3, fold in enumerate(classy['fold']):
                trueFold = trueClass['fold'][idx3]
                assert fold.keys() == trueFold.keys()
                assert np.array_equal(fold['predict'], trueFold['predict'])
                assert np.array_equal(fold['truelabel'], trueFold['truelabel'])
                assert np.array_equal(fold['feature'], trueFold['feature'])
                assert np.array_equal(fold['likelihood'], trueFold['likelihood'])
                assert np.array_equal(fold['confmat'], trueFold['confmat'])

def asserting_classifierPerf2(crossPerf2, trueCrossPerf2):
    assert crossPerf2.keys() == trueCrossPerf2.keys()
    for idx, bin in enumerate(crossPerf2['timebin']):
        trueBin = trueCrossPerf2['timebin'][idx]
        assert np.shape(bin) == np.shape(trueBin)
        for idx2, classy in enumerate(bin['classifier']):
            assert classy == trueBin['classifier'][idx2]


def test_labeling_groupby_np():
    a = np.array([1,2,3,4,5,6,7,8,9])
    b = np.array([1,2,3,1,2,3,1,2,3])
    a,b,c = labeling_groupby_np(a,b)
    assert np.all(a == np.array([1,4,7]))
    assert np.all(b == np.array([2,5,8]))
    assert np.all(c == np.array([3,6,9]))
    a = np.array(['face','object', 'object', 'object', 'landmark', 'face', 'landmark', 'landmark', 'face'])
    b = np.array([1,2,3,4,5,6,7,8,9])
    a,b,c = labeling_groupby_np(b,a)
    assert np.all(a == np.array([1,6,9]))
    assert np.all(c == np.array([2,3,4]))
    assert np.all(b == np.array([5,7,8]))

def test_mvpa_datapartition():
    with open('./testData/TestDatapart.pkl', 'rb') as file:
        TrueDatapart = load(file)

    asserting_dataparts(datapart, TrueDatapart)

def test_mvpa_trainCrossvalClassifier():

    with open('./testData/testCrossdata.pkl', 'rb') as file:
        trueCrossdata = load(file)

    asserting_crossvals(crossdata, trueCrossdata)

def  test_mvpa_classifierperf1():
    with open('./testData/testCrossPerf.pkl', 'rb') as file:
        trueCrossPerf = load(file)

    asserting_classifierPerf(crossPerf, trueCrossPerf)
    
def test_mvpa_applycrossvalclassifier():#BELOW NEEDS FIXING NOW NOT REALLY A TEST
    
    with open('./testData/testApplyCross.pkl', 'rb') as file:
        trueApplyCross = load(file)#give classifier

    asserting_applycrossval(applyCross, trueApplyCross)


def  test_mvpa_classifierperf2():
    with open('./testData/testCrossPerf2.pkl', 'rb') as file:
        trueCrossPerf2 = load(file)

    asserting_classifierPerf2(crossPerf2, trueCrossPerf2)

    


@pytest.mark.slow
def test_Do11TrainClassifier():
    Do11TrainClassifierNonAmp(random_state = 0)

    directory = f'/data/Visual/Subj01/7-ClassifierTraining/'

    with open('.' + directory + f'Subj01_study_datapart.pkl', 'rb') as file:
        datapart = load(file)['Subj01_study_datapart']


    with open('./testData' + directory + f'Subj01_study_datapart.pkl', 'rb') as file:
        truDatapart = load(file)['Subj01_study_datapart']


    asserting_dataparts(datapart, truDatapart)

    del datapart, truDatapart

    with open('.' + directory + f'Subj01_study_crossvalclass.pkl', 'rb') as file:
        crossdata = load(file)['Subj01_study_crossvalclass']


    with open('./testData' + directory + f'Subj01_study_crossvalclass.pkl', 'rb') as file:
        truCrossdata = load(file)['Subj01_study_crossvalclass']

    asserting_crossvals(crossdata, truCrossdata)

    del crossdata, truCrossdata

    with open('.' + directory + f'Subj01_study_crossvalclass_performance.pkl', 'rb') as file:
        crossperf = load(file)['Subj01_study_crossvalclass_performance']


    with open('./testData' + directory + f'Subj01_study_crossvalclass_performance.pkl', 'rb') as file:
        truCrossPerf2 = load(file)['Subj01_study_crossvalclass_performance']

    asserting_classifierPerf(crossperf, truCrossPerf2)

@pytest.mark.slow
def test_RandomNess_Do11TrainClassifier():
    Do11TrainClassifierNonAmp()

    directory = f'/data/Visual/Subj01/7-ClassifierTraining/'

    with open('.' + directory + f'Subj01_study_datapart.pkl', 'rb') as file:
        datapart = load(file)['Subj01_study_datapart']


    with open('./testData' + directory + f'Random_study_datapart.pkl', 'rb') as file:
        truDatapart = load(file)['Subj01_study_datapart']


    asserting_random_dataparts(datapart, truDatapart)

    del datapart, truDatapart
    '''
    with open('.' + directory + f'Subj01_study_crossvalclass.pkl', 'rb') as file:
        crossdata = load(file)['Subj01_study_crossvalclass']


    with open('./testData' + directory + f'Random_study_crossvalclass.pkl', 'rb') as file:
        truCrossdata = load(file)['Subj01_study_crossvalclass']

    asserting_crossvals(crossdata, truCrossdata)

    del crossdata, truCrossdata

    with open('.' + directory + f'Subj01_study_crossvalclass_performance.pkl', 'rb') as file:
        crossperf = load(file)['Subj01_study_crossvalclass_performance']


    with open('./testData' + directory + f'Random_study_crossvalclass_performance.pkl', 'rb') as file:
        truCrossPerf2 = load(file)['Subj01_study_crossvalclass_performance']

    asserting_classifierPerf(crossperf, truCrossPerf2)
    '''