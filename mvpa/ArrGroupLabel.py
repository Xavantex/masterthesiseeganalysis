import numpy as np

#Takes an array that is mapped to a group of labels and seperates them into different arrays, #NUMPY Written by Lowe Erickson
#def ArrGroupLabel(array: np.ndarray, labels: np.ndarray) -> np.ndarray:
#    assert(np.shape(array)==np.shape(labels))
#    uniq = np.unique(labels)
#    new = np.empty(np.size(uniq), dtype = np.ndarray)    
#    for index, ele in enumerate(uniq):
#        new[index] = array[np.where(labels==ele)]
#
#    return new 

#Something else

def groupby_np(arr, both=True):
    n = len(arr)
    extrema = np.nonzero(arr[:-1] != arr[1:])[0] + 1
    if both:
        last_i = 0
        for i in extrema:
            yield last_i, i
            last_i = i
        yield last_i, n
    else:
        yield 0
        yield from extrema
        yield n


def labeling_groupby_np(values, labels):
    slicing = labels.argsort()
    sorted_labels = labels[slicing]
    sorted_values = values[slicing]
    del slicing
    result = []
    for i, j in groupby_np(sorted_labels, True):
        result.append(sorted_values[i:j])
    return result