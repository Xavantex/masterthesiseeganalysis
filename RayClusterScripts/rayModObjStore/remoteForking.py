import ray
from numpy import array
"""
Helper functions to fork a number of remote objects into one remote object, which is easier to use ray.get with instead of manually retrieving each object.
Especially useful in a hierarchy of remote objects, or when working on multiple levels of parallelism.
Each function take an overarching name or dictionary to use when appending or creating a datastructure of the remote objects, and then puts all those remote objects into the same structure.
"""
@ray.remote
def dictFork(name:str, *args):
    """ Creates a new dictionary with a keyname and an array of remote objects and returns one remote object reference instead of multiple.
    name -> key name for the dictionary mapped to an array of the remote objects
    dtype -> NumPy type to store the remote objects as
    *args -> the objects to put in the NumPy array
    """
    return {name: array(args)}

@ray.remote
def dictAppend(dikt:dict, name:str, *args):
    """ Appends a new keyname with an array of the remote objects, creating one remote object reference.
    dikt -> a reference to an already created dictionary
    name -> the key name for the new object in the dictionary
    dtype -> NumPy type to store the remote objects as
    *args -> the remote objects to store under the new keyname
    """
    dikt[name] = array(args)
    return dikt

@ray.remote
def arrayFuse(*args):
    """Fuses a undefined number of remote objects into one array
    *args -> the remote objects
    """
    return array(args)