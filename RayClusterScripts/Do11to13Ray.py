from pickle import dumps
from rayModObjStore.Do11Ray import Do11
from rayModObjStore.Do12Ray import Do12
from rayModObjStore.Do13Ray import Do13
from openstack import connection
import ray
import numpy as np

#SideNotes, the ray scripts as they are written now do accept and use some of the cfg input,
#However as all the things were the same size as what the cfg stated I took some liberty with the ray scripts
#And they don't iterate over the cfg number but rather over the elements in the corresponding data structure
def main(epoch:list, size:list, NOF_PERMUTATIONS:int):
    """Executes Do11 to Do13 for a local ray server with default settings.

        epoch:list -> a list of epochs over which to iterate for Do11 to Do13

        size:list -> a list of the length for a dimensional list ie. [20,20]

        NOF_PERMUTATIONS:int -> The number of permutations to generate in Do13
    """
    #Attaches to a ray server to distribute the different jobs, the redis_password is provided when launching the ray server, the current is the standard pw at port 6739.
    ray.init(address='auto', _redis_password='5241590000000000')
    #What are the subject names we're processing?
    subjects = np.array(['Subj01', 'Subj02', 'Subj03',
                        'Subj04', 'Subj05', 'Subj06',
                        'Subj07', 'Subj08', 'Subj09',
                        'Subj11', 'Subj12', 'Subj13',
                        'Subj14', 'Subj15', 'Subj16',
                        'Subj17', 'Subj18', 'Subj19'])

    #declaring our Random_distribution we produce at the end, unecessary because it's python but good for clarity
    Random_Distribution = {'Plevel': [None] * NOF_PERMUTATIONS,
                            'Ttest': [None] * NOF_PERMUTATIONS}


    #Iterate over all wanted subjects and send the objects to the next script, saves on loading and saving.
    def oLoop(epo):
        crossval = Do11(subjects = subjects, epo = epo)
        cdata = Do12(subjects = subjects, epo = epo, crossval = crossval)
        RDist = Do13(NOF_PERMUTATIONS = NOF_PERMUTATIONS, subjects = subjects, epo = epo, size = size, permcdata = cdata, crossval = crossval)
        #this is unecessary deletions since it will fall out of scope when returning the object, but I want to clean it up to make sure memory is released
        del crossval, cdata

        return RDist


    #Outer loop is called here, to loop over the epochs.
    tempRandomDist = list(map(oLoop, epoch))

    #The final result the Random_distribution is fetched here, all the other results are at the moment saved in the object store,
    #The Whole process gets executed once ray.get() is actually called, before it is called the method calls are only scheduled and sent on their way to be calculated.
    #But once the ray.get is called it's blocked until it fetches the actual results. Any data dependencies between the methods are handled as long as you send them between the functions.
    Random_Distribution['Plevel'], Random_Distribution['Ttest'] = map(list, zip(*ray.get(tempRandomDist[0])))

    directory = f'data/Visual/{subjects[-1]}/8-ClassifierTesting/PermutationStudyDecodeTestVisual/'
    tempdict = {'Random_Distribution': Random_Distribution}

    #Create connection object to reach the object store.
    conn = connection.Connection(cloud='openstack')

    #Save the final results to the Container as well.
    conn.create_object(container = 'DroneContainer', name = directory + f'Random_Distribution.pkl', data = dumps(tempdict))

    #Clean up ray
    ray.shutdown()

if __name__ == '__main__':

    main(epoch = ['study'], size = [20, 20], NOF_PERMUTATIONS = 100)