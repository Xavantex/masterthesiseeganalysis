# MasterThesisEEGAnalysis

Repository of code for the analysis of EEG data, with the intent of optimizing
and speeding up the compute time tremendously.

SingleThreadScripts contain the scripts used in singlethread computations
axDo uses the Accelerator framework to utilize the local cores
RayClusterScripts uses ray to accelerate computations


install the needed packages for everything by changing into the project folder and
execute ```pip3 install -e .``` or replace source with path to project.
you don't have to run in editable mode with -e but it makes all changes to the code
update on the fly if you want to make changes to the code.
it also sets up the necessary directory workdirs/dev under axDo in the project.

to use the ray cluster fetch a clouds.yaml file from your cloud provider,
fill out the necessary information as shown in the example clouds.yaml
and drop it in the project folder where SetupCluster.py and Setup.py is located or at either
~/.config/openstack or /etc/openstack if you want to be able to use the eray command
from anywhere.

to setup/tear down the openstack cluster and ray cluster simply write ```eray up/down``` in the terminal

to only setup/tear down an openstack or ray cluster use ray first ```eray openstack/ray up/down```

after setting up everthing including a ray cluster use ```eray attach```
to ssh into the ray cluster head node.

to run a script use ```eray submit 'filename'```
where filename is path to a ray python file to execute.

to get an overview of the launched ray cluster you can use ```eray dashboard``` which will host
a server locally showing the usage and total amount of cores at the specified port,
it will be printed liek ```localhost:PORT```

you can use ```eray --help```
for information on what commands you can use.

you can also use the normal ray commands listed at <https://docs.ray.io/en/master/package-ref.html>,
the ray config is Orchestration/ClusterSetup25Ray.yaml

you can specify any amount of nodes to launch with 4 cores and 8 gb memory using ```--nodes x```,
you could change the flavor of the virtual machines in Orchestration/ClusterSetup25Ray.yaml
but at the time of writing c4m8 was the largest flavor.

you can use ```--cloudName NAMEOFCLOUD``` to use another cloud setting than the default openstack cloudname

you can define the openstack clustername with ```--stackName NAMEOFSTACK``` if you want a cluster not name 
Dronestack or another cluster beside DroneStack


to upload data located in the current directory under data/ use ```eray upload```
To upload data which isn't in the data folder in this project use 
```eray upload --data DATAPATH``` to specify 
a relative path or absolute path to upload!

In order for the Cluster to work on anything it will need data first to work on.

to download the results use ```eray download```, this will store the results under results/data/

# !!!WARNING!!!

uploading .mat files assumes that the directory only contains data, it will upload all .mat files
contained in that path!
However data will be stored in a container throughout deletion and creations of clouds, so it's
a one time process.


You can create only an ssh key with ```eray key create```, but this will be replaced with a new 
key if launching a new stack unless using ```eray stack create```, this is only meant to in case
you want to manually create the key and then launch the cluster.

you can destroy ssh keys with ```eray key destory``` if you want to delete the key on the cloud,
beware that this will make you unable to login into any current cluster and you will have to
relaunch it.