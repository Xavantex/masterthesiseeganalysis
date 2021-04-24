#!/usr/bin/python3
#from pathlib import Path
import subprocess
from openstack import connection
from pathlib import Path
import argparse
import os
from fileinput import FileInput
import sys

def main():
    #Arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('command', type=str, help='What command to execute, full/openstack/ray/stack')
    parser.add_argument('what', type=str, nargs='?', default=None, help='What to do up/down to create/destroy ray or openstack cluster, dashboard/attach/submit do the same as with the ray documentation')

    parser.add_argument('--nodes', '-n', type=int, help='Number of nodes')

    parser.add_argument('--cloudName', '-C', type=str, help='CloudName in clouds.yaml file in this directory')

    parser.add_argument('--stackName', '-SN', type=str, help='Set the stack name in openstack')

    parser.add_argument('--data', '-d', help='Set the directory of the data to upload')

    args = parser.parse_args()


    #Name of stack in cloud
    stackname = 'DroneStack'

    #Total number of servers 
    num_servers = 5
    #Keyname
    keyname = 'RayClusterKey'
    #PathToKey
    pathToKey = None
    #Head node IP
    rayMasterIP = None
    #IPs to drones
    dronesIP = None

    #Paths to config files
    currentDir = Path(__file__).parent.absolute()
    sshDir = Path.home() / '.ssh/config'
    openstackYAMLPath = currentDir / 'Orchestration/ClusterSetup25.yaml'
    rayYAMLPath = currentDir / 'Orchestration/ClusterSetup25Ray.yaml'

    #Container name in objectstore at openstack
    containerName = 'DroneContainer'

    #Name of stack in openstack
    cloudName = 'openstack'

    if args.cloudName:
        cloudName = args.cloudName

    if args.stackName:
        stackname = args.stackName

    #Checks if args.nodes given then sets number of servers to it
    if args.nodes:
        num_servers = args.nodes

    conn = connection.Connection(cloud=cloudName)

    def createOpenstackStack():

        containerNExist = True
        for cont in conn.list_containers():
            if cont.name == containerName:
                containerNExist = False

        if containerNExist:
            conn.create_container(containerName)

        if openstackYAMLPath.is_file():
            #Check that a stack with the name doesn't already exist
            if not conn.get_stack(stackname):
                stack = conn.create_stack(name=stackname, template_file=str(openstackYAMLPath), wait = True, key_name = keyname, num_servers = num_servers-1)
                for op in stack.outputs:
                    if op['output_key'] == 'RayMasterIP':
                        rayMasterIP = op['output_value']
                    
                    if op['output_key'] == 'RayDronesIP':
                        dronesIP = op['output_value']

            else:
                print(f'{stackname} already exists!')
                exit()
        else:
            print(f'You dont have the {openstackYAMLPath} file! pull it down from repo again')

        #Appends settings to ~/.ssh/config to manually ssh into ray cluster, this will be outside the docker environement
        flag = 0
        success = 0
        if sshDir.is_file():
            with FileInput([sshDir], inplace=True) as config:
                for line in config:
                    line = line.rstrip()
                    if 'Host RayCluster' in line:
                        flag = 1
                    if 'HostName' in line and flag == 1:
                        line = f'\tHostName {rayMasterIP}'
                        flag = 0
                        success = 1
                    print(line)

        #If the Host RayCluster is not found, will append new entry in config
        if success == 0:
            with open(sshDir, 'a+') as config:
                config.write(f'Host RayCluster\n')
                config.write('\tForwardAgent yes\n')
                config.write('\tStrictHostKeyChecking no\n')
                config.write(f'\tUserKnownHostsFile={Path.cwd()}/temp\n')
                config.write(f'\tHostName {rayMasterIP}\n')
                config.write('\tPreferredAuthentications publickey\n')
                config.write('\tUser ubuntu\n')
                config.write(f'\tIdentityFile {Path.cwd()}/{keyname}\n')

        #Sets the settings in the Ray yaml file to the specified ips and number of workers
        if Path(rayYAMLPath).is_file():
            with FileInput([str(rayYAMLPath)], inplace=True) as RayYaml:
                for line in RayYaml:
                    line = line.rstrip()
                    if line.strip().startswith('max_workers:'):
                        line = f'max_workers: {num_servers}'
                    elif line.strip().startswith('min_workers:'):
                        line = f'min_workers: {num_servers}'
                    elif line.strip().startswith('head_ip:'):
                        line = f'    head_ip: {rayMasterIP}'
                    elif line.strip().startswith('worker_ips:'):
                        line = f'    worker_ips: {dronesIP}'
                    print(line)
        else:
            print('You dont have the ./Orchestration/ClusterSetup25Ray.yaml file!')

            

    def destroyOpenstackStack():
        stack = conn.delete_stack(stackname, wait = True)
        if stack:
            print('Delete successful')
        else:
            print('Delete unsuccessful')

    def destroyRay():
        subprocess.run(['ray', 'down', '-y', str(rayYAMLPath)])

    def createRay():
        if args.nodes:
            subprocess.run(['ray', 'up', '-y', '--min-workers', num_servers, '--max-workers', num_servers, str(rayYAMLPath)])
        else:
            subprocess.run(['ray', 'up', '-y', str(rayYAMLPath)])

    def dashboardRay():
        subprocess.run(['ray', 'dashboard', str(rayYAMLPath)])

    def attachRay():
        subprocess.run(['ray', 'attach', '--tmux', str(rayYAMLPath)])

    def execRay():
        subprocess.run(['ray', 'exec', '--tmux', str(rayYAMLPath), args.what])

    def submitRay():
        #Check that argument exists and that it is an actual file
        if args.what == None or not Path(args.what).is_file():
            print('You have either not defined a file or it is does not exist!')
            exit()
        subprocess.run(['ray', 'submit', '--tmux', str(rayYAMLPath), args.what])

    def createKey():
        #Delete any potential registered key
        if not conn.get_keypair(keyname) or not Path(keyname).is_file():
            conn.delete_keypair(keyname)
            sshkey = conn.create_keypair(keyname)

            #Write new key and set appropriate permissions
            with open(keyname, 'w') as file:
                file.write(sshkey.private_key)
            os.chmod(keyname, 0o600)

    def destroyKey():
        conn.delete_keypair(keyname)

    def uploadData():
        #This assumes data is structured like /data/Visual/subjx/7-Class...
        root = currentDir / 'data/'
        
        if args.data:
            root = Path(args.data)
            if not root.is_dir():
                print('This is not a valid directory!')
                exit()

        for datapath in root.rglob('*.mat'):
            conn.create_object(containerName, str(datapath.relative_to(root)), filename=str(datapath))

    def downloadData():

        for result in conn.object_store.objects(containerName):
            if 'data/' in result.name:
                Path('./results/' + result.name).mkdir(parents=True, exist_ok=True)
                conn.get_object(container=containerName, obj=result.name, outfile='./results/' + result.name)


    #setup and create openstack cluster, key and ray cluster
    if args.command == 'up':
        createKey()
        createOpenstackStack()
        createRay()

    #Tear down raycluster, destroy the key locally and on cloud and finally tear down openstack stack
    elif args.command == 'down':
        destroyRay()
        destroyKey()
        destroyOpenstackStack()

    #Setup key and openstack stack, difference between this and stack create is that this will setup a new key to use
    elif args.command == 'openstack' and args.what == 'up':
        createKey()
        createOpenstackStack()

    #Destroy cloud and local sshkey, then destroy openstack stack
    elif args.command == 'openstack' and args.what == 'down':
        destroyKey()
        destroyOpenstackStack()

    #Just set up the ray cluster
    elif args.command == 'ray' and args.what == 'up':
        createRay()

    #Shut down ray cluster alone, will not destroy nodes on openstack
    elif args.command == 'ray' and args.what == 'down':
        destroyRay()

    #See dashboard of ray/overview of ray
    elif args.command == 'dashboard':
        dashboardRay()

    #attach and view ssh on ray cluster
    elif args.command == 'attach':
        attachRay()

    #Submit command to ray
    elif args.command == 'submit':
        submitRay()

    #create a key locally and on cloud
    elif args.command == 'key' and args.what == 'create':
        createKey()

    #destroy the RayClusterKey on cloud and locally
    elif args.command == 'key' and args.what == 'destroy':
        if input('You really want to destroy a key? you will be unable to acces cloud. y/N') == 'y':
            destroyKey()

    #Only create the openstack, will not work if the key RayClusterKey doesn't exist in the folder
    elif args.command == 'stack' and args.what == 'create':
        createOpenstackStack()

    #Only destroy an openstack stack this will not change the settings of the ray cluster, it's just faster but not doing manual cleanup of ray, might break caches
    elif args.command == 'stack' and args.what == 'destroy':
        destroyOpenstackStack()

    elif args.command == 'upload':
        uploadData()

    elif args.command == 'download':
        downloadData()

    conn.close()

if __name__ == '__main__':
    main()