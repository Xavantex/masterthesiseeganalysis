import os
os.environ["OMP_NUM_THREADS"] = "1"

#Build scripts for all 3 scripts.
#They execute the functions and their dependencies,
#Taking into account if the source code have been changed.
#Makes it easier to know if something has been executed and when etc.

def main(urd):
    epoch = ['study']

    subjects = ['Subj01', 'Subj02', 'Subj03',
                'Subj04', 'Subj05', 'Subj06',
                'Subj07', 'Subj08', 'Subj09',
                'Subj11', 'Subj12', 'Subj13',
                'Subj14', 'Subj15', 'Subj16',
                'Subj17', 'Subj18', 'Subj19']

    size = [20, 20]
    NOF_PERMUTATIONS = 100

    urd.build('AxDo11Amp', subjects = subjects, epoch = epoch)
    #Reference to this job so that it can be referenced in another script for example, which the accelerator will know if it has been executed already or not.
    NonAmpJob = urd.build('AxDo11NonAmp', subjects = subjects, epoch = epoch)
    urd.build('AxDo12', do11Job = NonAmpJob, subjects = subjects, epoch = epoch)
    NonAmpJob = urd.build('AxDo13_permutate', do11Job = NonAmpJob, subjects = subjects, epoch = epoch, size = size, NOF_PERMUTATIONS = NOF_PERMUTATIONS)
    urd.build('AxDo13_ttest', do13_permutateJob = NonAmpJob, subjects = subjects, epoch = epoch, size = size, NOF_PERMUTATIONS = NOF_PERMUTATIONS)

    #Prints the execution time.
    urd.joblist.print_exectimes()
