import mne



class TimeLock:

    def __init__(subject, directory, epoch, condition, response):
        
        self.directory = directory
        self.subject = subject
        self.epoch = epoch
        self.condition = condition
        self.response = response


    def run():
        
        for sindex, selem in enumerate(self.subject):
            for eindex, eelem in enumerate(self.epoch):
                #--------------------------------------
                #    LOAD CLEANED DATA
                #--------------------------------------
                if eelem == 'study':
                    for cindex, celem in enumerate(self.condition):
                        cleandata = mne.io.read_raw_fieldtrip('F:\\EXJOBB\\to_xavante\\Visual\\Subj01\\3-CleanData\\Subj01_CleanData_study_FA.mat')




if __name__ =='__main__':
    info = mne.create_info(['FC5', 'P7', 'P3', 'Fz', 'PO10', 'Fp1', 'FC1', 'FC6', 'T8', 'P4', 'Iz', 'O1', 'F4', 'Pz', 'CP5', 'F8', 'Fp2', 'PO9', 'CP6', 'T7', 'FCz', 'C4', 'F7', 'C3', 'Cz', 'F3', 'O2', 'FC2', 'CP1', 'P8', 'CP2'], 512, 'eog')
    cleandata = mne.io.read_epochs_fieldtrip('F:\\EXJOBB\\to_xavante\\Visual\\Subj01\\3-CleanData\\Subj01_CleanData_study_FA.mat', info=info, data_name='Subj01_CleanData_study_FA')

    print(cleandata)