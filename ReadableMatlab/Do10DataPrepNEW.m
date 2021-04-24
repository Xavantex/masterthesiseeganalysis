%--------------------------------------------------------------------------
%            Preparing the data for training the classifier
% This script takes the data from fieldtrip and put it in a way that
%         classification is Matlab can be performed
%--------------------------------------------------------------------------
%                             FREQUENCY
%--------------------------------------------------------------------------
clear all
tic
delete(gcp('nocreate'))
ticBytes(gcp);
mpiprofile on
%Visual
directory  = parallel.pool.Constant('F:\\EXJOBB\\to_xavante\\Visual\\'); %directory of the data
subject = parallel.pool.Constant({'Subj01' 'Subj02' 'Subj03' 'Subj04' 'Subj05' 'Subj06' 'Subj07' 'Subj08' 'Subj09' 'Subj11' 'Subj12' 'Subj13' 'Subj14' 'Subj15' 'Subj16' 'Subj17' 'Subj18' 'Subj19'});
%subject = {'Subj01'};

%Verbal
% directory  = '//psysrv004/psymemlab/Projects/TAPMVPA-LTH/Verbal/'; %directory of the data
% subject = {'Subj01' 'Subj02' 'Subj06' 'Subj07' 'Subj08' 'Subj09' 'Subj10' 'Subj11'...
%     'Subj12' 'Subj14' 'Subj15' 'Subj16' 'Subj17' 'Subj18' 'Subj19' 'Subj20' 'Subj21' 'Subj22'};


epoch = parallel.pool.Constant({'study' 'test'});
response = parallel.pool.Constant({'visual'}); %if Verbal, reponse lexical

parfor j  =  1:length(subject.Value)
    cfg = []; %May break something with this line <---
    cfg.channels = 1:31;
    cfg.freq = 4:45;
    cfg.timepoints = 1:5;
    cfg.numtimebins = 20;
    cfg.startingsample = 150;
    for e = 1:length(epoch.Value)
        fprintf('Starting %s %s.../',subject.Value{j},epoch.Value{e});
        if strcmp(epoch.Value{e},'study')
            %load data
            cnameFA = sprintf('%s/%s/5-FreqData/%s_wv_%s_FA.mat',directory.Value,subject.Value{j},subject.Value{j},epoch.Value{e});
            cnameLM = sprintf('%s/%s/5-FreqData/%s_wv_%s_LM.mat',directory.Value,subject.Value{j},subject.Value{j},epoch.Value{e});
            cnameOB = sprintf('%s/%s/5-FreqData/%s_wv_%s_OB.mat',directory.Value,subject.Value{j},subject.Value{j},epoch.Value{e});
            
            cdataFA = load(cnameFA);
            cdataLM = load(cnameLM);
            cdataOB = load(cnameOB);
            cnameFA = []; cnameLM = []; cnameOB = [];

            field = fieldnames(cdataFA);
            cdataFA = cdataFA.(field{1});

            field = fieldnames(cdataLM);
            cdataLM = cdataLM.(field{1});

            field = fieldnames(cdataOB);
            cdataOB = cdataOB.(field{1});
            field = [];

            %prepare data for classifiation           
            cfg.trialinfo = 0; cfg.name = {'Face','Landmark','Object'}; cfg.dimension = 3;
            cfg.label = cdataFA.label;
            
            cdataFA = mvpa_dataprep(cfg, cdataFA.powspctrm, cdataLM.powspctrm, cdataOB.powspctrm);
            cdataLM = []; cdataOB = [];

            %save data
            cd(sprintf('%s/%s/6-ClassificationData/',directory.Value,subject.Value{j}));
            dpname = sprintf('%s_%s_data',subject.Value{j},epoch.Value{e});
            parsave(dpname, cdataFA);
            dpname = []; cdataFA = [];

        elseif strcmp(epoch.Value{e},'test')
            for r = 1:length(response)
                fprintf('Starting %s.../',response.Value{r});
                % load data
                cnameFA = sprintf('%s/%s/5-FreqData/%s_wv_%s_FA_%s.mat',directory.Value,subject.Value{j},subject.Value{j},epoch.Value{e},response.Value{r});
                cnameLM = sprintf('%s/%s/5-FreqData/%s_wv_%s_LM_%s.mat',directory.Value,subject.Value{j},subject.Value{j},epoch.Value{e},response.Value{r});
                cnameOB = sprintf('%s/%s/5-FreqData/%s_wv_%s_OB_%s.mat',directory.Value,subject.Value{j},subject.Value{j},epoch.Value{e},response.Value{r});

                cdataFA = load(cnameFA);
                cdataLM = load(cnameLM);
                cdataOB = load(cnameOB);
                cnameFA = []; cnameLM = []; cnameOB =[];

                field = fieldnames(cdataFA);
                cdataFA = cdataFA.(field{1});

                field = fieldnames(cdataLM);
                cdataLM = cdataLM.(field{1});

                field = fieldnames(cdataOB);
                cdataOB = cdataOB.(field{1});
                field = [];

                % prepare data for classifiation
                cfg.label = cdataFA.label;
                cfg.trialinfo = 1; cfg.name = {'Face','Landmark','Object'}; cfg.dimension = 3;
                cfg.specifytrialinfo = [cdataFA.trialinfo; cdataLM.trialinfo; cdataOB.trialinfo];
                cdataFA = mvpa_dataprep(cfg, cdataFA.powspctrm, cdataLM.powspctrm, cdataOB.powspctrm);
                cdataLM = []; cdataOB = [];
                % save data
                cd(sprintf('%s/%s/6-ClassificationData/',directory.Value,subject.Value{j}));
                dpname = sprintf('%s_%s_data_%s',subject.Value{j},epoch.Value{e},response.Value{r});
                parsave(dpname, cdataFA);
                dpname = []; cdataFA = [];
            end
        end
        fprintf('%s %s finished.../n',subject.Value{j},epoch.Value{e});      
    end
end


%--------------------------------------------------------------------------
%                             AMPLITUDE
%--------------------------------------------------------------------------

%Verbal
% directory  = '//psysrv004/psymemlab/Projects/TAPMVPA-LTH/Verbal/';
% subject = {'Subj01' 'Subj02' 'Subj06' 'Subj07' 'Subj08' 'Subj09' 'Subj10' 'Subj11'...
%     'Subj12' 'Subj14' 'Subj15' 'Subj16' 'Subj17' 'Subj18' 'Subj19' 'Subj20' 'Subj21' 'Subj22'};

condition = parallel.pool.Constant({'FA', 'LM', 'OB'});


parfor j  =  1:length(subject.Value)
    cfg = []; %May break something with this line <---
    cfg.channels = 1:31;
    cfg.timepoints = 1:11;
    cfg.numtimebins = 40;
    cfg.startingsample = 758;
    for e = 1:length(epoch.Value)
        fprintf('Starting %s %s.../n', subject.Value{j}, epoch.Value{e});
        if strcmp(epoch.Value{e},'study')
            % load data
            cnameFA = sprintf('%s/%s/4-TimeLockData/%s_timelock_%s_FA.mat',directory.Value,subject.Value{j},subject.Value{j},epoch.Value{e});
            cnameLM = sprintf('%s/%s/4-TimeLockData/%s_timelock_%s_LM.mat',directory.Value,subject.Value{j},subject.Value{j},epoch.Value{e});
            cnameOB = sprintf('%s/%s/4-TimeLockData/%s_timelock_%s_OB.mat',directory.Value,subject.Value{j},subject.Value{j},epoch.Value{e});

            cdataFA = load(cnameFA);
            cdataLM = load(cnameLM);
            cdataOB = load(cnameOB);

            field = fieldnames(cdataFA);
            cdataFA = cdataFA.(field{1});

            field = fieldnames(cdataLM);
            cdataLM = cdataLM.(field{1});

            field = fieldnames(cdataOB);
            cdataOB = cdataOB.(field{1});
            

            % prepare data for classifiation           
            cfg.trialinfo = 0; cfg.name = {'Face','Landmark','Object'}; cfg.dimension = 2;
            cfg.label = cdataFA.label;
            dataprep = mvpa_dataprep(cfg, cdataFA.trial, cdataLM.trial, cdataOB.trial);

            % save data
            cd(sprintf('%s/%s/6-ClassificationData/',directory.Value,subject.Value{j}));
            dpname = sprintf('%s_%s_data_amp',subject.Value{j},epoch.Value{e});
            parsave(dpname, dataprep);

        elseif strcmp(epoch.Value{e},'test')
            for r = 1:length(response.Value)
                fprintf('Starting %s.../n',response.Value{r});
                % load data
                cnameFA = sprintf('%s/%s/4-TimeLockData/%s_timelock_%s_FA_%s.mat',directory.Value,subject.Value{j},subject.Value{j},epoch.Value{e},response.Value{r});
                cnameLM = sprintf('%s/%s/4-TimeLockData/%s_timelock_%s_LM_%s.mat',directory.Value,subject.Value{j},subject.Value{j},epoch.Value{e},response.Value{r});
                cnameOB = sprintf('%s/%s/4-TimeLockData/%s_timelock_%s_OB_%s.mat',directory.Value,subject.Value{j},subject.Value{j},epoch.Value{e},response.Value{r});

                cdataFA = load(cnameFA);
                cdataLM = load(cnameLM);
                cdataOB = load(cnameOB);

                field = fieldnames(cdataFA);
                cdataFA = cdataFA.(field{1});

                field = fieldnames(cdataLM);
                cdataLM = cdataLM.(field{1});

                field = fieldnames(cdataOB);
                cdataOB = cdataOB.(field{1});
            

                % prepare data for classifiation
                cfg.label = cdataFA.label;
                cfg.trialinfo = 1; cfg.name = {'Face','Landmark','Object'}; cfg.dimension = 2;
                cfg.specifytrialinfo = [cdataFA.trialinfo; cdataLM.trialinfo; cdataOB.trialinfo];
                dataprep = mvpa_dataprep(cfg, cdataFA.trial, cdataLM.trial, cdataOB.trial);

                % save data
                cd(sprintf('%s/%s/6-ClassificationData/',directory.Value,subject.Value{j}));
                dpname = sprintf('%s_%s_data_amp_%s',subject.Value{j},epoch.Value{e},response.Value{r});
                parsave(dpname, dataprep);

            end
        end
        fprintf('%s %s finished.../n',subject.Value{j},epoch.Value{e});
    end
end

tocBytes(gcp)
toc
mpiprofile viewer
function parsave(fname, dataprep)
    newName = fname;
    S.(newName) = dataprep;
    save(fname, '-struct', 'S')
end