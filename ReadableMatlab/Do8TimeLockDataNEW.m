%--------------------------------------------------------------------------
%                           TIME LOCK DATA 
%--------------------------------------------------------------------------
clear all
tic
delete(gcp('nocreate'))
ticBytes(gcp);
mpiprofile on
%Visual
directory  = parallel.pool.Constant('F:\\EXJOBB\\to_xavante\\Visual\\'); %directory of the data
subject = parallel.pool.Constant({'Subj01' 'Subj02' 'Subj03' 'Subj04' 'Subj05' 'Subj06' 'Subj07' 'Subj08' 'Subj09'...
    'Subj11' 'Subj12' 'Subj13' 'Subj14' 'Subj15' 'Subj16' 'Subj17' 'Subj18' 'Subj19'});

%Verbal
% directory  = '/Users/pex/phd/matlab/200316_play/Verbal/'; %directory of the data
% subject = {'Subj01' 'Subj02' 'Subj06' 'Subj07' 'Subj08' 'Subj09' 'Subj10' 'Subj11'...
%     'Subj12' 'Subj14' 'Subj15' 'Subj16' 'Subj17' 'Subj18' 'Subj19' 'Subj20' 'Subj21' 'Subj22'};


epoch = parallel.pool.Constant({'study' 'test'});
condition = parallel.pool.Constant({'FA', 'LM', 'OB'});
response = parallel.pool.Constant({'visual'}); %if Verbal, reponse lexical


parfor j  =  1:length(subject.Value)
    for e = 1:length(epoch.Value)
        %----------------------------------------------------------------------
        % LOAD CLEANED DATA
        %----------------------------------------------------------------------
        if strcmp(epoch.Value{e},'study')
            for cn = 1:length(condition.Value)
                cname = sprintf('%s%s/3-CleanData/%s_CleanData_%s_%s.mat',directory.Value,subject.Value{j},subject.Value{j},epoch.Value{e},condition.Value{cn});
                cdata = load(cname);
                field = fieldnames(cdata);
                cdata = cdata.(field{1});
                %Filtering the data
                cfg = []; cfg.channel = {'all' '-VEOG' '-HEOG'}; cfg.demean = 'yes'; cfg.baselinewindow = [-0.2 0]; cfg.lpfilter = 'yes';  cfg.lpfreq = 30;
                cdata = ft_preprocessing(cfg, cdata);
                % Time Lock Data
                cd(sprintf('%s%s%s',directory.Value,subject.Value{j},'/4-TimeLockData/'))
                cfg = []; cfg.channel = {'all' '-VEOG' '-HEOG'}; cfg.keeptrials = 'yes';
                datatimelock = ft_timelockanalysis(cfg, cdata);
                tlname = sprintf('%s_timelock_%s_%s',subject.Value{j},epoch.Value{e},condition.Value{cn});
                parsave(tlname, datatimelock);
            end
        elseif strcmp(epoch.Value{e},'test')
           for cn = 1:length(condition.Value)
                for r = 1:length(response.Value)
                    cname = sprintf('%s/%s/3-CleanData/%s_CleanData_%s_%s_%s.mat',directory.Value,subject.Value{j},subject.Value{j},epoch.Value{e},condition.Value{cn},response.Value{r});
                    cdata = load(cname);
                    field = fieldnames(cdata);
                    cdata = cdata.(field{1});
                    %Filtering the data
                    cfg = []; cfg.channel = {'all' '-VEOG' '-HEOG'}; cfg.demean = 'yes'; cfg.baselinewindow = [-0.2 0]; cfg.lpfilter = 'yes';  cfg.lpfreq = 30;
                    cdata = ft_preprocessing(cfg, cdata);
                    % Time Lock Data
                    cd(sprintf('%s%s%s',directory.Value,subject.Value{j},'/4-TimeLockData/'))
                    cfg = []; cfg.channel = {'all' '-VEOG' '-HEOG'}; cfg.keeptrials = 'yes';
                    datatimelock = ft_timelockanalysis(cfg, cdata);
                    tlname = sprintf('%s_timelock_%s_%s_%s',subject.Value{j},epoch.Value{e},condition.Value{cn},response.Value{r});
                    parsave(tlname, datatimelock);
                end
            end
        end
    end
    %----------------------------------------------------------------------
    % Clear from memory variables from the current subject
    %----------------------------------------------------------------------
    %eval(sprintf('%s','clear -regexp ^',subject{j}))    
end

tocBytes(gcp)
toc
mpiprofile viewer
function parsave(fname, datatimelock)
    newName = fname;
    S.(newName) = datatimelock;
    save(fname, '-struct', 'S')
end