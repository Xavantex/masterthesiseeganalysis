%--------------------------------------------------------------------------
%            Time-Frequency Domain using 5 cycle Morlet wavelet
%--------------------------------------------------------------------------
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
                cname = sprintf('%s/%s/3-CleanData/%s_CleanData_%s_%s.mat',directory.Value,subject.Value{j},subject.Value{j},epoch.Value{e},condition.Value{cn});
                cdata = load(cname);
                field = fieldnames(cdata);
                cdata = cdata.(field{1});
                %Filtering the data
                cfg = []; cfg.channel = {'all' '-VEOG' '-HEOG'};
                cdata = ft_preprocessing(cfg, cdata);
                % WAVELET METHOD power 1-50 Hz
                cd(sprintf('%s%s%s',directory.Value,subject.Value{j},'/5-FreqData/'))
                cfg = []; cfg.method = 'wavelet'; cfg.output = 'pow'; cfg.channel = {'all' '-VEOG' '-HEOG'};
                cfg.width = 5; cfg.foi = 1:1:50; cfg.toi = -1.5:0.009765625:2.49804687500000; cfg.keeptrials = 'yes';  cfg.pad = 'nextpow2';
                datafreqana = ft_freqanalysis(cfg, cdata);
                fa = sprintf('%s_wv_%s_%s',subject.Value{j},epoch.Value{e},condition.Value{cn});
                parsave(fa, datafreqana);
            end
        elseif strcmp(epoch.Value{e},'test')
            for cn = 1:length(condition.Value)
                for r = 1:length(response.Value)
                    cname = sprintf('%s/%s/3-CleanData/%s_CleanData_%s_%s_%s.mat',directory.Value,subject.Value{j},subject.Value{j},epoch.Value{e},condition.Value{cn},response.Value{r});
                    cdata = load(cname);
                    field = fieldnames(cdata);
                    cdata = cdata.(field{1});
                    %Filtering the data
                    cfg = []; cfg.channel = {'all' '-VEOG' '-HEOG'};
                    cdata = ft_preprocessing(cfg, cdata);
                    % WAVELET METHOD power 1-50 Hz
                    cd(sprintf('%s%s%s',directory.Value,subject.Value{j},'/5-FreqData/'))
                    cfg = []; cfg.method = 'wavelet'; cfg.output = 'pow'; cfg.channel = {'all' '-VEOG' '-HEOG'};
                    cfg.width = 5; cfg.foi = 1:1:50; cfg.toi = -1.5:0.009765625:2.49804687500000; cfg.keeptrials = 'yes'; cfg.pad = 'nextpow2';
                    datafreqana = ft_freqanalysis(cfg, cdata);
                    fa = sprintf('%s_wv_%s_%s_%s',subject.Value{j},epoch.Value{e},condition.Value{cn},response.Value{r});
                    parsave(fa, datafreqana);
                end
            end
        end
        %----------------------------------------------------------------------
        % Clear from memory variables from the current subject
        %----------------------------------------------------------------------
        %eval(sprintf('%s','clear -regexp ^',subject{j}))
    end
end

tocBytes(gcp)
toc
mpiprofile viewer
function parsave(fname, datafreqana)
    newName = fname;
    S.(newName) = datafreqana;
    save(fname, '-struct', 'S')
end