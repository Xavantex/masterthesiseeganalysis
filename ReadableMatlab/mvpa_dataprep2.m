function data = mvpa_dataprep(cfg,varargin)

% This function takes the output from the ft_freqanalysis or ft_timelockanalysis (fieldtrip function)
% and rearranges it into the right format for the MVPA classification.
%
% data = mvpa_dataprep(cfg,data1.powspctrm, data2.powspctrm,...)
%
% WHERE:
% DATA is an output from the ft_freqanalysis (data.powspctrm) organized in a 4D Matrix (Obs X Chan X Freq X Time)
% CFG is a structure whit the following definitions:
%
% cfg.dimension       = 2 if ft_timelockanalysis or 3 if ft_freqanalysis
% cfg.name            = string array with the name of each category {'Category1' 'Category2' ....}.
%                       The names should be entered in the same order as the data.powspctrm
% cfg.label           = string with the channel labels {'Channel1';'Channel2';...}.
%                       It should match the channel in the data and it should have
%                       the same order as in the data.powspctrm. 
% cfg.channels        = numerical vector with the channel numbers;
% cfg.freq            = numerical vector corresponding to the frequencies;
% cfg.timepoints      = numerical vector with the time points within each timebin;
% cfg.numtimebins     = number of timebins in the data that will be read
% cfg.startingsample  = number of the first sample in the data to be read 
% cfg.trialinfo       = 1 (yes) or 0 (no) whether or not a trialinfo matrix is included into the dataprep
%
% If 1 (yes) specify also
% cfg.specifytrialinfo = [data1.trialinfo; data2.trialinfo;....]; or a Matrix corresponding to the trial
% definition where each row corresponds to one trial and each column to a trial specification (for example, accuracy, RT, etc)
% 
% The OUTPUT will be a structure with the following fields:
% DATA.categories           = a numerical vector, corresponding to the number of each category,
%                             with as many rows as observations in the data.
% DATA.categories_name      = a string vector, corresponding to the name of each category,
%                             with as many rows as observations in the data.
% DATA.features_name        = a string vector, corresponding to the name of each feature (chan_fre_timepoint),
%                             with as many rows as features in the data.
% DATA.features             = a structure with as many fields as timebins read from the data.
%                             Each field contains a matrix (Obs X feature)
% DATA.numclassifiers       = number of timebins that were read from the data.
% ---------------------------------------------------------------------------------------------------


% Count the total number of trials
NTrials = 0;
num_obs = zeros(length(varargin));
for i = 1:length(varargin)
    num_obs{i} = size(varargin{i},1);
    NTrials = NTrials + num_obs{i};
end

% Create data.category
a = 1;
for i = 1:length(varargin)
    data.category(a:NTrials) = i;
    a = num_obs{i}+a;
end

% Create data.category_name
for i = 1:length(varargin)
    for cn = 1:length(data.category)
        if data.category(cn) == i
            data.category_name{cn} = cfg.name{i};
        end
    end
end

data.category = data.category';
data.category_name = data.category_name';

% Create data.feature_name
if cfg.dimension == 3
numfeat = 1;
for ch = 1:length(cfg.label)
    for f = 1:length(cfg.freq)
        for t = 1:length(cfg.timepoints)
            data.feature_name{numfeat} = strcat(cfg.label{ch},'_freq_',num2str(cfg.freq(f)),'_tpoint_',num2str(cfg.timepoints(t)));
            numfeat = numfeat + 1;
        end
    end
end
data.feature_name = data.feature_name';
elseif cfg.dimension == 2
numfeat = 1;
for ch = 1:length(cfg.label)
    for t = 1:length(cfg.timepoints)
        data.feature_name{numfeat} = strcat(cfg.label{ch},'_tpoint_',num2str(cfg.timepoints(t)));
        numfeat = numfeat + 1;
    end
end
data.feature_name = data.feature_name';
end

% Create the timebin matrix. Timebin is a matrix where each column 
% corresponds to each timebin and each line corresponds to the timepoint
% within each timebin (in samples)
beginning_sample = cfg.startingsample;
end_sample = cfg.startingsample + length(cfg.timepoints)-1;
temp = beginning_sample:end_sample;
timebin = zeros(length(temp), cfg.numtimebins);
for i = 1:cfg.numtimebins
    timebin(:,i) = beginning_sample:end_sample;
    beginning_sample = beginning_sample + length(cfg.timepoints);
    end_sample = end_sample + length(cfg.timepoints);
end

%trialinfo?
if cfg.trialinfo == 1
    data.trialinfo = cfg.specifytrialinfo;
end

% data.feature{timebins}

if cfg.dimension == 3
    for tb = 1:cfg.numtimebins
        fprintf('preparing data for timebin %d...\n',tb);
        raw_data = []; numobs = 1;
        for i = 1:length(varargin)
            for obs = 1:size(varargin{i},1)
                numfeat = 1;
                for ch = 1:length(cfg.channels)
                    for f = 1:length(cfg.freq)
                        for t = 1:length(cfg.timepoints)
                            raw_data(numobs,numfeat) = varargin{i}(obs,cfg.channels(ch),cfg.freq(f),timebin(t,tb));
                            numfeat = numfeat + 1;
                        end
                    end
                end
                numobs = numobs + 1;
            end
            clear obs
        end
        data.feature{tb} = raw_data;
    end
elseif cfg.dimension == 2
    for tb = 1:cfg.numtimebins
        fprintf('preparing data for timebin %d...\n',tb);
        raw_data = []; numobs = 1;
        for i = 1:length(varargin)
            for obs = 1:size(varargin{i},1)
                numfeat = 1;
                for ch = 1:length(cfg.channels)
                    for t = 1:length(cfg.timepoints)
                        raw_data(numobs,numfeat) = varargin{i}(obs,cfg.channels(ch),timebin(t,tb));
                        numfeat = numfeat + 1;
                    end
                end
                numobs = numobs + 1;
            end
            clear obs
        end
        data.feature{tb} = raw_data;
    end
end

data.numclassifiers = cfg.numtimebins;

% ---------------------------------------------------------------------------------------------------
% Bram�o, I., November, 2015
% ---------------------------------------------------------------------------------------------------

