function [data] = mvpa_datapartition(cfg,prepdata)

% This function takes the output from the mvpa_dataprep and partitions the data
% for further cross-validation procedure.
% For each partition fold a feature reduction step is performed:
% The feature Selection Step performs a One-way ANOVA for each feature.
% Only includes feature if p < 0.05
% When a fetuare is included is z-normalized!
%
% [data] = mvpa_datapartiation(k,prepdata)
%
% WHERE:
% PREPDATA is the output from the mvpa_dataprep
% CFG is a structure whit the following definitions:
%
% cfg.fold = number of partitions to be performed in the data.
%
% The OUTPUT will be a structure with the following fields:
% DATA.numclassifier	= is the number of timebins read from the data,
%                         corresponds to the number of different classifiers that will be trained.
% DATA.fold				= is the number of partitions.
% DATA.classifier       = is a structure with as many fields as timebins.
%                         In each DATA.classifier field there is a structure corresponding
%                         to each partition fold with the following information stored:
%
%                   selectedfeature_name      = a string vector containing the names of
%                                                the features that will be used for training the model
%                   category_training        = a numerical vector contacting the category of each
%                                                observation included in the training data set
%                   category_training_name   = a string vector containing the name of each category
%                                                included in the training data set
%                   category_test            = a numerical vector contacting the category of each observation
%                                                included in the test data set
%                   category_test_name       = a string vector contacting the name of each category
%                                                included in the test data set
%                   feature_training          = a matrix (Obs X feature) with the features of each observation
%                                                in the training data set
%                   feature_test              = matrix (Obs X feature) with the features of each observation
%                                                in the test data set
% 
% ---------------------------------------------------------------------------------------------------

category = prepdata.category;
category_name = prepdata.category_name;
feature_name = prepdata.feature_name';
classifiernumber = length(prepdata.feature);

for c = 1:classifiernumber
    fprintf('Starting data partition for classifier %d...\n',c);
    cp = cvpartition(category,'Kfold',cfg.fold);
    feature = prepdata.feature{c};
    for k = 1:cfg.fold
        fprintf('fold %d...\n',k);
        data.classifier{c}.fold{k}.selectedfeature_name = {};
        data.classifier{c}.fold{k}.category_training = category(training(cp,k));
        data.classifier{c}.fold{k}.category_training_name = category_name(training(cp,k));
        data.classifier{c}.fold{k}.category_test = category(test(cp,k));
        data.classifier{c}.fold{k}.category_test_name = category_name(test(cp,k));
        feature_training_all = feature(training(cp,k),:);
        feature_test_all = feature(test(cp,k),:);
        data.classifier{c}.fold{k}.feature_training = []; data.classifier{c}.fold{k}.feature_test = [];
        data.classifier{c}.fold{k}.selectedfeature_name = [];
        for a = 1:length(feature_training_all)
            p = anova1(feature_training_all(:,a),data.classifier{c}.fold{k}.category_training,'off');
            if p < 0.05
                data.classifier{c}.fold{k}.feature_training(:,1+end) = zscore(feature_training_all(:,a));
                data.classifier{c}.fold{k}.feature_test(:,1+end) = zscore(feature_test_all(:,a));
                data.classifier{c}.fold{k}.selectedfeature_name{end+1} = feature_name{a};
            end
        end
    end
    fprintf('Data for classifier %d is partitioned...\n',c);
end    

data.numclassifier = classifiernumber;
data.fold = cfg.fold;

% ---------------------------------------------------------------------------------------------------
% Bramão, I. November 2015
% ---------------------------------------------------------------------------------------------------