function [predictedvalues] = mvpa_applycrossvalclassifier(cfg,model,data)

% cfg.fold = 10;
% cfg.classifier = 20; % How many classifiers? ex.3
% cfg.classifiernumber = 20; % Which classifier? [4 8 9]
% cfg.timebins = 115; % How many time bins?
% cfg.timebinsnumber = 3; % Which timebins? [4 8 9]
% cfg.category_predict = {'Face' 'Landmark' 'Object'};
% cfg.category_model = {'Face' 'Landmark' 'Object'};
% cfg.trials = 'selection';
% cfg.trialselection = 'trialinfo(:,3) == 1';


%selected the trails in the data that are going to be preditectd!!!!!!




% select observations based on cfg.trailinfo
if strcmp(cfg.trials,'selection')
    indiceobs = eval(sprintf('data.%s',cfg.trialselection));
    cat_name = data.category_name(indiceobs,:);
else
    cat_name = data.category_name;
    indiceobs(1:length(cat_name)) = 1;
end
[ib,~] = ismember(cat_name,cfg.category_predict);
cat_name = cat_name(logical(ib),:);

% Observation Indices for feature selection!
[indiceobs_2,~] = ismember(data.category_name,cfg.category_predict);

indiceobs = indiceobs + indiceobs_2';
[indiceobs,~] = ismember(indiceobs,2);

rc = length(cfg.category_model); %num cats in the model!
for tb = 1:cfg.timebinsnumber
%     fprintf('applying classifier to timebin %d..\n', tb);
    feature_tb = zscore(data.feature{tb}(logical(indiceobs),:)); %zscored features
    for c = 1:cfg.classifiernumber
%     fprintf('classifier %d..\n', c);
        confmatfinal(rc,rc) = 0;
        for k = 1:cfg.fold
%             fprintf('fold %d..\n', k);
            if ~ isempty (model.classifier{c}.fold{k})
                %extract feature for that cassifier and fold
                [indicefeature,~] = ismember(data.feature_name,model.classifier{c}.fold{k}.feature);
                feature = feature_tb(:,logical(indicefeature));
                % precit!!!!
                [label,score] = predict(model.classifier{c}.fold{k}.model,feature);
                % confusion matrix
                [~,labelnumber] = ismember(label,cfg.category_model);
                [~,catnumber] = ismember(cat_name,cfg.category_model);
                confmat = confusionmat(catnumber,labelnumber);
                
                if length(confmat) < length(confmatfinal)
                    id = find(ismember(cfg.category_model,label));
                    rw = find(confmat,1);
                    confmat_new(rc,rc) = 0;
                    for i = 1:length(id)
                        confmat_new(rw,id(i)) = confmat(rw,i);
                    end
                    confmat = confmat_new;
                    clear confmat_new
                end
                predictedvalues.timebin{tb}.classifier{c}.fold{k}.predict = label;
                predictedvalues.timebin{tb}.classifier{c}.fold{k}.truelabel = cat_name;
                predictedvalues.timebin{tb}.classifier{c}.fold{k}.feature = model.classifier{c}.fold{k}.feature;
                predictedvalues.timebin{tb}.classifier{c}.fold{k}.likelihood = score;
                predictedvalues.timebin{tb}.classifier{c}.fold{k}.confmat = confmat;
                clear label score
                confmatfinal = confmatfinal + confmat;
                clear confmat
            end
        end
        predictedvalues.timebin{tb}.confmatfinal{c} = confmatfinal/10;
        clear confmatfinal
        %fprintf('classifier successfully applied to timebin %d \n', tb);
    end
end