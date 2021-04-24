function [model] = mvpa_traincrossvalclassifier(cfg,data)
% ---------------------------------------------------------------------------------------------------
% 
% [model] = mvpa_traincrossvalclassifier(cfg,data)
% 
% WHERE:
% DATA is an output from mvpa_datapartition
% cfg.training_algorithm
% training_algorithm = 1 - SVM with three classes, onevsall strategy and linear kernel
% training_algorithm = 2 - SVM with two classes and linear kernel
% cfg.fold = 10;
% cfg.classifiernumber = 20;
% Output model is structure with has many fields has classifiers trained
% ---------------------------------------------------------------------------------------------------


rc = length(cfg.category_model);
for c = 1:cfg.classifiernumber
    fprintf('traning classifier %d...\n', c);
    confmatfinal(rc,rc) = 0;
    for k = 1:cfg.fold
    fprintf('fold %d...\n', k);        
        if ~isempty(data.classifier{c}.fold{k}.feature_training)
            if cfg.training_algorithm == 1
                template = templateSVM('KernelFunction', 'linear', 'PolynomialOrder', [], 'KernelScale', 'auto', 'BoxConstraint', 1, 'Standardize', 1);
                ClassifierSVM = fitcecoc(data.classifier{c}.fold{k}.feature_training,data.classifier{c}.fold{k}.category_training_name,'Learners', template, 'Coding', 'onevsall', 'PredictorNames',data.classifier{c}.fold{k}.selectedfeature_name ,'ClassNames', unique(data.classifier{c}.fold{k}.category_training_name));
                noncrossvalmodel.classifier{c}.fold{k} = compact(ClassifierSVM);
            elseif cfg.training_algorithm == 2
                ClassifierSVM = fitcsvm(data.classifier{c}.fold{k}.feature_training, data.classifier{c}.fold{k}.category_training_name,'KernelFunction', 'linear', 'PolynomialOrder', [], 'KernelScale', 'auto', 'BoxConstraint', 1, 'Standardize', 1,'PredictorNames',data.classifier{c}.fold{k}.selectedfeature_name ,'ClassNames', unique(data.classifier{c}.fold{k}.category_training_name));
                noncrossvalmodel.classifier{c}.fold{k} = compact(ClassifierSVM);
            end
            model.classifier{c}.fold{k} = '';
            [label,score] = predict(noncrossvalmodel.classifier{c}.fold{k},data.classifier{c}.fold{k}.feature_test);
            [~,labelnumber] = ismember(label,cfg.category_model);
            [~,catnumber] = ismember(data.classifier{c}.fold{k}.category_test_name,cfg.category_model);
            
            confmat = confusionmat(catnumber,labelnumber);
            model.classifier{c}.fold{k}.model = noncrossvalmodel.classifier{c}.fold{k};
            model.classifier{c}.fold{k}.predict = label;
            model.classifier{c}.fold{k}.truelabel = data.classifier{c}.fold{k}.category_test_name;
            model.classifier{c}.fold{k}.feature = data.classifier{c}.fold{k}.selectedfeature_name;        
            model.classifier{c}.fold{k}.likelihood = sum(label == data.classifier{c}.fold{k}.category_test_name)/len(label);
            model.classifier{c}.fold{k}.confmat = confmat;
            
            confmat = model.classifier{c}.fold{k}.confmat;
            confmatfinal = confmatfinal + confmat;
            clear confmat
        end
    end
    model.confmatfinal{c} = confmatfinal;
    clear confmatfinal
    fprintf('classifier %d successfully trained \n', c);
end

% ---------------------------------------------------------------------------------------------------
% Bram�o, I. November 2015
% ---------------------------------------------------------------------------------------------------

