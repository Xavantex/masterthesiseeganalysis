function [performance] = mvpa_classifierperf(cfg,data)

% cfg.performance = 1,2,...
% cfg.categories = {'Faces' 'Landmarks' 'Objects'};
% 1 - classifier
% 2 - timebins
% cfg.classifiernumber = 20;
% cfg.timebinsnumber = 20;



if cfg.performance == 1
    for c = 1:cfg.classifiernumber
        NumTrialTotal = sum(sum(data.confmatfinal{c}));
        for cat = 1:length(cfg.category_model)
            eval(sprintf('category(%d) = data.confmatfinal{c}(%d,%d);',cat,cat,cat))
            eval(sprintf('performance.classifier{c}.%s = (data.confmatfinal{c}(%d,%d)/NumTrialTotal)*100;',cfg.category_model{cat},cat,cat))
        end
        performance.classifier{c}.Total = (sum(category)/NumTrialTotal)*100;
    end
elseif cfg.performance == 2
    for tb = 1:cfg.timebinsnumber
        for c = 1:cfg.classifiernumber
            NumTrialTotal = sum(sum(data.timebin{tb}.confmatfinal{c}));
            for cat = 1:length(cfg.category_model)
                eval(sprintf('category(%d) = data.timebin{tb}.confmatfinal{c}(%d,%d);',cat,cat,cat))
                eval(sprintf('performance.timebin{tb}.classifier{c}.%s = (data.timebin{tb}.confmatfinal{c}(%d,%d)/NumTrialTotal)*100;',cfg.category_model{cat},cat,cat))
            end
            performance.timebin{tb}.classifier{c}.Total = (sum(category)/NumTrialTotal)*100;
        end
    end
elseif cfg.performance == 3
    for tb = 1:cfg.timebinsnumber
        for c = 1:cfg.classifiernumber
            NumTrialTotal = sum(sum(data.timebin{tb}.confmatfinal{c}));
            if strcmp(cfg.category_predict,'Face')
                performance.timebin{tb}.classifier{c}.Face = (data.timebin{tb}.confmatfinal{c}(1,1)/NumTrialTotal)*100;
                performance.timebin{tb}.classifier{c}.Landmark = (data.timebin{tb}.confmatfinal{c}(1,2)/NumTrialTotal)*100;
                performance.timebin{tb}.classifier{c}.Object = (data.timebin{tb}.confmatfinal{c}(1,3)/NumTrialTotal)*100;
            elseif strcmp(cfg.category_predict,'Landmark')
                performance.timebin{tb}.classifier{c}.Face = (data.timebin{tb}.confmatfinal{c}(2,1)/NumTrialTotal)*100;
                performance.timebin{tb}.classifier{c}.Landmark = (data.timebin{tb}.confmatfinal{c}(2,2)/NumTrialTotal)*100;
                performance.timebin{tb}.classifier{c}.Object = (data.timebin{tb}.confmatfinal{c}(2,3)/NumTrialTotal)*100;
            elseif strcmp(cfg.category_predict,'Object')
                performance.timebin{tb}.classifier{c}.Face = (data.timebin{tb}.confmatfinal{c}(3,1)/NumTrialTotal)*100;
                performance.timebin{tb}.classifier{c}.Landmark = (data.timebin{tb}.confmatfinal{c}(3,2)/NumTrialTotal)*100;
                performance.timebin{tb}.classifier{c}.Object = (data.timebin{tb}.confmatfinal{c}(3,3)/NumTrialTotal)*100;
            end
        end
    end    
end
