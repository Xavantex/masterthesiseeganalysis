%------------------------------------------------------------------------------
%  TEST THE CROSS VALIDATED CLASSIFIER ON THE RETRIEVAL DATA
%------------------------------------------------------------------------------
%  USE DATA FROM visual CLASSIFIED TRIALS - FREQUENCY
%------------------------------------------------------------------------------
clear all
tic
delete(gcp('nocreate'))
ticBytes(gcp);
mpiprofile on

%Visual
directory  = 'F:\\EXJOBB\\to_xavante\\Visual\\'; %directory of the data
subject = {'Subj01' 'Subj02' 'Subj03' 'Subj04'};% 'Subj05' 'Subj06' 'Subj07' 'Subj08' 'Subj09' 'Subj11' 'Subj12' 'Subj13' 'Subj14' 'Subj15' 'Subj16' 'Subj17' 'Subj18' 'Subj19'};
%subject = {'Subj01'};
epoch = {'study'};

%Verbal
% directory  = '//psysrv004/psymemlab/Projects/TAPMVPA-LTH/Verbal/'; %directory of the data
% subject = {'Subj01' 'Subj02' 'Subj06' 'Subj07' 'Subj08' 'Subj09' 'Subj10' 'Subj11'...
%     'Subj12' 'Subj14' 'Subj15' 'Subj16' 'Subj17' 'Subj18' 'Subj19' 'Subj20' 'Subj21' 'Subj22'};


parfor j = 1:length(subject)
    for e = 1:length(epoch)
        disp(subject{j})
        cd(sprintf('%s%s',directory,subject{j},'/6-ClassificationData/'));
        cname = sprintf('%s_test_data_visual',subject{j}); %give imput data
        cdata = load(cname, cname);
        field = fieldnames(cdata);
        cdata = cdata.(field{1});

        cd(sprintf('%s/%s/7-ClassifierTraining/',directory,subject{j}));
        crossname = sprintf('%s_%s_crossvalclass',subject{j},epoch{e}); %give classifier
        crossdata = load(crossname, crossname);
        field = fieldnames(crossdata);
        crossdata = crossdata.(field{1});

        % apply calssifier into the correct visual/ correct visual trials
        cfg = []; cfg.fold = 10; cfg.classifiernumber = 20; cfg.timebinsnumber = 20;
        cfg.category_predict = {'Face' 'Landmark' 'Object'}; cfg.trials = 'all';
        cfg.category_model = {'Face' 'Landmark' 'Object'};
        predtest_visual = mvpa_applycrossvalclassifier(cfg, crossdata, cdata);
        disp(subject{j})
        disp('Done!')
        cd(sprintf('%s/%s/8-ClassifierTesting/',directory,subject{j}))
        predtestname = sprintf('%s_%s_predtest_visual',subject{j},epoch{e});
        parsave(predtestname, predtest_visual);
        % Performance
        cfg = []; cfg.performance = 2; cfg.category_model = {'Face' 'Landmark' 'Object'};
        cfg.category_predict = {'Face' 'Landmark' 'Object'};
        cfg.classifiernumber = 20; cfg.timebinsnumber = 20;
        predtest_visual_performance = mvpa_classifierperf(cfg, predtest_visual);
        predtestpername = sprintf('%s_%s_predtest_visual_performance',subject{j},epoch{e});
        parsave(predtestpername, predtest_visual_performance);
        disp('Performance calculated!')
    end
end

tocBytes(gcp)
toc
mpiprofile viewer

function parsave(fname, crossval)
    newName = fname;
    S.(newName) = crossval;
    save(fname, '-struct', 'S')
end