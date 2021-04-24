%--------------------------------------------------------------------------
% Training the Classifier
% Frequency
%--------------------------------------------------------------------------
clear all
tic
delete(gcp('nocreate'))
ticBytes(gcp);
%mpiprofile on

%Visual
directory  = parallel.pool.Constant('F:\\EXJOBB\\to_xavante\\Visual\\'); %directory of the data
subject = parallel.pool.Constant({'Subj01' 'Subj02' 'Subj03' 'Subj04' 'Subj05' 'Subj06' 'Subj07' 'Subj08' 'Subj09' 'Subj11' 'Subj12' 'Subj13' 'Subj14' 'Subj15' 'Subj16' 'Subj17' 'Subj18' 'Subj19'});
%subject = {'Subj01'};

%Verbal
% directory  = '//psysrv004/psymemlab/Projects/TAPMVPA-LTH/Verbal/'; %directory of the data
% subject = {'Subj01' 'Subj02' 'Subj06' 'Subj07' 'Subj08' 'Subj09' 'Subj10' 'Subj11'...
%     'Subj12' 'Subj14' 'Subj15' 'Subj16' 'Subj17' 'Subj18' 'Subj19' 'Subj20' 'Subj21' 'Subj22'};


epoch = parallel.pool.Constant({'study'});
response = parallel.pool.Constant({'visual'}); %if Verbal, reponse lexical

parfor j = 1:length(subject.Value)
    disp(subject.Value{j})
    for e  =  1:length(epoch.Value)
        cd(sprintf('%s%s%s',directory.Value,subject.Value{j},'/6-ClassificationData/'));
        cname = sprintf('%s_%s_data',subject.Value{j},epoch.Value{e});
        cdata = load(cname, cname);
        field = fieldnames(cdata);
        cdata = cdata.(field{1});
        % partiation of the data
        cfg = []; cfg.classifiernumber = 20; cfg.fold = 10;
        datapart = mvpa_datapartition(cfg, cdata);
        disp('Data was partitioned and feature reduction for each partion is complete')

        %Train the cross validated classifier
        cfg = []; cfg.training_algorithm = 1; cfg.fold = 10; cfg.classifiernumber = 20; cfg.category_model = {'Face' 'Landmark' 'Object'};
        crossvalclass = mvpa_traincrossvalclassifier(cfg, datapart);
        disp('Cross Validated model successfully performed!')

        %Performance
        cfg = []; cfg.performance = 1; cfg.category_model = {'Face' 'Landmark' 'Object'};
        cfg.classifiernumber = 20;
        crossvalclass_performance = mvpa_classifierperf(cfg, crossvalclass);
        disp('Performance calculated!')
        
        %save
        cd(sprintf('%s/%s/7-ClassifierTraining/',directory.Value,subject.Value{j}));
        crossvalname = sprintf('%s_%s_crossvalclass',subject.Value{j},epoch.Value{e});
        parsave(crossvalname, crossvalclass);
        crossvalname = sprintf('%s_%s_datapart',subject.Value{j},epoch.Value{e});
        parsave(crossvalname, datapart);
        crossvalname = sprintf('%s_%s_crossvalclass_performance',subject.Value{j},epoch.Value{e});
        parsave(crossvalname, crossvalclass_performance);
    end
end
%clear crossvalclass datapart
%mpiprofile viewer

%pause

%mpiprofile clear
%mpiprofile on
% 

%--------------------------------------------------------------------------
% Training the Classifier
% Amplitude
%--------------------------------------------------------------------------

%Visual
%subject = {'Subj01'};

%Verbal
% directory  = '//psysrv004/psymemlab/Projects/TAPMVPA-LTH/Verbal/';
% subject = {'Subj01' 'Subj02' 'Subj06' 'Subj07' 'Subj08' 'Subj09' 'Subj10' 'Subj11'...
%     'Subj12' 'Subj14' 'Subj15' 'Subj16' 'Subj17' 'Subj18' 'Subj19' 'Subj20' 'Subj21' 'Subj22'};


%if Verbal, reponse lexical

parfor j = 1:length(subject.Value)
    disp(subject.Value{j})
    for e  =  1:length(epoch.Value)
        cd(sprintf('%s%s%s',directory.Value,subject.Value{j},'/6-ClassificationData/'));
        cname = sprintf('%s_%s_data_amp',subject.Value{j},epoch.Value{e});
        cdata = load(cname, cname);
        field = fieldnames(cdata);
        cdata = cdata.(field{1});

        % partiation of the data
        cfg = []; cfg.classifiernumber = 40; cfg.fold = 10;
        datapart = mvpa_datapartition(cfg,cdata);
        disp('Data was partitioned and feature reduction for each partion is complete')

        %Train the cross validated classifier
        cfg = []; cfg.training_algorithm = 1; cfg.fold = 10; cfg.classifiernumber = 40; cfg.category_model = {'Face' 'Landmark' 'Object'};
        crossvalclass = mvpa_traincrossvalclassifier(cfg, datapart);
        disp('Cross Validated model successfully performed!')

        %Performance
        cfg = []; cfg.performance = 1; cfg.category_model = {'Face' 'Landmark' 'Object'};
        cfg.classifiernumber = 40;
        crossvalclass_performance = mvpa_classifierperf(cfg, crossvalclass);
        disp('Performance calculated!')

        %save
        cd(sprintf('%s/%s/7-ClassifierTraining/',directory.Value,subject.Value{j}));
        crossvalname = sprintf('%s_%s_crossvalclass_amp',subject.Value{j},epoch.Value{e});
        parsave(crossvalname, crossvalclass);
        crossvalname = sprintf('%s_%s_datapart_amp',subject.Value{j},epoch.Value{e});
        parsave(crossvalname, datapart);
        crossvalname = sprintf('%s_%s_crossvalclass_performance_amp',subject.Value{j},epoch.Value{e});
        parsave(crossvalname, crossvalclass_performance);
    end
end
clear crossvalclass datapart

tocBytes(gcp)
toc
%mpiprofile viewer

function parsave(fname, crossval)
    newName = fname;
    S.(newName) = crossval;
    save(fname, '-struct', 'S')
end