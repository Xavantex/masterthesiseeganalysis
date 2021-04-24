%------------------------------------------------------------------------------
%  PERMUTATION FOR EACH PARTICIPANT!!!! FREQUENCY
%------------------------------------------------------------------------------
clear all
tic
%Visual
directory  = 'F:\\EXJOBB\\to_xavante\\Visual\\'; %directory of the data
subjects = {'Subj01' 'Subj02' 'Subj03' 'Subj04' 'Subj05' 'Subj06' 'Subj07' 'Subj08' 'Subj09' 'Subj11' 'Subj12' 'Subj13' 'Subj14' 'Subj15' 'Subj16' 'Subj17' 'Subj18' 'Subj19'};
%subjects = {'Subj01'};
epoch = {'study'};
% Keeping the number of permutations low just to save time for now:
NOF_PERMUTATIONS = 5;
%NOF_PERMUTATIONS = 5;

%Verbal
% directory  = '//psysrv004/psymemlab/Projects/TAPMVPA-LTH/Verbal/'; %directory of the data
% subject = {'Subj01' 'Subj02' 'Subj06' 'Subj07' 'Subj08' 'Subj09' 'Subj10' 'Subj11'...
%     'Subj12' 'Subj14' 'Subj15' 'Subj16' 'Subj17' 'Subj18' 'Subj19' 'Subj20' 'Subj21' 'Subj22'};

tstart = tic;
for j = 1:length(subjects)
    for e = 1:length(epoch)
        disp(subjects{j})
        cd(sprintf('%s%s',directory,subjects{j},'/6-ClassificationData/'));
        cname = sprintf('%s_test_data_visual',subjects{j}); %give imput data
        cdata = load(cname, cname);
        field = fieldnames(cdata);
        cdata = cdata.(field{1});

        cd(sprintf('%s/%s/7-ClassifierTraining/',directory,subjects{j}))
        crossname = sprintf('%s_%s_crossvalclass',subjects{j},epoch{e})
        crossdata = load (crossname, crossname);
        field = fieldnames(crossdata);
        crossdata = crossdata.(field{1});


        for PERMUTATION = 1:NOF_PERMUTATIONS %Number of Permutations
            R = randperm(length(cdata.category_name));
            per.category_name=cdata.category_name(R);%PER%d
            per.category=cdata.category(R);
            per.feature_name = cdata.feature_name;
            per.trialinfo = cdata.trialinfo;
            per.feature = cdata.feature;
            per.numclassifiers = cdata.numclassifiers;
            
            % apply calssifier into the shuffled category trials
            cfg.fold = 10; cfg.classifiernumber = 20; cfg.timebinsnumber = 20;
            cfg.category_predict = {'Face' 'Landmark' 'Object'}; cfg.trials = 'all';
            cfg.category_model = {'Face' 'Landmark' 'Object'};
            per_predtest = mvpa_applycrossvalclassifier(cfg, crossdata, per);%PER%d_predtest
            
            %Construct and save the Permutation Map
            tstartin = tic;
            sub_P = zeros(20,20);
            for col = 1:20 %timebins at retrieval
                for row = 1:20 %classifiers
                    trials = sum(per_predtest.timebin{col}.confmatfinal{row},2);
                    sub_P(row,col) = ((per_predtest.timebin{col}.confmatfinal{row}(1,1)/trials(1)*100) + (per_predtest.timebin{col}.confmatfinal{row}(2,2)/trials(2)*100) + (per_predtest.timebin{col}.confmatfinal{row}(3,3)/trials(3)*100))/3;
                end
            end
            tendin = toc(tstartin);
            cd(sprintf('%s/%s/8-ClassifierTesting/PermutationStudyDecodeTestVisual',directory,subjects{j}));
            subname = sprintf('%s_P%d',subjects{j},PERMUTATION);
            parsave(subname, sub_P);
            fprintf('Permutation %d done! /n', PERMUTATION);
        end
    end
end
clear per per_predtest
fprintf('First Loop %s', toc(tstart));
fprintf('Inner first loop %s', tendin);

%------------------------------------------------------------------------------
% For each permutation a T-Test is calculated
% (classification accuracy is compared against chance (33.33))
% Each T test is stored in a Struture called Random_Distribution
%------------------------------------------------------------------------------ 
clear all
directory  = 'F:\\EXJOBB\\to_xavante\\Visual\\'; %directory of the data
subject = {'Subj01' 'Subj02' 'Subj03' 'Subj04' 'Subj05' 'Subj06' 'Subj07' 'Subj08' 'Subj09' 'Subj11' 'Subj12' 'Subj13' 'Subj14' 'Subj15' 'Subj16' 'Subj17' 'Subj18' 'Subj19'};
NOF_PERMUTATIONS = 5;

%InfoSubjects_mvpa
%subjects = fieldnames(Infosubjects);
%Infosubjects = struct2cell(Infosubjects);
epoch = {'study'};

Permutation_total(1,1:20) = 0;
for PERMUTATION = 1:NOF_PERMUTATIONS
    for e = 1:length(epoch)
        cdata = zeros(length(subjects));
        for j = 1:length(subjects)
            cd(sprintf('%s/%s/8-ClassifierTesting/PermutationStudyDecodeTestVisual',directory,subjects{j}))
            cname = sprintf('%s_P%d',subjects{j},PERMUTATION);
            cdata(j) = load(cname);%(cname(j));
            field = fieldnames(cdata(j));
            cdata(j) = cdata(j).(field{1});
            cdata(j) = imgaussfilt(cdata(j),1);
        end
        tstartin = tic;
        X = zeros(length(subjects));
        T_P = zeros(20,20);
        P_P = zeros(20,20);
        for col = 1:20 %timebins at retrieval
            for row = 1:20 %classifiers
                for j = 1:length(subjects)
                    X(j) = cdata(j, row, col);
                end
                ttstartin = tic;
                [H,P,CI,STATS] = ttest(X,33.33,'tail','both');
                ttendin = toc(ttendin);
                T_P(row,col) = STATS(1).tstat;
                P_P(row,col) = P;
            end
        end
        tendin = toc(tstartin);
        clear cdata
        fprintf('permutation %d.../n',PERMUTATION);
        
        Random_Distribution.Plevel{PERMUTATION} = P_P;
        Random_Distribution.Ttest{PERMUTATION} = T_P;
    end
end

save Random_Distribution Random_Distribution
fprintf('Second loop %s', toc(tstart));
fprintf('Second inner loop %s', tendin);
fprintf('Second in inner loop %s', ttendin);

function parsave(fname, crossval)
    newName = fname;
    S.(newName) = crossval;
    save(fname, '-struct', 'S')
end