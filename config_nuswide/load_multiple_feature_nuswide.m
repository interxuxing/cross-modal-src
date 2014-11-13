%function [FeaTr, FeaTe, AnnoTr, AnnoTe, TagTr, TagTe] = load_multiview_feature_nuswide(config_file)
% is to load 3 views features: visual, textual, concept for valid images
%
% return 'Fea' (visual), 'Anno' (concept), 'Tag' (textual) for both valid
% train/test images
function [FeaTr, FeaTe, AnnoTr, AnnoTe, TagTr, TagTe] = load_multiple_feature_nuswide(config_file)

eval(config_file);

featNames = {'BoW', 'CH', 'CM55', 'CORR', 'EDH', 'Gist', 'WT'};
annoNames = {'Concepts81'};
tagNames = {'Tags1k'};
dataFolder = [IMAGE_ANNOTATION_DIR,'\'];
tic;

%% 1st parse concepts (annotations)
fprintf('1st parse concepts (annotations) \n');
for anno = annoNames
    anno = char(anno);
    if ispc
        matchStr = regexp(ls_win(dataFolder), ['\w*_', anno, '_Train'], 'match');
    elseif isunix
        matchStr = regexp(ls(dataFolder), ['\w*_', anno, '_Train'], 'match');
    end
    
    load(strcat(dataFolder, matchStr{1}));
	AnnoTr = double(valid_train_81(valid_train_81_index, :));
    
    if ispc
        matchStr = regexp(ls_win(dataFolder), ['\w*_', anno, '_Test'], 'match');
    elseif isunix
        matchStr = regexp(ls(dataFolder), ['\w*_', anno, '_Test'], 'match');
    end
	load(strcat(dataFolder, matchStr{1}));
    AnnoTe = double(valid_test_81(valid_test_81_index, :));
end

%% 2. parse tag feature files
fprintf('2. parse tag feature files \n');
for tag = tagNames
    tag = char(tag);
    if ispc
        matchStr = regexp(ls_win(dataFolder), ['\w*_', tag, '_Train'], 'match');
    else isunix
        matchStr = regexp(ls(dataFolder), ['\w*_', tag, '_Train'], 'match');
    end
    
    load(strcat(dataFolder, matchStr{1}));
    TagTr = double(full(valid_feature_matrix(valid_train_81_index, :)));
    
    if ispc
        matchStr = regexp(ls_win(dataFolder), ['\w*_', tag, '_Test'], 'match');
    else isunix
        matchStr = regexp(ls(dataFolder), ['\w*_', tag, '_Test'], 'match');
    end
    
    load(strcat(dataFolder, matchStr{1}));
    TagTe = double(full(valid_feature_matrix(valid_test_81_index, :)));
end


%% 3. parse visual feature files
FeaTr = [];
FeaTe = [];
fprintf('3. parse visual feature files \n');
for feat = featNames
    feat = char(feat);
    fprintf('... for feature type %s ... \n', feat);
    if ispc
        matchStr = regexp(ls_win(dataFolder), ['\w*_', feat, '_Train'], 'match');
    elseif isunix
        matchStr = regexp(ls(dataFolder), ['\w*_', feat, '_Train'], 'match');
    end
    
    load(strcat(dataFolder, matchStr{1}));
	currFeaTr = normalizeL2(valid_feature_matrix(valid_train_81_index, :));
	FeaTr = [FeaTr, currFeaTr];

    if ispc
        matchStr = regexp(ls_win(dataFolder), ['\w*_', feat, '_Test'], 'match');
    elseif isunix
        matchStr = regexp(ls(dataFolder), ['\w*_', feat, '_Test'], 'match');
    end
    
    load(strcat(dataFolder, matchStr{1}));
	currFeaTe = normalizeL2(valid_feature_matrix(valid_test_81_index, :));
	FeaTe = [FeaTe, currFeaTe];
end


% %%%% generate valid index for cross-validation in fasttag code
% nTr = size(FeaTr,1);
% ValIdx = false(1, nTr);
% ValIdx(randsample(nTr, floor(nTr*0.2))) = true;

toc;
end


% function X_out = normalizeL2(X_in)
%     X_out = X_in ./ norm(X_in, 2);
% end

function X_out = normalizeL2(X_in)
    [numImg, numDim] = size(X_in);
    X_out = zeros(numImg, numDim);
    for j = 1 : numImg
        summ = sqrt(sum(X_in(j,:).^2));
        if (summ > 0)
            X_out(j,:) = X_in(j,:)/summ;
        end
    end
end

% function X_out = normalizeL1(X_in)
%     X_out = X_in ./ norm(X_in, 1);
% end