%function [FeaTr, FeaTe, AnnoTr, AnnoTe, TagTr, TagTe] = load_multiview_feature_mirflickr(config_file_mirflickr)
% is to load 3 views features: visual, textual, concept for valid images
%
% return 'Fea' (visual), 'Anno' (concept), 'Tag' (textual) for both valid
% train/test images
function [FeaTr, FeaTe, AnnoTr, AnnoTe, TagTr, TagTe] = load_multiple_feature_mirflickr(config_file)

if nargin ~= 1
    error('please input 1 parameter of config file');
end

eval(config_file);

if ~exist(fullfile(MODEL_DIR, filesep))
    mkdir(fullfile(MODEL_DIR, filesep));
end

dataFolder = fullfile(IMAGE_ANNOTATION_DIR, filesep);
dimen = DIM;

tic;

%% load multiple features, totally 15 types of features
%   here features are d x nsamples matrix, labels are t x nsamples
if ~exist([dataFolder, 'data,dimen=', num2str(dimen), '.mat'], 'file')
    fprintf('Feature file not exist, generate then load it. \n');
    [xTr, yTr, xTe, yTe, valIdx] = loaddata(dataFolder, dimen);
else
    fprintf('Feature file already exists, load it. \n');
    load([dataFolder, 'data,dimen=', num2str(dimen), '.mat']);
end

% transpose matrix to nsample x d
FeaTr = xTr';
FeaTe = xTe';
TagTr = yTr';
TagTe = yTe';

%% then we parse the tag features nsamples x c
AnnoTr = dlmread(fullfile(dataFolder, 'mirflickr_train_classes.txt'));
AnnoTe = dlmread(fullfile(dataFolder, 'mirflickr_test_classes.txt'));

fprintf('Finished load features, annotations for mirflickr dataset! \n');

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