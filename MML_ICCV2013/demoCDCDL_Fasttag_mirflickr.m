%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% demoCDCDL_Fasttag_mirflickr: a demo script to run CDCDL
%   method to get coupled sparse feature representation,
%   and then use traditional ridge regression / fasttag
%   method for model learning
%
% Xing Xu
% Limu, Kyushu University, Japan
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function demoCDCDL_Fasttag_mirflickr()

%% load the trained model on mirflickr dataset
model_dir = './model/mirflickr';
model_name = 'mirflickr_model_300.mat';

% here we load the pre-trained model file, here we have the parameters of
% the dictionaries and mapping matrices
load(fullfile(model_dir, model_name));

addpath(genpath('SPAMS_2.5'));
addpath(genpath('../config_mirflickr'))

load params;

% load mirflickr
% first load V (visual feature matrix, nxd), C (concept matrix, nxc)
[VTr, VTe, CTr, CTe, TTr, TTe] = load_multiple_feature_mirflickr('config_file_mirflickr');


% first use the coupled dictionalries to recover the sparse representations
% of training samples

X_a = VTr'; % d x nsamples
X_b = TTr'; % t x nsamples
yTr = CTr';


AlphahTr = full(mexLasso(X_a, Dh, param));
AlphalTr = full(mexLasso(X_b, Dl, param));
AlphaTr = [AlphahTr; AlphalTr];

X_a = VTe'; % nsamples x d
X_b = TTe'; % nsamples x t
yTe = CTe';

AlphahTe = full(mexLasso(X_a, Dh, param));
AlphalTe = full(mexLasso(X_b, Dl, param));
AlphaTe = [AlphahTe; AlphalTe];

addpath(genpath('/media/Data/myproject/icme2014/fasttag'));

%% first evaluate, use ridge regression model to train annotation classifiers

[W_lr] = linear_regression(AlphaTr, yTr, AlphaTe, yTe, 5);
pred_Te = W_lr * AlphaTe;

resultsTag = evaluatePR(yTe, pred_Te, 5, 'tag');
fprintf('... For tag measure: \n\t P %f, R %f, N+ %d \n', resultsTag.prec, resultsTag.rec, resultsTag.retrieved);

resultsImage = evaluatePR(yTe, pred_Te, 5, 'image');
fprintf('... For image measure: \n\t P %f, R %f, N+ %d \n', resultsImage.prec, resultsImage.rec, resultsImage.retrieved);

% for image modality
[W_lrh] = linear_regression(AlphahTr, yTr, AlphahTe, yTe, 5);
pred_Te = W_lrh * AlphahTe;
resultsTag = evaluatePR(yTe, pred_Te, 5, 'tag');
fprintf('... For tag measure from image modality: \n\t P %f, R %f, N+ %d \n', resultsTag.prec, resultsTag.rec, resultsTag.retrieved);

resultsImage = evaluatePR(yTe, pred_Te, 5, 'image');
fprintf('... For image measure from image modality: \n\t P %f, R %f, N+ %d \n', resultsImage.prec, resultsImage.rec, resultsImage.retrieved);

% for text modality
[W_lrl] = linear_regression(AlphalTr, yTr, AlphalTe, yTe, 5);
pred_Te = W_lrl * AlphalTe;

resultsTag = evaluatePR(yTe, pred_Te, 5, 'tag');
fprintf('... For tag measure from tag modality: \n\t P %f, R %f, N+ %d \n', resultsTag.prec, resultsTag.rec, resultsTag.retrieved);

resultsImage = evaluatePR(yTe, pred_Te, 5, 'image');
fprintf('... For image measure from tag modality: \n\t P %f, R %f, N+ %d \n', resultsImage.prec, resultsImage.rec, resultsImage.retrieved);

%% use redge regression model to train tag classifiers

% for image modality
[W_lrh] = linear_regression(AlphahTr, TTr', AlphahTe, TTe', 5);
pred_Te = W_lrh * AlphahTe;
resultsTag = evaluatePR(TTe', pred_Te, 5, 'tag');
fprintf('... For tag measure from image modality: \n\t P %f, R %f, N+ %d \n', resultsTag.prec, resultsTag.rec, resultsTag.retrieved);

resultsImage = evaluatePR(TTe', pred_Te, 5, 'image');
fprintf('... For image measure from image modality: \n\t P %f, R %f, N+ %d \n', resultsImage.prec, resultsImage.rec, resultsImage.retrieved);


%% second evaluate CDCDL+Fasttag
% generate the valIdx
nTr = size(yTr, 2);
k = floor(nTr * 0.15);

valIdx = false(1, nTr);
valIdx(randsample(nTr, k)) = true;

% evaluate image+text modalities
[W_fasttag] = fasttag(AlphaTr, yTr, AlphaTe, yTe, 5, valIdx);
pred_Te = W_fasttag * AlphaTe;

resultsTag = evaluatePR(yTe, pred_Te, 5, 'tag');
fprintf('... For tag measure: \n\t P %f, R %f, N+ %d \n', resultsTag.prec, resultsTag.rec, resultsTag.retrieved);

resultsImage = evaluatePR(yTe, pred_Te, 5, 'image');
fprintf('... For image measure: \n\t P %f, R %f, N+ %d \n', resultsImage.prec, resultsImage.rec, resultsImage.retrieved);

% evaluate iamge modality
[W_fasttagh] = fasttag(AlphahTr, yTr, AlphahTe, yTe, 5, valIdx);
pred_Te = W_fasttagh * AlphahTe;

resultsTag = evaluatePR(yTe, pred_Te, 5, 'tag');
fprintf('... For tag measure from image modality: \n\t P %f, R %f, N+ %d \n', resultsTag.prec, resultsTag.rec, resultsTag.retrieved);

resultsImage = evaluatePR(yTe, pred_Te, 5, 'image');
fprintf('... For image measure from image modality: \n\t P %f, R %f, N+ %d \n', resultsImage.prec, resultsImage.rec, resultsImage.retrieved);

% evaluate text modality
[W_fasttagl] = fasttag(AlphalTr, yTr, AlphalTe, yTe, 5, valIdx);
pred_Te = W_fasttagl * AlphalTe;

resultsTag = evaluatePR(yTe, pred_Te, 5, 'tag');
fprintf('... For tag measure from text modality: \n\t P %f, R %f, N+ %d \n', resultsTag.prec, resultsTag.rec, resultsTag.retrieved);

resultsImage = evaluatePR(yTe, pred_Te, 5, 'image');
fprintf('... For image measure from text modality: \n\t P %f, R %f, N+ %d \n', resultsImage.prec, resultsImage.rec, resultsImage.retrieved);

%% use fasttag to train tag classifiers

% for image modality
[W_lrh] = fasttag(AlphahTr, TTr', AlphahTe, TTe', 5, valIdx);
pred_Te = W_lrh * AlphahTe;
resultsTag = evaluatePR(TTe', pred_Te, 5, 'tag');
fprintf('... For tag measure from image modality: \n\t P %f, R %f, N+ %d \n', resultsTag.prec, resultsTag.rec, resultsTag.retrieved);

resultsImage = evaluatePR(TTe', pred_Te, 5, 'image');
fprintf('... For image measure from image modality: \n\t P %f, R %f, N+ %d \n', resultsImage.prec, resultsImage.rec, resultsImage.retrieved);

end

