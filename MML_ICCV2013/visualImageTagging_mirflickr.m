%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% visualImageTagging_mirflickr: visualize the typical image tagging
%   examples on mirflickr dataset (new, similar as ICCV2013)
% Xing Xu
% Limu, Kyushu University, Japan
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function visualImageTagging_mirflickr()

%% first set the paths of images and features of mirflickr dataset
Src_Feature_Dir = '/media/Data/dataset/data/mirflickr/mirflickr.20101118';
Img_Dir = '/media/Data/dataset/data/mirflickr/mirflickr-images';

%% then load the trained model on mirflickr dataset
model_dir = './model/mirflickr';
model_name = 'mirflickr_model_300.mat';

% here we load the pre-trained model file, here we have the parameters of
% the dictionaries and mapping matrices
load(fullfile(model_dir, model_name));


%% given the papramters, we need to first recover the cross-modal data
% first load the features

addpath(genpath('SPAMS_2.5'));
addpath(genpath('../config_mirflickr'))

load params;

% load mirflickr
% first load V (visual feature matrix, nxd), C (concept matrix, nxc)
[VTr, VTe, CTr, CTe, TTr, TTe] = load_multiple_feature_mirflickr('config_file_mirflickr');


%% here our target is, given only visual modality, recover the text modality
Test_a = VTe';

Alphah = full(mexLasso(Test_a, Dh, param));
projected_Y_a = Alphah' * W_a;

pred_Alphah = projected_Y_a / W_a;

pred_Alphal = projected_Y_b / W_b;
pred_Test_b = pred_Alphal * Dl';


resultsTag = evaluatePR(TTe', pred_Test_b', 5, 'tag');
fprintf('... For tag measure from image modality: \n\t P %f, R %f, N+ %d \n', resultsTag.prec, resultsTag.rec, resultsTag.retrieved);

end