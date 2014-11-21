%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% demoLFCS_mirflickr: a demo script to run LFCS (wang ICCV2013) method
%   on mirflickr dataset (new, similar as ICCV2013)
% Xing Xu
% Limu, Kyushu University, Japan
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function demoLFCS_mirflickr()

addpath(genpath('SPAMS_2.5'));
addpath(genpath('../config_mirflickr'))
model_dir = './model/mirflickr';
file_model = 'model_LFCS.mat';

if ~exist(fullfile(model_dir, model_name), 'file')

    [VTr, VTe, CTr, CTe, TTr, TTe] = load_multiple_feature_mirflickr('config_file_mirflickr');

    X_a = VTr; % nsample x d
    X_b = TTr;

    Test_a = VTe;
    Test_b = TTe;

    train_Y = CTr; % nsample x c
    test_Y = CTe;

    %% training parameters for LCFS method
    lambda_1 = 0.1;
    lambda_2 = 0.001;
    ite = 5;

    [W_a, W_b] = LCFS_ite( X_a, X_b, train_Y, lambda_1, lambda_2, ite);

    projected_Y_a = Test_a * W_a;
    projected_Y_b = Test_b * W_b;

    
    %% evaluation
    resultsTag = evaluatePR(test_Y', projected_Y_a', 5, 'tag');
    fprintf('... For tag measure from image modality: \n\t P %f, R %f, N+ %d \n', resultsTag.prec, resultsTag.rec, resultsTag.retrieved);

    resultsTag = evaluatePR(test_Y', projected_Y_b', 5, 'tag');
    fprintf('... For tag measure from text modality: \n\t P %f, R %f, N+ %d \n', resultsTag.prec, resultsTag.rec, resultsTag.retrieved);


    resultsImage = evaluatePR(test_Y', projected_Y_a', 5, 'image');
    fprintf('... For image measure from image modality: \n\t P %f, R %f, N+ %d \n', resultsImage.prec, resultsImage.rec, resultsImage.retrieved);

    resultsImage = evaluatePR(test_Y', projected_Y_b', 5, 'image');
    fprintf('... For image measure from text modality: \n\t P %f, R %f, N+ %d \n', resultsImage.prec, resultsImage.rec, resultsImage.retrieved);

    save(fullfile(model_dir, file_model), 'W_a', 'W_b', 'projected_Y_a', 'projected_Y_b', 'test_Y');

else
    load(fullfile(model_dir, model_name));
end

%% evaluation for using visual modality to recover text modality
pred_projected_Y_b = projected_Y_a / W_b;

resultsTag = evaluatePR(TTe', pred_projected_Y_b', 5, 'tag');
fprintf('... For tag measure from image modality: \n\t P %f, R %f, N+ %d \n', resultsTag.prec, resultsTag.rec, resultsTag.retrieved);


resultsImage = evaluatePR(TTe', pred_projected_Y_b', 5, 'image');
fprintf('... For image measure from image modality: \n\t P %f, R %f, N+ %d \n', resultsImage.prec, resultsImage.rec, resultsImage.retrieved);

end