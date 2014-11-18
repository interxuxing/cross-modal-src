%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% demoCDCDL: Cross-domain Coupled Dictionary Learning for Image Annotation
% Xing Xu
% Limu, Kyushu University, Japan
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function demoCDCDL_mirflickr
    clear;
%     clc;
    
    %% load data
    addpath(genpath('SPAMS_2.5'));
    addpath(genpath('../config_mirflickr'))
    
    if ~exist('./model/', 'dir')
        mkdir('./model/');
    end
    MODEL_DIR = './model/mirflickr';
    
    load params;
    
    % load nuswide data
    % first load V (visual feature matrix, nxd), C (concept matrix, nxc)
    [VTr, VTe, CTr, CTe, TTr, TTe] = load_multiple_feature_mirflickr('config_file_mirflickr');
    
    % trainspose the features with column-wise dxn
    X_a = VTr';
    X_b = TTr';
    
    train_Y = CTr;
    
    Test_a = VTe';
    Test_b = TTe';
    test_Y = CTe;
    
    %% training parameters
    lambda_1 = 0.1;
    lambda_2 = 0.001;
    ite = 5;
    
    
    [dim_a, num] = size( X_a );
    [dim_b, num] = size( X_b );
    [num, c] = size(train_Y);
    
    %% Parameters setting
    c = 300;
    par.mu = par.mu*1;
    par.K 	= c;
    param.K = c;
    par.L	= c;

    
    NU = [0.0001, 0.001, 0.01, 0.1, 1, 10];
    MU = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10];
    %% param is for SPMAS toolbox
    param.L = c;
    param.lambda        = par.lambda1; % not more than 20 non-zeros coefficients
    param.lambda2       = par.lambda2;
    param.mode          = 2;       % penalized formulation
    param.approx=0;
    param.verbose       = true;
    param.iter          = 100;
    
    fprintf('Coupled dictionary learning with size c = %d \n', c);
    file_model = sprintf('mirflickr_model_%d.mat', c);
    
    %% Intialize D,A, and W(U)
    D = mexTrainDL([X_a;X_b], param); % train dictionaries Dh, Dl from training data X_a, X_b
    Dh = D(1:size(X_a,1),:); 
    Dl = D(size(X_a,1)+1:end,:);
    Wl = eye(size(Dh, 2));
    Wh = eye(size(Dl, 2));
    Alphah = mexLasso([X_a;X_b], D, param);
    Alphal = Alphah;
    clear D;
    
    % Iteratively solve D,A, and W (U)
    
    
        par.nu = 0.001;
        par.mu = 0.0001;
        par.epsilon = 0.001;
        
        tic;
        [Alphah, Alphal, XH_t, XL_t, Dh, Dl, Wh, Wl, Uh, Ul, f] = coupled_DL_recoupled(Alphah, Alphal, X_a, X_b, Dh, Dl, Wh, Wl, par);
        clear XH_t XL_t;
        toc;
        
        save(fullfile(MODEL_DIR, file_model), 'Dh', 'Dl', 'Wh', 'Wl', 'Uh', 'Ul');
        %% now use the sparse representations for new modal learning
        lambda_1 = 0.1;
        lambda_2 = 0.001;
        ite = 5;
    
        [W_a, W_b] = LCFS_ite( Alphah', Alphal', train_Y, lambda_1, lambda_2, ite);
        
        %% test
        % first project the testing samples on the common feature space
        Alphah = full(mexLasso(Test_a, Dh, param));
        Alphal = full(mexLasso(Test_b, Dl, param));
        projected_Y_a = Alphah' * W_a;
        projected_Y_b = Alphal' * W_b;

        
        %% evaluate with P, R, N+ measures
        resultsTag = evaluatePR(test_Y', projected_Y_a', 5, 'tag');
        fprintf('... For tag measure from image modality: \n\t P %f, R %f, N+ %d \n', resultsTag.prec, resultsTag.rec, resultsTag.retrieved);

        resultsTag = evaluatePR(test_Y', projected_Y_b', 5, 'tag');
        fprintf('... For tag measure from text modality: \n\t P %f, R %f, N+ %d \n', resultsTag.prec, resultsTag.rec, resultsTag.retrieved);
        
        
        resultsImage = evaluatePR(test_Y', projected_Y_a', 5, 'image');
        fprintf('... For image measure from image modality: \n\t P %f, R %f, N+ %d \n', resultsImage.prec, resultsImage.rec, resultsImage.retrieved);
        
        resultsImage = evaluatePR(test_Y', projected_Y_b', 5, 'image');
        fprintf('... For image measure from text modality: \n\t P %f, R %f, N+ %d \n', resultsImage.prec, resultsImage.rec, resultsImage.retrieved);
        
        save(fullfile(MODEL_DIR, file_model), 'W_a', 'W_b', 'projected_Y_a', 'projected_Y_b', 'test_Y', '-append');
        fprintf('Finished testing model for c = %d, saved model results \n', c);
    
end