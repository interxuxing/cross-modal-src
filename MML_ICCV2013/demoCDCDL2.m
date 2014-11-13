%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% demoCDCDL: Cross-domain Coupled Dictionary Learning for Image Annotation
% Xing Xu
% Limu, Kyushu University, Japan
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function demoCDCDL2
    clear;
    clc;
    
    %% load data
    addpath(genpath('SPAMS_2.5'));
    
    load params;
    
    load('./voc_data/Train.mat');
    load('./voc_data/Test.mat');
    X_a = TrImage';
    X_b = TrText';
    
    train_Y = SY2MY(trY)';
    train_Y(find(train_Y == -1)) = 0;
    
    Test_a = TeImage';
    Test_b = TeText';
    test_Y = teY;
%     test_Y(find(test_Y == -1)) = 0;
    
    %% training parameters
    lambda_1 = 0.1;
    lambda_2 = 0.001;
    ite = 5;
    
    
    [dim_a, num] = size( X_a );
    [dim_b, num] = size( X_b );
    [c, num] = size(train_Y);
    
    %% Parameters setting
    c = 80;
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
        
        [Alphah, Alphal, XH_t, XL_t, Dh, Dl, Wh, Wl, Uh, Ul, f] = coupled_DL_recoupled(Alphah, Alphal, X_a, X_b, Dh, Dl, Wh, Wl, par);
        clear XH_t XL_t;

        %% now use the sparse representations for new modal learning
        lambda_1 = 0.1;
        lambda_2 = 0.001;
        ite = 5;
    
        [W_a, W_b] = LCFS_ite( Alphah', Alphal', trY, lambda_1, lambda_2, ite);
        
        %% test
        % first project the testing samples on the common feature space
        Alphah = full(mexLasso(Test_a, Dh, param));
        Alphal = full(mexLasso(Test_b, Dl, param));
        projected_Y_a = Alphah' * W_a;
        projected_Y_b = Alphal' * W_b;

        
        map1 = calculateMAP( projected_Y_a, projected_Y_b, test_Y );
        str = sprintf( 'The MAP of image as query is %f%%\n', map1 *100 );
        disp(str);

        map2 = calculateMAP( projected_Y_b, projected_Y_a, test_Y );
        str = sprintf( 'The MAP of text as query is %f%%\n', map2 *100 );
        disp(str);  
    
end