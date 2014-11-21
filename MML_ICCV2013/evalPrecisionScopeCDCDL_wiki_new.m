%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% evalPrecisionScopeCDCDL_wiki_new: evaluate the precision-scope
%   on wiki dataset (new, similar as ICCV2013)
% Xing Xu
% Limu, Kyushu University, Japan
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function evalPrecisionScopeCDCDL_wiki_new
    clear;
    clc;
    
    model_dir = './model/wiki';
    if ~exist(model_dir, 'dir')
        mkdir(model_dir);
    end
    
    addpath(genpath('SPAMS_2.5'));
    dir_wiki_data = './wiki_data';
      
    %% load data   
    load params;
    
    load(fullfile(dir_wiki_data, 'raw_features_new.mat'));
    
    X_a = I_tr'; % X_a should be dim x nsamples
    X_b = T_tr';
    
    Test_a = I_te';
    Test_b = T_te';     
    
    train_Y = SY2MY(Y_tr); % train_Y is n x c
    train_Y(find(train_Y == -1)) = 0;
    test_Y = SY2MY(Y_te);
    test_Y(find(test_Y == -1)) = 0;
    
    % get dimension size for visual, textual
    [dim_a, num] = size( X_a );
    [dim_b, num] = size( X_b );
    [num, c] = size(train_Y);
    
    %% Parameters setting
    C = [220];
    NU = [0.0001, 0.001, 0.01, 0.1, 1, 10];
    MU = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10];
    
    %% loop to change the dictionary size c, for evaluation
%     c = 50;
best_map = 0; 

for c = C
    fprintf('Coupled dictionary learning with size c = %d \n', c);
    res_file = sprintf('wiki_CDCDL_%d_new.mat', c);
    
    if ~exist(fullfile(model_dir, res_file), 'file')
    
    par.mu = par.mu*1;
    par.K 	= c;
    param.K = c;
    par.L	= c;
    
    %% param is for SPMAS toolbox
    param.L = c;
    param.lambda        = par.lambda1; % not more than 20 non-zeros coefficients
    param.lambda2       = par.lambda2;
    param.mode          = 2;       % penalized formulation
    param.approx=0;
    param.verbose       = false;
    param.iter          = 100;
    
    
    %% Intialize D,A, and W(U)
    D = mexTrainDL([X_a;X_b], param); % �����ֵ�Dh, Dlһ��ѵ��
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

    [Alphah, Alphal, XH_t, XL_t, Dh, Dl, Wh, Wl, Uh, Ul, f] = coupled_DL_recoupled(Alphah, Alphal, X_a, X_b, Dh, Dl, Wh, Wl, par);
    clear XH_t XL_t;

    %% now use the sparse representations for new modal learning
    %% training parameters for LCFS method
    lambda_1 = 0.1;
    lambda_2 = 0.001;
    ite = 5;

    [W_a, W_b] = LCFS_ite( Alphah', Alphal', train_Y, lambda_1, lambda_2, ite);

    %% test
    %% first project the testing samples on the common feature space with sparse representation
    Alphah = full(mexLasso(Test_a, Dh, param));
    Alphal = full(mexLasso(Test_b, Dl, param));
    %% then project the sparse representation to annotation space
    projected_Y_a = Alphah' * W_a;
    projected_Y_b = Alphal' * W_b;

   
        
    else
        load(fullfile(model_dir, res_file));
    end
    
        
    map1 = calculateMAP( projected_Y_a, projected_Y_b, Y_te );
    str = sprintf( 'The MAP of image as query is %f%%\n', map1 *100 );
        disp(str);
        
    map2 = calculateMAP( projected_Y_b, projected_Y_a, Y_te );
    str = sprintf( 'The MAP of text as query is %f%%\n', map2 *100 );
        disp(str);
     
    map3 = calculatePrecisionScope( projected_Y_a, projected_Y_b, Y_te );
    map4 = calculatePrecisionScope( projected_Y_b, projected_Y_a, Y_te );    
    
    if best_map < (map1 + map2)/2
        best_map = (map1 + map2)/2;
         save(fullfile(model_dir, res_file), 'Dh', 'Dl', 'Wh', 'Wl', 'Uh', 'Ul', 'W_a', 'W_b', ...
            'projected_Y_a', 'projected_Y_b', 'test_Y');
    end
    
    fprintf('Finished CDCDL method with dictionary size %d \n', c);
end

fprintf('finished! \n');
end