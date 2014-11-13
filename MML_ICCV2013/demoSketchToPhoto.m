% Coupled Dictionary and Feature Space Learning with Application to Cross-Domain Image Synthesis and Recognition.  
% De-An Huang and Yu-Chiang Frank Wang
% IEEE International Conference on Computer Vision (ICCV), 2013.
%
% Contact: Yu-Chiang Frank Wang (ycwang@citi.sinica.edu.tw)
%
% Demo Sketch to Photo Recognition

    
%% Access directories
addpath(genpath('SPAMS'));
addpath(genpath('Data'));
addpath(genpath('YIQRGB'));
                
%% Load parmeters and dataset
load params      
load CUFS_CUHK      

%% Randomly choose training and testing samples
P = cell(188,1);
S = cell(188,1);
for k = 1:100
    P{k} = testing_photo{k};
    S{k} = testing_sketch{k};
end
for k = 1:88
    P{k+100} = training_photo{k};
    S{k+100} = training_sketch{k};
end
R = randperm(188);
for k = 1:100
    testing_photo{k} = P{R(k)};
    testing_sketch{k} = S{R(k)};
end
for k = 1:88
    training_photo{k} = P{R(k+100)};
    training_sketch{k} = S{R(k+100)};
end

%% Parameters setting
par.mu = par.mu*1;
par.K 	= 50;
param.K = 50;
par.L	= 50;
param.L = 50;
param.lambda        = par.lambda1; % not more than 20 non-zeros coefficients
param.lambda2       = par.lambda2;
param.mode          = 2;       % penalized formulation
param.approx=0;

%% Prework on training samples
size1 = size(training_photo{1},1);
size2 = size(training_photo{1},2);
XH_t = zeros(size1*size2,88);
XL_t = zeros(size1*size2,88);
for k = 1:88
    fprintf('%d\n',k);
%     img_yiq = RGB2YIQ(double(training_photo{k}));
    XH_t(:,k) = reshape(img_yiq(:,:,1),[size1*size2 1] );
    
    tmp = double(training_sketch{k});
    if size(tmp,3) > 1
%         img_yiq = RGB2YIQ(tmp);
        tmp = img_yiq(:,:,1);
    end
    XL_t(:,k) = reshape(tmp,[size1*size2 1]); 
end

XH_t = XH_t - repmat(mean(XH_t), [size(XH_t,1) 1]);
XL_t = XL_t - repmat(mean(XL_t), [size(XL_t,1) 1]);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Start our proposed algorithm on training samples %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% Intialize D,A, and W(U)
D = mexTrainDL([XH_t;XL_t], param); % 两个字典Dh, Dl一起训练
Dh = D(1:size(XH_t,1),:); 
Dl = D(size(XH_t,1)+1:end,:);
Wl = eye(size(Dh, 2));
Wh = eye(size(Dl, 2));
Alphah = mexLasso([XH_t;XL_t], D, param);
Alphal = Alphah;
clear D;
	% Iteratively solve D,A, and W (U)
[Alphah, Alphal, XH_t, XL_t, Dh, Dl, Wh, Wl, Uh, Ul, f] = coupled_DL_recoupled(Alphah, Alphal, XH_t, XL_t, Dh, Dl, Wh, Wl, par);
clear XH_t XL_t;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Finish our proposed alorithm on training samples %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Prework on testing samples
XH_t = zeros(size1*size2,100);
XL_t = zeros(size1*size2,100);
for k = 1:100  
    img_yiq = RGB2YIQ(double(testing_photo{k}));
    XH_t(:,k) = reshape(img_yiq(:,:,1),[size1*size2 1] );
    
    tmp = double(testing_sketch{k});
    if size(tmp,3) > 1
        img_yiq = RGB2YIQ(tmp);
        tmp = img_yiq(:,:,1);
    end
    XL_t(:,k) = reshape(tmp,[size1*size2 1]);
            
end


XH_t = XH_t - repmat(mean(XH_t), [size(XH_t,1) 1]);
XL_t = XL_t - repmat(mean(XL_t), [size(XL_t,1) 1]);


%% Project the testing samples on the common feature space
Alphah = full(mexLasso(XH_t, Dh, param));
Alphal = full(mexLasso(XL_t, Dl, param));
photo_coeffs = inv(Uh) * Alphah;
sketch_coeffs = inv(Ul) * Alphal;


%% Direct NN method to find datapairs on common feature space
correct = 0;
for test = 1:100
    dist = zeros(100,1);
    for train = 1:100
        dist(train) = norm(photo_coeffs(:,train) - sketch_coeffs(:,test));
    end
    minD = min(dist);
    predict = find(dist == minD);
    if predict(1) == test
        correct = correct + 1;
    end    
end

fprintf('Recogntion Rate: %d\n',correct);
%% correlation based
%{
correct1 = 0;
for test = 1:100
    corr = zeros(100,1);
    for train = 1:100
        tmp = corrcoef(photo_coeffs(:,train) , sketch_coeffs(:,test));
        corr(train) = tmp(2);
    end
    %corr
    maxC = max(corr);
    predict = find(corr == maxC);
    if predict(1) == test
        correct1 = correct1 + 1;
    end  
    %pause;
end
%}              
