% Coupled Dictionary and Feature Space Learning with Application to Cross-Domain Image Synthesis and Recognition.  
% De-An Huang and Yu-Chiang Frank Wang
% IEEE International Conference on Computer Vision (ICCV), 2013.
%
% Contact: Yu-Chiang Frank Wang (ycwang@citi.sinica.edu.tw)
%
% Main Function of Coupled Dictionary Learning
% Input:
% Alphap,Alphas: Initial sparse coefficient of two domains
% Xp    ,Xs    : Image Data Pairs of two domains
% Dp    ,Ds    : Initial Dictionaries
% Wp    ,Ws    : Initial Projection Matrix
% par          : Parameters 
%
%
% Output
% Alphap,Alphas: Output sparse coefficient of two domains
% Dp    ,Ds    : Output Coupled Dictionaries
% Up    ,Us    : Output Projection Matrix for Alpha
% 

function [Alphap, Alphas, Xp, Xs, Dp, Ds, Wp, Ws, Up, Us, f] = coupled_DL_recoupled(Alphap, Alphas, Xp, Xs, Dp, Ds, Wp, Ws, par)

%% parameter setting

[dimX, numX]        =       size(Xp);
dimY                =       size(Alphap, 1);
numD                =       size(Dp, 2);
rho                 =       par.rho;
lambda1             =       par.lambda1;
lambda2             =       par.lambda2;
mu                  =       par.mu;
sqrtmu              =       sqrt(mu);
nu                  =       par.nu;
nIter               =       par.nIter;
t0                  =       par.t0;
epsilon             =       par.epsilon;
param.lambda        = 	    lambda1; % not more than 20 non-zeros coefficients
param.lambda2       =       lambda2;
param.mode          = 	    2;       % penalized formulation
param.approx=0;
param.K = par.K;
param.L = par.L;
f = 0;

%% Initialize Us, Up as I

Us = Ws; 
Up = Wp; 

%% Iteratively solve D A U

for t = 1 : nIter

    %% Updating Alphas and Alphap
    f_prev = f;
    Alphas = mexLasso([Xs;sqrtmu * full(Alphap)], [Ds; sqrtmu * Ws],param);
    Alphap = mexLasso([Xp;sqrtmu * full(Alphas)], [Dp; sqrtmu * Wp],param);
    dictSize = par.K;

    %% Updating Ds and Dp 
    for i=1:dictSize
       ai        =    Alphas(i,:);
       Y         =    Xs-Ds*Alphas+Ds(:,i)*ai;
       di        =    Y*ai';
       di        =    di./(norm(di,2) + eps);
       Ds(:,i)    =    di;
    end

    for i=1:dictSize
       ai        =    Alphap(i,:);
       Y         =    Xp-Dp*Alphap+Dp(:,i)*ai;
       di        =    Y*ai';
       di        =    di./(norm(di,2) + eps);
       Dp(:,i)    =    di;
    end

    %% Updating Ws and Wp => Updating Us and Up
    ts = inv(Up)*Alphap;
    tp = inv(Us)*Alphas;
    Us = (1 - rho) * Us  + rho * Alphas * ts' * inv( ts * ts' + par.nu * eye(size(Alphas, 1)));
    Up = (1 - rho) * Up  + rho * Alphap * tp' * inv( tp * tp' + par.nu * eye(size(Alphap, 1)));
    Ws = Up * inv(Us); 
    Wp = Us * inv(Up);

    %% Find if converge

    P1 = Xp - Dp * Alphap;
    P1 = P1(:)'*P1(:) / 2;
    P2 = lambda1 *  norm(Alphap, 1);    
    P3 = Alphas - Wp * Alphap;  % Wp = Us * inv(Up)
    P3 = P3(:)'*P3(:) / 2;
    P4 = nu * norm(Up, 'fro');
    fp = 1 / 2 * P1 + P2 + mu * (P3 + P4);
    
    P1 = Xs - Ds * Alphas;
    P1 = P1(:)'*P1(:) / 2;
    P2 = lambda1 *  norm(Alphas, 1);    
    P3 = Alphap - Ws * Alphas; % Wp = Us * inv(Up);
    P3 = P3(:)'*P3(:) / 2;
    P4 = nu * norm(Us, 'fro');  %%
    fs = 1 / 2 * P1 + P2 + mu * (P3 + P4);
    
    f = fp + fs;
	
    %% if converge then break
    ratio = abs(f_prev - f) / f;
    fprintf('Iter: %d, f: %f, f_prev: %f, ratio: %f \n', t, f, f_prev, ratio);
    if (ratio < epsilon)
        break;
    end

end
    
