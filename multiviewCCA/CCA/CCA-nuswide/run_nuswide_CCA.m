% This a script to run Multiview-CCA on NUS-WIDE dataset

addpath('../CCA/');

k = 500; % only preserve top k engen-values when doing svd for tag feature
d = 256; % dimension of projected CCA

% first load V (visual feature matrix, nxd), C (concept matrix, nxc)
[VTr, VTe, CTr, CTe, TTr, TTe] = load_multiple_feature_nuswide('config_file_nuswide_CCA');

% then we first do sparse-SVD for T
T = [TTr; TTe];
[T_U, T_S, T_V] = svds(T);
T_SVD = T_U * T_S;

T_k = T_SVD(:, 1:k);

TTr_k = T_k(1:size(TTr,1), :);
TTe_k = T_k(size(TTr,1)+1:end, :);

%% do multiview CCA
[Wx, D] = CCA3(VTr, TTr_k, CTr);

%% now project top-d for each view
d_v = size(VTr, 2);
d_t = size(TTr, 2);
d_c = size(CTr, 2);

index = [ones(d_v,1);ones(d_t,1)*2;ones(d_c,1)*3];

Wx_V = Wx(index == 1, 1:d);
Wx_T = Wx(index == 2, 1:d);
Wx_C = Wx(index == 3, 1:d);

Dx = D(1:d, 1:d);

%% then we can use Xv*Wx_V*Dx, Xt*Wx_T*Dx, Xc*Wx_C*Dx to get projected CCA features
%   for views V, T, C