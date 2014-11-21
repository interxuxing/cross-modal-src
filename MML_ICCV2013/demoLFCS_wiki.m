%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% demoLFCS_wiki: Learning Coupled Feature Spaces for Cross-modal Matching
% 	on wiki dataset
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function demoLFCS_wiki
dir_wiki_data = './wiki_data';
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

%% training parameters for LCFS method
lambda_1 = 0.1;
lambda_2 = 0.001;
ite = 5;

[W_a, W_b] = LCFS_ite( X_a', X_b', train_Y, lambda_1, lambda_2, ite);

projected_Y_a = Test_a' * W_a;
projected_Y_b = Test_b' * W_b;


map1 = calculateMAP( projected_Y_a, projected_Y_b, Y_te );
str = sprintf( 'The MAP of image as query is %f%%\n', map1 *100 );
    disp(str);

map2 = calculateMAP( projected_Y_b, projected_Y_a, Y_te );
str = sprintf( 'The MAP of text as query is %f%%\n', map2 *100 );
    disp(str);
    
map3 = calculatePrecisionScope( projected_Y_a, projected_Y_b, Y_te );
map4 = calculatePrecisionScope( projected_Y_b, projected_Y_a, Y_te );

frpintf('finished! \n');

end