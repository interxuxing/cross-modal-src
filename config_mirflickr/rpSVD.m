function [X] = rpSVD(dataFolder, dimen);


featNames = {'DenseHue.hvecs', 'DenseHueV3H1.hvecs', 'DenseSift.hvecs', 'DenseSiftV3H1.hvecs', ...
    'Gist.fvec', 'HarrisHue.hvecs', 'HarrisHueV3H1.hvecs', 'HarrisSift.hvecs', 'HarrisSiftV3H1.hvecs', ...
    'Hsv.hvecs32', 'HsvV3H1.hvecs32', 'Lab.hvecs32', 'LabV3H1.hvecs32', 'Rgb.hvecs32', 'RgbV3H1.hvecs32'};
	
tic;
X = [];
for feat = featNames
    % loop to load each type of feature, matching with name in feaNames¡£
    % in windows os, can be done like follows:
    % A = dir(dataFolder); B = struct2cell(A); 
    % C =B(1,:); regexp(C, ...)
    feat = char(feat);
        
    if ispc
        matchStr = regexp(ls_win(dataFolder), ['\w*_train_', feat], 'match');
    elseif isunix
        matchStr = regexp(ls(dataFolder), ['\w*_train_', feat], 'match');
    end
        tmp = vec_read(strcat(dataFolder, matchStr{1}));
	combine = double(tmp);
	[nTr, d0] = size(tmp);

    if ispc
        matchStr = regexp(ls_win(dataFolder), ['\w*_test_', feat], 'match');
    elseif isunix
        matchStr = regexp(ls(dataFolder), ['\w*_test_', feat], 'match');
    end
    
	tmp = vec_read(strcat(dataFolder, matchStr{1}));
	nTe = size(tmp, 1);
	combine = [combine; double(tmp)];

	if ~strcmp(feat, 'Gist.fvec')
		combine = spdiags(sum(abs(combine), 2)+eps, 0, size(combine, 1), size(combine, 1))*combine;
    end	
    if strcmp(feat, 'Gist.fvec')
		PhiX = combine';
    else
        PhiX = homogeneous_feature_map(combine', 1, 0.6, 'intersection', 1, 1);
    end

	homoDim = size(PhiX, 1);
	if size(PhiX, 1) <= dimen
		X = [X; PhiX];
		fprintf('feature = %s, dimen = %d, homoDim = %d, RPDim = %d\n', feat, d0, size(PhiX, 1), size(PhiX, 1));
	else
		[d, n] = size(PhiX);
		R = randn(n, dimen);
        Y = PhiX*R;
        [Q, ~] = qr(Y, 0);
		PhiX = Q'*PhiX;
        X = [X; PhiX];
		fprintf('feature = %s, dimen = %d, homoDim = %d, RPDim = %d\n', feat, d0, homoDim, dimen);
	end
end
toc;
