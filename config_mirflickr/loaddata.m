function [xTr, yTr, xTe, yTe, valIdx] = loaddata(dataFolder, dimen)

[X] = rpSVD(dataFolder, dimen);

if ispc
    matchStr = regexp(ls_win(dataFolder), '\w*_train_tags.hvecs', 'match');
elseif isunix
    matchStr = regexp(ls(dataFolder), '\w*_train_tags.hvecs', 'match');
end
yTr = vec_read(strcat(dataFolder, matchStr{1}));
yTr = double(yTr');

if ispc
    matchStr = regexp(ls_win(dataFolder), '\w*_test_tags.hvecs', 'match');
elseif isunix
    matchStr = regexp(ls(dataFolder), '\w*_test_tags.hvecs', 'match');
end
yTe = vec_read(strcat(dataFolder, matchStr{1}));
yTe = double(yTe');

nTr = size(yTr, 2);
nTe = size(yTe, 2);

xTr = X(:, 1:nTr);
xTe = X(:, nTr+1:end);
clear('X');

valIdx = false(1, nTr);
valIdx(randsample(nTr, nTe)) = true;

save([dataFolder, 'data,dimen=', num2str(dimen), '.mat'], 'xTr', 'yTr', 'xTe', 'yTe', 'valIdx', '-v7.3');
