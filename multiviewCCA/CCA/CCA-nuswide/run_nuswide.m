addpath('util/')
addpath('preprocess/')
addpath('baseline/')
addpath('fasttag/')

topK = 5;

[xTr, xTe, yTr, yTe, valIdx] = load_multiple_feature_nuswide('config_file_nuswide');
xTr = double(xTr');
xTe = double(xTe');
yTr = double(yTr');
yTe = double(yTe');

xTr = [xTr; ones(1, size(xTr, 2))];
xTe = [xTe; ones(1, size(xTe, 2))];

% fasttag
[W_fasttag, results] = fasttag(xTr, yTr, xTe, yTe, topK, valIdx);


scorePredictTestLabels = results.pred;

%%% evaluate with standard measure and save results
 
resultsTag = evaluatePR(yTe, scorePredictTestLabels, 5, 'tag');
fprintf('... For tag measure: \n\t P %f, R %f, N+ %d \n', resultsTag.prec, resultsTag.rec, resultsTag.retrieved);

resultsImage = evaluatePR(yTe, scorePredictTestLabels, 5, 'image');
fprintf('... For image measure: \n\t P %f, R %f, N+ %d \n', resultsImage.prec, resultsImage.rec, resultsImage.retrieved);

%%% evaluate AUC and Haming loss
[tpr, fpr] = evalROC(exp(scorePredictTestLabels'), target');

[area, area2] = evalAUC(fpr, tpr);

hardPREDs = zeros(size(scorePredictTestLabels)) -1;
for n = 1:size(yTe, 2)
        gt = yTe(:, n);
        confidence = scorePredictTestLabels(:, n);
        [so, si] = sort(-confidence);
        si = si(1:topK);
	hardPREDs(si, n) = 1;
end

target = yTe;
target(target == 0) = -1;
HammingLoss = Hamming_loss(hardPREDs, target);


fprintf('Finished fasttag for nus-wide \n');