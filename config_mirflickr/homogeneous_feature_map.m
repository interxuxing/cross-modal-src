function PhiX = homogeneous_feature_map(X, order, L, kernel_type, gamma, normalize) 
% homogeneous_feature_map
%
% INPUT:
% X is a matrix nDim x nSamples of data
% order is the term "n" of [1].
% L For an explanation see [1].
% kernel_type can be 'intersection' or 'chi-square' 
% normalize 0 (default)
%           1 This normalizes the data using the gamma-norm
%
% OUTPUT:
% PhiX  the mapped data
%
% [1] Vedaldi and Zisserman,
%     Efficient additive kernels via explicit feature maps, PAMI, 2011
%  Writen by Alessandro Bergamo	aleb@cs.dartmouth.edu


if nargin < 6
    normalize = 0;
end

nDim = size(X,1);
nSamples = size(X, 2);

% normalize the data
if normalize
   tt = (sum(abs(X).^gamma, 1)).^(1/gamma);
   tt(find(tt == 0)) = 1; % to handle the case when a vector is zero
   X = bsxfun(@times, X, 1./tt); 
end

sqrtXL = sqrt(L) .* (X.^(gamma/2)); % matrix size(X)
J = 1:order; % 1 x order
JL = L .* J; % 1 x order
Lambda = JL; 
index = find(X==0);
X(index) = 10^-20;
lnX = log(X); % matrix size(X)
switch kernel_type
    case 'intersection'  % intersection kernel
        sqrtKappa0 = sqrt(2.0 / pi);
        Kappa = (2.0/pi) * 1./(1+4*Lambda.^2);
    case 'chi-square' % Chi-square kernel
        sqrtKappa0 = 1.0;
        Kappa = 2.0 ./ (exp(-pi*Lambda) + exp(pi*Lambda));
    otherwise
        fprintf('ERROR! kernel type "%s" not supported\n', kernel_type);
        keyboard;
end
sqrt2Kappa = sqrt(2*Kappa);
% output
PhiX = zeros((2*order+1)*nDim, nSamples);
% for each dimensione of the data set
for i=1:nDim
    offset = (i-1)*(2*order+1);
    % let's calculate for lambda=0
    PhiX(offset+1,:) = sqrtKappa0 * sqrtXL(i,:);
    % let's calculate the real and the imaginary part for lambda=jL where j=1:order
    part1 = repmat(sqrt2Kappa', [1,nSamples]) .* repmat(sqrtXL(i,:), [order,1]);
    part2 = JL' * lnX(i,:); 
    PhiX(offset+2*(1:order), :) = part1 .* cos(part2);
    PhiX(offset+1+2*(1:order), :) = part1 .* sin(part2);
end
end
