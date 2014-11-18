%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Description
%  compute PrecisionScope in range top-50
%   top-1000 retrieved samples

%Input
%  queryset     n*dim_a data matrix 
%  targetset     n*dim_b data matrix
%  test_Y       n*1 label vector

%Output
%  map   a vector of MAP scores with [50:50:1000] range
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function map = calculatePrecisionScope( queryset, targetset, test_Y )
    
    Dist = CosineDist(queryset, targetset);
    [asDist index] = sort(Dist, 2, 'ascend');
    classIndex = test_Y(index);
    
    AP = [];    
    [num c] = size( queryset );
    
    scales = [50:50:1000];
    epsilon = 1e-3;
    for s = 1 : length(scales)
        AP = [];
        scale = scales(s);
        for k = 1 : num
            ClassIndex = find(classIndex(k, :) == test_Y(k));
            reClassIndex = ClassIndex(ClassIndex <= scale);

            relength = length(reClassIndex);
            AP =[AP relength/scale];
        end
        map(s) = mean (AP);
    end
    
end

