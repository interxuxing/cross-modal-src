%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function visualize_miflickr_img(img_idx) is a script 
%   to visualize the image with its annotations, tags in
%   the test set
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function visualize_mirflickr_img(img_idx)

% if nargin ~= 1
%     img_idx = [1:10]; % if no img_idx input, show top-10 images
% end

%% set the paths
Src_Feature_Dir = 'C:\workspace\program\image-annotation\others\Annotation_Demo\mirflickr.20101118';
Img_Dir = 'C:\workspace\program\image-annotation\others\Annotation_Demo\mirflickr';

%% parse the imagelist, classes, tags vocabulary
testlist_file = 'mirflickr_test_list.txt';
classlist_file = 'mirflickr_classes.txt';
textlist_file = 'mirflickr_dictionary.txt';

% read the testlist of imagename
fid = fopen(fullfile(Src_Feature_Dir, testlist_file));
Content = textscan(fid, '%s\n');
fclose(fid);
ImageTestList = Content{1};

% read the categories (classes) vocabulary
fid = fopen(fullfile(Src_Feature_Dir, classlist_file));
Content = textscan(fid, '%s\n');
fclose(fid);
Classes = Content{1};

% read the tags vocabulary
fid = fopen(fullfile(Src_Feature_Dir, textlist_file));
Content = textscan(fid, '%s\n');
fclose(fid);
Tags = Content{1};


%% then parse annotations (classes), tags for each image
testanno_file = 'mirflickr_test_classes.txt';
testtags_file = 'mirflickr_test_tags.hvecs';

Content = dlmread(fullfile(Src_Feature_Dir, testanno_file));
AnnoTestList = double(Content);

TagsTestList = double(vec_read(fullfile(Src_Feature_Dir, testtags_file)));

%% now loop to show each image

% for i = img_idx
for i = 1 : size(AnnoTestList, 1)
    image_name = [ImageTestList{i}, '.jpg'];
    
    res = AnnoTestList(i, :);
    anno = Classes(res ~= 0);
    
    res = TagsTestList(i, :);
    tags = Tags(res ~= 0);
    
    if length(tags) < length(anno)
        continue;
    end
    
    
    Im = imread(fullfile(Img_Dir, image_name));
    figure(i), hold on, imshow(Im), hold off;
    
    fprintf('For image: %s: \n', image_name);
    fprintf('Annotations: ');
    for j = 1 : length(anno)
        fprintf('%s ', anno{j});
    end
    fprintf('\n');
    
    fprintf('Tags: ');
    for j = 1 : length(tags)
        fprintf('%s ', tags{j});
    end
    fprintf('\n');
    
end


end