%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function visualize_wiki_img(img_idx) is a script
%   to visualize the image with its class, text in
%   the test set
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function visualize_wiki_img(img_idx)

if nargin ~= 1
    img_idx = [1:10]; % if no img_idx input, show top-10 images
end

IMG_DIR = './images';
TXT_DIR = './texts';

testlist_file = 'testset_txt_img_cat.list';
classlist_file = 'categories.list';

%% read the testlist and categories (classes)
fid = fopen(classlist_file);
Content = textscan(fid, '%s\n');
fclose(fid);
Classes = Content{1};

fid = fopen(testlist_file);
Content = textscan(fid, '%s\t%s\t%s\n');
fclose(fid);
TextTestList = Content{1}; % cell
ImageTestList = Content{2}; % cell
ClassTestList = Content{3}; % cell


%% now show the image with its class and text
for i = img_idx
    classnum_img = ClassTestList{i};
    class_img = Classes{str2double(classnum_img)};
    
    image_img = ImageTestList{i};
    text_img = TextTestList{i};
    
    image_img_name = fullfile(IMG_DIR, class_img, [image_img, '.jpg']);
    text_img_name = fullfile(TXT_DIR, [text_img, '.xml']);
    
    text_info = parse_text_xml(text_img_name, 'text');
    
    Im = imread(image_img_name);
    figure(i), hold on, imshow(Im), axis off;
    title(class_img);
    hold off;
    
    fprintf('Image: %s \n\n', image_img);
    fprintf('Class: %s \n\n', class_img);
    fprintf('Text: %s \n', text_info);
end



end


function text_info = parse_text_xml(fileXml, keyItem)
% this function is a parser to return the content in fileXml with keyItem
% since the xml file is not standard xml file, so here we use textscan and 
% string functions to return the text content

fid = fopen(fileXml);
M = textscan(fid, '%s');
fclose(fid);
Content = M{1};


startItem = ['<',keyItem,'>'];
endItem = ['</', keyItem, '>'];

% find the start idx
res = strcmp(startItem, Content);
startIdx = find(res == 1);

res = strcmp(endItem, Content);
endIdx = find(res == 1);

text_info = '';

if endIdx > startIdx
    for i = [startIdx+1 : endIdx-1]
        text_info = [text_info, Content{i}];
    end
    text_info = [text_info, '\n']; 
else
    error(sprintf('no %s matches in the xml file! \n', keyItem));
end


end