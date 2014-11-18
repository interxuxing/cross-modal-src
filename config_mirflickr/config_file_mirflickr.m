%%%%% Global configuration file %%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% DIRECTORIES - please change if copying the code to a new location
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
DEVICE = 2; % 1, laptop 2, desktop (ubuntu, yoshinaga)

if 1 == DEVICE
    % Path for laptop
    IMAGE_SOURCE_DIR = '';
    IMAGE_ANNOTATION_DIR = 'C:\workspace\program\image-annotation\others\Annotation_Demo\mirflickr.20101118';
    RUN_DIR = './';
    MODEL_DIR = './model/mirflickr';
    LOGFILE_DIR = './log';
    DIM = 500;
elseif 2 == DEVICE
    % Path for desktop
    IMAGE_SOURCE_DIR = '';
    IMAGE_ANNOTATION_DIR = '/media/Data/dataset/data/mirflickr/mirflickr.20101118';
    RUN_DIR = './';
    MODEL_DIR = './model/mirflickr';
    LOGFILE_DIR = './log';
    DIM = 500;
else
    error('Error value for DEVICE, should be either 1 or 2!');
end