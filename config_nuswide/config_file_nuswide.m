%%%%% Global configuration file %%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% DIRECTORIES - please change if copying the code to a new location
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
DEVICE = 2; % 1, laptop 2, desktop (ubuntu, yoshinaga)

if 1 == DEVICE
    % Path for laptop
    IMAGE_SOURCE_DIR = '';
    IMAGE_ANNOTATION_DIR = 'C:\workspace\program\image-annotation\benchmark-dataset\NUS-WIDE\NUS-WIDE\NUS-WIDE-crop\Low_Level_Features';
    RUN_DIR = './';
    MODEL_DIR = './model/nuswide';
    LOGFILE_DIR = './log';
elseif 2 == DEVICE
    % Path for desktop
    IMAGE_SOURCE_DIR = '';
    IMAGE_ANNOTATION_DIR = '/home/limu/workspace/data/NUS-WIDE-crop/Low_Level_Features';
    RUN_DIR = './';
    MODEL_DIR = './model';
    LOGFILE_DIR = './log';
else
    error('Error value for DEVICE, should be either 1 or 2!');
end


