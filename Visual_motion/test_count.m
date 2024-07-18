clear
clc

%% Example 1: MATLAB Scripts
InputFileDir = './Data/DVS_Gesture_dataset/DvsGesture/flops_count_test/';
ResultDir = './Results/flops_count_test/';

profile on
my_run_model(InputFileDir, ResultDir);

profileStruct = profile('info');
[flopTotal,Details]  = FLOPS('my_run_model','valuable',profileStruct);
