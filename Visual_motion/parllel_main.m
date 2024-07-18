clear all
currentFolder=pwd;
parpool('local')
spmd(4)
    if labindex == 1 
         InputFileDir = fullfile(currentFolder, 'Data', 'DVS_Gesture_dataset', 'flops_count_test/');
         ResultDir = fullfile(currentFolder, 'Results', 'flops_count_test/');
         my_run_model(InputFileDir, ResultDir);
    end
end
