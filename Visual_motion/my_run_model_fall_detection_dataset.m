function my_run_model_fall_detection_dataset(InputFileDir, ResultDir)
% InputFileDir = './Data/Fall Detection/worker01/';
% ResultDir = './Results/Fall Detection Dataset Result/worker01/';
addpath(genpath('functions'))

MainDir = pwd;

files = dir([InputFileDir '*.mat']);

for fLoop = 1:numel(files)
    
    close all
    
    file = [InputFileDir files(fLoop).name];
    [fp,fn,fe] = fileparts(file);
    
    cd(ResultDir)
    
    %% Layer 1
    L1_rname = '_layer1.mat';
    L1_ResultfilesName = [fn L1_rname]; 
       
    if ~exist(L1_ResultfilesName, 'file')
        % Read in aedat data (Requires events, frames, and IMU data) (aedat4 converted to .mat via aedat4tomat.py)
        load(file);
        
        % Be care for the size of event data must be (event_num x 1)
        TD.p = transpose(double(p));
        TD.x = transpose(double(x));
        TD.y = transpose(double(y));
        TD.ts = transpose(double(t));
        
        clearvars p x y t 

        for event_num = 1:length(TD.ts)
            if(TD.x(event_num) == 0)
                TD.x(event_num) = 1;
            end
            if(TD.y(event_num) == 0)
                TD.y(event_num) = 1;
            end
        end
        
        %generate the filter speeds and directions
        filter_speeds = sqrt(2)*0.02*sqrt(2).^(0:7); %pixels per millisecond
        filter_directions = -135:45:180;
    
        %% Setup the layer parameters
        % FilterSize        = 5;
        filter_size         = [3,3];
        spike_threshold     = 50;
        reset_potential     = -128;
        decay_rate          = filter_speeds*50*1e-3;
    
    
        %split the events into positive and negative polarity
        polarities = unique(TD.p);
        pos_indices = (TD.p ~= polarities(2)); %assume the first index is the positive index
        TDpos.x     = TD.x(pos_indices);
        TDpos.y     = TD.y(pos_indices);
        TDpos.ts    = TD.ts(pos_indices);
    
        neg_indices = (TD.p ~= polarities(1)); %assume the first index is the positive index
        TDneg.x     = TD.x(neg_indices);
        TDneg.y     = TD.y(neg_indices);
        TDneg.ts    = TD.ts(neg_indices);
        tic
        layer1_output = run_layer_1(TDneg, TDpos, filter_speeds, filter_directions, filter_size, spike_threshold, decay_rate, reset_potential, fn);
    
        %%% Display the results:
        % show_result(layer1_output)
    
        %%% Save the results:
        save([ResultDir, filesep, fn, '_layer1'], 'layer1_output')
        clearvars -except files fLoop InputFileDir MainDir ResultDir layer1_output fn
    else    
        load(L1_ResultfilesName);
        fprintf('Find L1_Resultfile: ', L1_ResultfilesName)
    end
    
    

    %% Layer 2
    L2_rname = '_layer2.mat';
    L2_ResultfilesName = [fn L2_rname];

    if ~exist(L2_ResultfilesName, 'file')
        filter_size = [5,5];
        use_delays = 0; %should second layer neurons have delays?
    
        filter_speeds = sqrt(2)*0.02*sqrt(2).^(0:7); %pixels per millisecond
        filter_directions = -135:45:180;
        filters = generate_filters(filter_speeds, filter_directions, filter_size);
    
        % set all the delays to zero
        [num_speeds, num_directions] = size(filters);
        if use_delays == 0
            parfor speed_index = 1:num_speeds
                for direction_index = 1:num_directions
                    filters{speed_index,direction_index}(:) = 0;
                end
            end
        end
    
        spike_threshold      = 60;
        decay_rate           = ((filter_speeds*spike_threshold*1e-3)/100); %use a much slower decay in layer 2. pretty much zero
        reset_potential      = -50;
    
        fprintf('Simulating Layer 2: Processing merged stream of %i events\n', length(layer1_output.ts));
        layer2_output = spike_layer_2(layer1_output, filters, spike_threshold, decay_rate, reset_potential, fn);
    
        %%% Display the results:
        % show_result(layer2_output)
    
        %%% Save the results:
        save([ResultDir, filesep, fn, '_layer2'], 'layer2_output')
        
        clearvars -except files fLoop InputFileDir MainDir ResultDir
        cd(MainDir)
    else
        fprintf('Find L2_Resultfile: ', L2_ResultfilesName)
        clearvars -except files fLoop InputFileDir MainDir ResultDir
        cd(MainDir)
        continue
    end    
end
