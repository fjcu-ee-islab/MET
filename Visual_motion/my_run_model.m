function my_run_model(InputFileDir, ResultDir)

% InputFileDir = fullfile(MainDir, 'Data', 'DVS_Gesture_dataset', 'flops_count_test/');
% ResultDir = fullfile(MainDir, 'Results', 'flops_count_test/');

addpath(genpath('functions'))
addpath(genpath('functions/AedatTools'))


MainDir = pwd;

files = dir([InputFileDir '*.aedat']);
for fLoop = 1:numel(files)
    
%     close all
    
    file = [InputFileDir files(fLoop).name];
    [fp,fn,fe] = fileparts(file);
   
    
    %% Layer 1
    rname = '_layer1.mat';
    ResultfilesName = [fn rname];


    cd(ResultDir)
    %{
    if exist(ResultfilesName, 'file')
        cd(MainDir)
        continue;
    end
    %}
    
    % Read in aedat data (Requires events, frames, and IMU data) (aedat4 converted to .mat via aedat4tomat.py)
    aedat = struct;
    aedat.importParams.filePath = file;
    aedat = ImportAedat(aedat);
    
    for event_num = 1:length(aedat.data.polarity.timeStamp)
        if(aedat.data.polarity.x(event_num) == 0)
            aedat.data.polarity.x(event_num) = 1;
        end
        if(aedat.data.polarity.y(event_num) == 0)
            aedat.data.polarity.y(event_num) = 1;
        end
    end
    
    TD.p = aedat.data.polarity.polarity;
    TD.x = aedat.data.polarity.x;
    TD.y = aedat.data.polarity.y;
    TD.ts = aedat.data.polarity.timeStamp;

    TD.p = double(TD.p);
    TD.x = double(TD.x);
    TD.y = double(TD.y);
    TD.ts = double(TD.ts);
    
    TD.p = TD.p(1:1000,1);
    TD.x = TD.x(1:1000,1);
    TD.y = TD.y(1:1000,1);
    TD.ts = TD.ts(1:1000,1);

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


    %{
    if ~exist('Results', 'dir')
        mkdir('Results')
    end
    %}

    %%% Display the results:
    %show_result(layer1_output)

    %%% Save the results:
    save([ResultDir, fn, '_layer1'], 'layer1_output')

    %clearvars -except files fLoop InputFileDir MainDir ResultDir layer1_output fn

    
    %%
    filter_size = [5,5];
    use_delays = 0; %should second layer neurons have delays?

    filter_speeds = sqrt(2)*0.02*sqrt(2).^(0:7); %pixels per millisecond
    filter_directions = -135:45:180;
    filters = generate_filters(filter_speeds, filter_directions, filter_size);

    % set all the delays to zero
    [num_speeds, num_directions] = size(filters);
    if use_delays == 0
        for speed_index = 1:num_speeds
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
    %show_result(layer2_output)

    %%% Save the results:
    save([ResultDir, fn, '_layer2'], 'layer2_output');
    
    clearvars -except files fLoop InputFileDir MainDir ResultDir

end
