clc; clear all; close all;

% Start a parallel pool
if isempty(gcp('nocreate'))
    parpool;
end

% Initialize tables for Layup 1, Layup 2, and Layup 3
initTable = table('Size', [0, 14], 'VariableTypes', repmat({'double'}, 1, 14), ...
                  'VariableNames', {'Load', 'Cycles', 'Actuator_AVG', 'Sensor_AVG', 'Amplitude_AVG', 'Frequency_AVG', ...
                                    'Actuator_Mean', 'Actuator_StdDev', 'Actuator_Energy', 'Sensor_Mean', 'Sensor_StdDev', ...
                                    'Sensor_Energy', 'Avg_Amplitude', 'Avg_Frequency'});

layup1Table = initTable;
layup2Table = initTable;
layup3Table = initTable;

% Create a cell array of directories
dirr = {'Layup1\L1_S11_F\PZT-data\*.mat',
        'Layup1\L1_S12_F\PZT-data\*.mat',
        'Layup1\L1_S18_F\PZT-data\*.mat',
        'Layup1\L1_S19_F\PZT-data\*.mat',
        'Layup2\L2_S11_F\PZT-data\*.mat',
        'Layup2\L2_S17_F\PZT-data\*.mat',
        'Layup2\L2_S18_F\PZT-data\*.mat',
        'Layup2\L2_S20_F\PZT-data\*.mat',
        'Layup3\L3_S11_F\PZT-data\*.mat',
        'Layup3\L3_S13_F\PZTdata\*.mat',
        'Layup3\L3_S14_F\PZT-data\*.mat',
        'Layup3\L3_S18_F\PZT-data\*.mat',
        'Layup3\L3_S20_F\PZT-data\*.mat'};


% Loop through each directory and load .mat files
for d = 1:length(dirr)
    matFiles = dir(dirr{d});
    for i = 1:length(matFiles)
        filePath = fullfile(matFiles(i).folder, matFiles(i).name);
        tempData = load(filePath);
        
        if isfield(tempData, 'coupon')
            load_data = tempData.coupon.load;
            cycles_data = tempData.coupon.cycles;
            
            % Initialize default values for missing fields
            avg_actuator_data = NaN;
            avg_sensor_data = NaN;
            avg_amplitude_data = NaN;
            avg_frequency_data = NaN;
            actuatorMean = NaN;
            actuatorStdDev = NaN;
            actuatorEnergy = NaN;
            sensorMean = NaN;
            sensorStdDev = NaN;
            sensorEnergy = NaN;

            if isfield(tempData.coupon, 'path_data')
                actuatorAmplitude = [tempData.coupon.path_data.actuator];
                sensorAmplitude = [tempData.coupon.path_data.sensor];
                amplitude_data = [tempData.coupon.path_data.amplitude]; % Assuming this field exists
                frequency_data = [tempData.coupon.path_data.frequency]; % Assuming this field exists

                avg_actuator_data = mean(actuatorAmplitude);
                avg_sensor_data = mean(sensorAmplitude);
                avg_amplitude_data = mean(amplitude_data);
                avg_frequency_data = mean(frequency_data);
                
                % Calculate time domain features
                actuatorMean = mean(actuatorAmplitude);
                actuatorStdDev = std(actuatorAmplitude);
                actuatorEnergy = sum(actuatorAmplitude.^2);

                sensorMean = mean(sensorAmplitude);
                sensorStdDev = std(sensorAmplitude);
                sensorEnergy = sum(sensorAmplitude.^2);
            end

            % Create a new row as a table
            newRow = table({load_data}, {cycles_data}, {avg_actuator_data}, {avg_sensor_data}, {avg_amplitude_data}, {avg_frequency_data}, ...
                           {actuatorMean}, {actuatorStdDev}, {actuatorEnergy}, {sensorMean}, {sensorStdDev}, {sensorEnergy}, ...
                           {avg_amplitude_data}, {avg_frequency_data}, ...
                           'VariableNames', initTable.Properties.VariableNames);

            % Append new row to the appropriate layup table
            if contains(dirr{d}, 'Layup1')
                layup1Table = [layup1Table; newRow];
            elseif contains(dirr{d}, 'Layup2')
                layup2Table = [layup2Table; newRow];
            elseif contains(dirr{d}, 'Layup3')
                layup3Table = [layup3Table; newRow];
            end
        end
    end
end

%% Create cell arrays to store results for each layup
% Remove rows containing NaN values
layup1Table = removeEmptyRows(layup1Table);
layup2Table = removeEmptyRows(layup2Table);
layup3Table = removeEmptyRows(layup3Table);

%% Save the tables
save('layup1Table.mat', 'layup1Table');
save('layup2Table.mat', 'layup2Table');
save('layup3Table.mat', 'layup3Table');




function cleanedTable = removeEmptyRows(inputTable)
    % Initialize logical vector for empty rows
    emptyRows = false(height(inputTable), 1);

    % Loop through each variable (column) in the table
    for varName = inputTable.Properties.VariableNames
        currentCol = inputTable.(varName{1});
        if iscell(currentCol)
            emptyRows = emptyRows | cellfun(@isempty, currentCol);
        elseif isnumeric(currentCol)
            emptyRows = emptyRows | isnan(currentCol);
        end
    end

    % Remove rows where any column has empty or NaN data
    cleanedTable = inputTable(~emptyRows, :);
end
