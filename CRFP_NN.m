% Main MATLAB script for Predictive Maintenance using Neural Networks
% Author: Mohamed Ibrahim
% Date: 2023-09-25 
%
% This script performs the following:
% 1. Loads the data table
% 2. Prepares the training and testing sets
% 3. Normalizes the features and labels
% 4. Configures and trains a neural network
% 5. Evaluates the neural network on the test set
% 6. Plots relevant results
%% Load the data
clc; clear all; close all;


% Start a parallel pool
if isempty(gcp('nocreate'))
    parpool;
end
tic

combinedPZTDataTable = table('Size', [0, 14], 'VariableTypes', {'double', 'double',  ...
                                                               'double', 'double', 'double', 'double', ...
                                                               'double', 'double', 'double', ...
                                                               'double', 'double', 'double', ...
                                                               'double', 'double'}, ...
                             'VariableNames', {'Load', 'Cycles',  ...
                                               'Actuator_AVG', 'Sensor_AVG', 'Amplitude_AVG', 'Frequency_AVG', ...
                                               'Actuator_Mean', 'Actuator_StdDev', 'Actuator_Energy', ...
                                               'Sensor_Mean', 'Sensor_StdDev', 'Sensor_Energy', ...
                                               'Avg_Amplitude', 'Avg_Frequency'});


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
                avg_amplitude_data = mean(amplitude_data);  % New line for average amplitude
                avg_frequency_data = mean(frequency_data);  % New line for average frequency
                % Calculate time domain features
                actuatorMean = mean(actuatorAmplitude);
                actuatorStdDev = std(actuatorAmplitude);
                actuatorEnergy = sum(actuatorAmplitude.^2);

                sensorMean = mean(sensorAmplitude);
                sensorStdDev = std(sensorAmplitude);
                sensorEnergy = sum(sensorAmplitude.^2);
            end

            % Create a new row as a table
            newRow = table({load_data}, {cycles_data},  ...
                           {avg_actuator_data}, {avg_sensor_data}, {avg_amplitude_data}, {avg_frequency_data}, ...
                           {actuatorMean}, {actuatorStdDev}, {actuatorEnergy}, ...
                           {sensorMean}, {sensorStdDev}, {sensorEnergy}, ...
                           {avg_amplitude_data}, {avg_frequency_data}, ... % New variables for average amplitude and frequency
                           'VariableNames', {'Load', 'Cycles',  ...
                                             'Actuator_AVG', 'Sensor_AVG', 'Amplitude_AVG', 'Frequency_AVG', ...
                                             'Actuator_Mean', 'Actuator_StdDev', 'Actuator_Energy', ...
                                             'Sensor_Mean', 'Sensor_StdDev', 'Sensor_Energy', ...
                                             'Avg_Amplitude', 'Avg_Frequency'}); % New names for average amplitude and frequency


            % Append new row to the existing table
            combinedPZTDataTable = [combinedPZTDataTable; newRow];
        end
    end
end
%% Identify rows where any of the columns contain empty data
emptyLoadRows = cellfun(@isempty, combinedPZTDataTable.Load);
emptyCyclesRows = cellfun(@isempty, combinedPZTDataTable.Cycles);

% Identify rows where any column is empty
emptyRows = emptyLoadRows | emptyCyclesRows; 
% Remove rows where any column has empty data
combinedPZTDataTable(emptyRows,:) = [];

% Convert the Load and Cycles columns to numeric arrays (assuming they were stored as cell arrays of vectors)
load_data = cell2mat(cellfun(@(x) x(:), combinedPZTDataTable.Load, 'UniformOutput', false));
cycles_data = cell2mat(cellfun(@(x) x(:), combinedPZTDataTable.Cycles, 'UniformOutput', false));
% Remove rows where load_data or cycles_data is non-positive
validIndices = cycles_data > 0;
load_data = load_data(validIndices);
cycles_data = cycles_data(validIndices);


%% perfrom regression

%% Constructing the Feature Matrix and Label Matrix from combinedPZTDataTable
% Convert table to array
dataArray = table2array(combinedPZTDataTable);

% Define feature and label matrices
featureMatrix = dataArray(:, 3:end); % All columns from 'Actuator_AVG' onwards
labelMatrix = dataArray(:, 1:2); % The first two columns 'Load' and 'Cycles'

%% Data Partition: Training and Test Sets
% Assuming 70% of the data is used for training and 30% for testing
n = size(featureMatrix, 1); % Total number of samples
nTrain = floor(0.7 * n); % Number of training samples

% Randomly shuffle the data
rng('default'); % For reproducibility
shuffleIdx = randperm(n);

% Create training and test sets
XTrain = cell2mat(featureMatrix(shuffleIdx(1:nTrain), :));
YTrain = cell2mat(labelMatrix(shuffleIdx(1:nTrain), :));
XTest = cell2mat(featureMatrix(shuffleIdx(nTrain+1:end), :));
YTest = cell2mat(labelMatrix(shuffleIdx(nTrain+1:end), :));

% Adding a column of ones for the intercept term in the regression model
XTrain = [ones(size(XTrain, 1), 1), XTrain];
XTest = [ones(size(XTest, 1), 1), XTest];

%% Data Normalization
% Normalize Features
featureMin = min(XTrain, [], 1);
featureMax = max(XTrain, [], 1);
XTrain_norm = (XTrain - featureMin) ./ (featureMax - featureMin);
XTest_norm = (XTest - featureMin) ./ (featureMax - featureMin);

% Normalize Labels
labelMin = min(YTrain, [], 1);
labelMax = max(YTrain, [], 1);
YTrain_norm = (YTrain - labelMin) ./ (labelMax - labelMin);
YTest_norm = (YTest - labelMin) ./ (labelMax - labelMin);

%% Neural Network
% Now you can use the normalized training and test sets
% (XTrain_norm, YTrain_norm, XTest_norm, YTest_norm)
% to train, validate, and test your neural network.

%% Neural Network Configuration
hiddenLayerSize = 10;  % Number of neurons in the hidden layer
trainFcn = 'trainlm';  % Training function: Levenberg-Marquardt

% Create the neural network
net = fitnet(hiddenLayerSize, trainFcn);
% Modify training parameters
net.trainParam.epochs = 100; % Maximum number of epochs to train
net.trainParam.lr = 0.05; % Learning rate
net.trainParam.goal = 1e-5; % Performance goal
net.trainParam.min_grad = 1e-5; % Minimum performance gradient
net.trainParam.showWindow = true; % Show training GUI window
net.trainParam.showCommandLine = false; % Hide command line output
net.trainParam.time = inf; % Maximum time to train in seconds
% Define the data division ratios
trainFraction = 0.7;
valFraction = 0.15;
testFraction = 0.15;

% Update the net.divideParam structure
net.divideParam.trainRatio = trainFraction;
net.divideParam.valRatio = valFraction;
net.divideParam.testRatio = testFraction;

%% Train the Neural Network
[net, tr] = train(net, XTrain', YTrain');

%% Test the Neural Network
yPred = net(XTest');

%% Evaluate the Model
evaluateNNModel(net, XTrain_norm, YTrain_norm, XTest_norm, YTest_norm, threshold);
toc
%% Scatter Plot for YPred(:, 1) against YTest(:, 2)
yPred=yPred';
% Extracting the predicted Load from YPred
predictedLoad = yPred(:, 1);

% Extracting the actual Cycles from YTest
actualCycles = log(abs(YTest(:, 2)));

figure;

% Scatter plot
scatter(actualCycles,predictedLoad,'*');
xlabel('Predicted Cycles');
ylabel('Predicted Load');
title('Scatter Plot of Predicted Load vs Actual Cycles');
grid on;
%%
%% Calculate RUL
initialLifeCycles = 1000; % Assumption
predictedCycles = yPred(:, 2); % Extracting the predicted Cycles from yPred

% Calculate RUL
RUL = (initialLifeCycles - predictedCycles);
%% Fit a Line to the S-N Curve
% Linear fit (polynomial of degree 1)
[xData, yData] = prepareCurveData( predictedCycles, predictedLoad );

% Set up fittype and options.
ft = fittype( 'exp1' );
opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
opts.Algorithm = 'Levenberg-Marquardt';
opts.Display = 'Off';
opts.Robust = 'LAR';
opts.StartPoint = [2.92333334207048 4.34659183209305e-07];

% Fit model to data.
[fitresult, gof] = fit( xData, yData, ft, opts );

% Plot fit with data.
figure( 'Name', 'S-N Curve' ); 
h = plot( fitresult, xData, yData,'*',50 );
legend( h, 'predictedLoad vs. predictedCycles', 'S-N Curve', 'Location', 'NorthEast', 'Interpreter', 'none' );
% Label axes
xlabel( 'predictedCycles', 'Interpreter', 'none' );
ylabel( 'predictedLoad', 'Interpreter', 'none' );
grid on
%% Plot RUL
[xData, yData] = prepareCurveData( predictedCycles, RUL );

% Set up fittype and options.
ft = fittype( 'poly1' );

% Fit model to data.
[fitresult, gof] = fit( xData, yData, ft );

% Plot fit with data.
figure( 'Name', 'RUL' );
h = plot( fitresult, xData, yData,'o',100);
legend( h, 'RUL vs. predictedCycles', 'Remaining Useful Life', 'Location', 'NorthEast', 'Interpreter', 'none' );
% Label axes
xlabel( 'predictedCycles', 'Interpreter', 'none' );
ylabel( 'RUL', 'Interpreter', 'none' );
grid on