%% perfrom regression
load('layup3Table.mat')
%% Constructing the Feature Matrix and Label Matrix from combinedPZTDataTable
% Convert table to array
dataArray = table2array(layup3Table);

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
%evaluateNNModel(net, XTrain_norm, YTrain_norm, XTest_norm, YTest_norm, threshold);
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
opts.Display = 'Off';
opts.MaxFunEvals = 1;
opts.Normalize = 'on';
opts.Robust = 'Bisquare';
opts.StartPoint = [2.10585254542642 -0.911348086153246];


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
