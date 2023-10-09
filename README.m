%% CFRP Project: Predicting the S-N Curve and RUL using NASA dataset

% =====================================================================
% Overview:
% =====================================================================
% This project aims to predict the S-N (Stress vs. Number of Cycles) curve and 
% the Remaining Useful Life (RUL) for Carbon Fiber Reinforced Plastics (CFRPs) 
% using a dataset provided by NASA. Regression analysis is performed on three 
% distinct layups of the CFRPs to achieve these predictions.

% =====================================================================
% Files in this Project:
% =====================================================================

% 1. CFRP_NASA.m:
%    - Purpose: Main script responsible for data extraction.
%    - Functionality: Extracts relevant data from the individual layup files 
%      (layup 1.m, layup 2.m, layup 3.m) to be used for regression analysis.

% 2. layup 1.m, layup 2.m, layup 3.m:
%    - Purpose: Individual scripts for each layup that perform regression.
%    - Functionality: Each script will carry out regression analysis specific 
%      to its layup, determining the S-N curve and RUL.

% 3. evaluateNN_model.m:
%    - Purpose: Model evaluation script.
%    - Functionality: Evaluates the quality of the neural network model fit, 
%      offering insights into the accuracy of the model, potential overfitting 
%      scenarios, and other pertinent metrics.

% =====================================================================
% How to Use:
% =====================================================================

% 1. Data Extraction:
%    - Execute the CFRP_NASA.m script.
%    - The script will extract the necessary data from each of the layup files.

% 2. Regression Analysis:
%    - For each layup, run its corresponding script (layup 1.m, layup 2.m, 
%      layup 3.m) to perform the regression analysis and deduce the S-N curve and RUL.

% 3. Model Evaluation:
%    - After obtaining the regression results, run the evaluateNN_model.m script 
%      to evaluate the fit's quality.
%    - This script provides insights about the quality of the neural network model, 
%      emphasizing its predictive accuracy and will warn about potential overfitting situations.

% =====================================================================
% Dependencies:
% =====================================================================
% - MATLAB must be properly installed and set up on your system.
% - Ensure that all project files are available in the same directory or a specified path.

