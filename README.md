# ML_Project


# Predictive Maintenance using Neural Networks

## Project Overview
This project aims to predict the Remaining Useful Life (RUL) and potential failure of machinery using data obtained from sensors. The data collected include features like 'Load', 'Cycles', and various other sensor readings. The project uses MATLAB and specifically focuses on neural networks to build the predictive model.

## Objectives
- To perform exploratory data analysis on machinery sensor data
- To build and train a neural network model for regression
- To evaluate the model on test data
- To predict the Remaining Useful Life (RUL) of the machinery
## Files in this Project:

### 1. **CFRP_Data.m**:  
   - **Purpose**: Main script responsible for data extraction.
   - **Functionality**: Extracts relevant data from the individual layup files (`layup 1.mat`, `layup 2.mat`, `layup 3.mat`) to be used for regression analysis.

### 2. **layup 1.m**, **layup 2.m**, **layup 3.m**:
   - **Purpose**: Individual scripts for each layup that perform regression.
   - **Functionality**: Each script will carry out regression analysis specific to its layup, determining the S-N curve and RUL.

### 3. **evaluateNN_model.m**:
   - **Purpose**: Model evaluation script.
   - **Functionality**: Evaluates the quality of the neural network model fit, offering insights into the accuracy of the model, potential overfitting scenarios, and other pertinent metrics.

## How to Use:

1. **Data Extraction**:
   - Execute the `CFRP_NASA.m` script.
   - The script will extract the necessary data from each of the layup files.

2. **Regression Analysis**:
   - For each layup, run its corresponding script (`layup 1.m`, `layup 2.m`, `layup 3.m`) to perform the regression analysis and deduce the S-N curve and RUL.

3. **Model Evaluation**:
   - After obtaining the regression results, run the `evaluateNN_model.m` script to evaluate the fit's quality.
   - This script provides insights about the quality of the neural network model, emphasizing its predictive accuracy and will warn about potential overfitting situations.

## Dependencies
- MATLAB 2023
- Neural Network Toolbox

## Setup and Usage

### Data Preparation
Load the provided `.mat` file containing the data table for sensor readings, load, and cycles.

### Running the Code
Run the script `main_script.m` to perform the following tasks:

1. Data partition into training and testing sets
2. Feature scaling and normalization
3. Neural network configuration and training
4. Model evaluation using MSE
5. Generate plots for result visualization

```matlab
% Navigate to the project folder and run
main_script
```

### Interpret Results
After the model is trained, you'll see several figures showing:

- Scatter Plot of Predicted Load vs Actual Cycles
- S-N Curve showing predicted load against predicted cycles
- Remaining Useful Life (RUL) vs Predicted Cycles

## Files and Folders

- `main_script.m`: Main MATLAB script for running the project
- `layup1Table.mat`: MATLAB data file containing sensor readings, load, and cycles
- `/plots`: Folder containing saved plots (if applicable)

## Evaluation Metrics
The model is evaluated based on Mean Squared Error (MSE).

## Authors
- Mohamed Ibrahim

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.

