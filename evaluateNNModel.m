function [mseTrain, mseTest, isOverfitting] = evaluateNNModel(net, XTrain_norm, YTrain_norm, XTest_norm, YTest_norm, threshold)
    % Train the Neural Network
    [net, tr] = train(net, XTrain_norm', YTrain_norm');

    % Predictions on the training set
    yPredTrain = net(XTrain_norm');

    % Predictions on the test set
    yPredTest = net(XTest_norm');

    % Calculate MSE for the training set
    mseTrain = mean((yPredTrain' - YTrain_norm).^2);

    % Calculate MSE for the test set
    mseTest = mean((yPredTest' - YTest_norm).^2);

    % Display the MSE values
    fprintf('MSE on Training Set: %f\n', mseTrain);
    fprintf('MSE on Test Set: %f\n', mseTest);

    % Check for overfitting
    isOverfitting = false;
    if mseTrain < mseTest
        difference = mseTest - mseTrain;
        if difference > threshold % Threshold based on domain knowledge or empirical testing
            disp('Warning: The model might be overfitting.');
            isOverfitting = true;
        end
    else
        disp('The model seems to generalize well.');
    end
end
