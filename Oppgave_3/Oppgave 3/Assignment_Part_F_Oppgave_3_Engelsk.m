clear; close all; clc;
%% Part 1a: Read data
% Filename
filename = 'airfoil_self_noise.dat';
% Read data from the .dat file into a matrix
airfoilData = readmatrix(filename);
% Check the size of the loaded data
[numRows, numCols] = size(airfoilData);
fprintf('Loaded data with %d rows (instances) and %d columns.\n', numRows, numCols);
% Separate input (columns 1-5) and output (column 6)
inputs = airfoilData(:, 1:5);
output = airfoilData(:, 6);
fprintf('Separated data into %d inputs and 1 output.\n', size(inputs, 2));
% First rows
disp('First 5 rows of input data:');
disp(inputs(1:5, :));
disp('First 5 rows of output data:');
disp(output(1:5));
%% Part 1b: Split Dataset
% Combine inputs and output for splitting
airfoilAllData = [inputs, output];
% Set split ratio
trainRatio = 0.70;
% Total number of data points
N = size(airfoilAllData, 1);
% Number of training points
numTrain = round(trainRatio * N);
% same split every time
rng(0);
idxPermuted = randperm(N);
% Select indices for training and testing
trainInd = idxPermuted(1:numTrain);
testInd = idxPermuted(numTrain+1:end);
% Create training and test sets based on the indices
% Columns 1-5: inputs, Column 6: output
trainData_Ex3 = airfoilAllData(trainInd, :);
testData_Ex3 = airfoilAllData(testInd, :);
% Show the size of the sets for checking
fprintf('Total data (Airfoil): %d rows\n', N);
fprintf('Training data (Airfoil): %d rows (%.0f%%)\n', size(trainData_Ex3, 1), trainRatio*100);
fprintf('Test data (Airfoil): %d rows (%.0f%%)\n', size(testData_Ex3, 1), (1-trainRatio)*100);
%% Part 1c: Generate Initial FIS
fprintf('Generating initial FIS with Subtractive Clustering...\n');
% Method and influence radius
clusterInfluence = 0.45;
genOpt_Ex3 = genfisOptions('SubtractiveClustering',...
                          'ClusterInfluenceRange', clusterInfluence);
% Generate initial FIS based on training data
try
    initialFIS_Ex3 = genfis(trainData_Ex3(:, 1:end-1), trainData_Ex3(:, end), genOpt_Ex3);
catch ME
    error('Error during FIS generation with genfis: %s\nPlease consider adjusting ClusterInfluenceRange.', ME.message);
end
% Show the number of rules generated
numRulesGenerated = numel(initialFIS_Ex3.Rules);
fprintf('Initial FIS generated with %d rules (Cluster Influence = %.2f).\n', ...
        numRulesGenerated, clusterInfluence);
% Plot structure
figure; plotfis(initialFIS_Ex3);
figure; plotmf(initialFIS_Ex3, 'input', 1); % View MFs for input 1
%% Part 2: Adjust (Tune) FIS parameters
testInput_Ex3 = testData_Ex3(:, 1:5); % Input columns (1-5)
actualTestOutput_Ex3 = testData_Ex3(:, 6); % Actual output (column 6)
%% Part 2.1: Tuning with ANFIS
fprintf('\n--- Starting Tuning with ANFIS ---\n');
numEpochs_tuning = 100; % Number of epochs for tuning
anfisOpt = anfisOptions('InitialFIS', initialFIS_Ex3, 'EpochNumber', numEpochs_tuning);
fprintf('Tuning FIS with ANFIS (%d epochs)...\n', numEpochs_tuning);
[tunedFIS_anfis, trainError_anfis] = anfis(trainData_Ex3, anfisOpt);
fprintf('Tuning with ANFIS completed. Final training RMSE: %.6f\n', trainError_anfis(end));
% Evaluate the ANFIS-tuned FIS on test data
predictedTest_anfis = evalfis(tunedFIS_anfis, testInput_Ex3);
% Calculate RMSE
rmse_test_anfis = sqrt(mean((actualTestOutput_Ex3 - predictedTest_anfis).^2));
fprintf('RMSE on test set after ANFIS tuning: %.6f\n', rmse_test_anfis);
% Plot
figure;
scatter(actualTestOutput_Ex3, predictedTest_anfis, 'b.');
hold on;
mn_anfis = min([actualTestOutput_Ex3; predictedTest_anfis]);
mx_anfis = max([actualTestOutput_Ex3; predictedTest_anfis]);
plot([mn_anfis mx_anfis], [mn_anfis mx_anfis], 'r--');
hold off;
xlabel('Actual Output'); ylabel('Predicted Output (ANFIS Tuned)');
title('Test Data: Actual vs. ANFIS-Tuned Predicted');
legend('Predictions', 'Ideal (y=x)', 'Location', 'best');
grid on; axis equal;
%% Part 2.2: Tuning with Genetic Algorithm (GA)
fprintf('\n--- Starting Tuning with Genetic Algorithm (GA) ---\n');
% Get the tunable parameter settings for Input MFs
fprintf('Getting tunable settings for initialFIS_Ex3...\n');
[in_t, ~, ~] = getTunableSettings(initialFIS_Ex3);
% Specify tuning of ONLY input membership functions for GA.
paramSet_ga = in_t;
fprintf('Specifying tuning of ONLY input membership functions for GA.\n');
% Set tuning options for the GA method
gaTuneOpt = tunefisOptions('Method', 'ga');
gaTuneOpt.MethodOptions.PopulationSize = 100;
gaTuneOpt.MethodOptions.MaxGenerations = 50;
% Tune the initial FIS with the GA method and training data
fprintf('Tuning FIS with GA (Pop=%d, Gen=%d)...\n', ...
        gaTuneOpt.MethodOptions.PopulationSize, gaTuneOpt.MethodOptions.MaxGenerations);
try

    [tunedFIS_ga, tuningInfo_ga] = tunefis(initialFIS_Ex3, ...   % Initial FIS
                                             paramSet_ga, ...
                                             trainData_Ex3(:, 1:5), ...
                                             trainData_Ex3(:, 6), ...
                                             gaTuneOpt);           % GA options
    % Getting best RMSE
    fprintf('Tuning with GA completed. Best training RMSE (fval): %.6f\n', tuningInfo_ga.tuningOutputs.fval);
    % Evaluate the GA-tuned FIS on test data
    predictedTest_ga = evalfis(tunedFIS_ga, testInput_Ex3);
    % Calculate RMSE for the test set for the GA-tuned model
    rmse_test_ga = sqrt(mean((actualTestOutput_Ex3 - predictedTest_ga).^2));
    fprintf('RMSE on test set after GA tuning: %.6f\n', rmse_test_ga);
    % Plot
    figure;
    scatter(actualTestOutput_Ex3, predictedTest_ga, 'g.'); % Green dots
    hold on;
    mn_ga = min([actualTestOutput_Ex3; predictedTest_ga]);
    mx_ga = max([actualTestOutput_Ex3; predictedTest_ga]);
    plot([mn_ga mx_ga], [mn_ga mx_ga], 'r--');
    hold off;
    xlabel('Actual Output'); ylabel('Predicted Output (GA Tuned)');
    title('Test Data: Actual vs. GA-Tuned Predicted');
    legend('Predictions', 'Ideal (y=x)', 'Location', 'best');
    grid on; axis equal;
catch ME_ga
    warning('Could not complete GA tuning. Error: %s', ME_ga.message);
    % Setting dummy values if GA failed to allow saving
    tunedFIS_ga = initialFIS_Ex3;
    rmse_test_ga = NaN;
end
%% Part 2.3: Tuning with Particle Swarm Optimization (PSO)
fprintf('\n--- Starting Tuning with Particle Swarm Optimization (PSO) ---\n');
% Get the tunable parameter settings for Input MFs again
fprintf('Getting tunable settings for initialFIS_Ex3...\n');
[in_t, ~, ~] = getTunableSettings(initialFIS_Ex3);
paramSet_pso = in_t; % Specifying tuning of ONLY input MFs
fprintf('Specifying tuning of ONLY input membership functions for PSO.\n');
% Set tuning options for the PSO method
psoTuneOpt = tunefisOptions('Method', 'particleswarm');
psoTuneOpt.MethodOptions.SwarmSize = 100; % Number of particles
psoTuneOpt.MethodOptions.MaxIterations = 50; % Number of iterations
% Tune the initial FIS with the PSO method and training data
fprintf('Tuning FIS with PSO (Swarm=%d, Iter=%d)...\n', ...
        psoTuneOpt.MethodOptions.SwarmSize, psoTuneOpt.MethodOptions.MaxIterations);
try
    % Requesting 2 outputs from tunefis
    [tunedFIS_pso, tuningInfo_pso] = tunefis(initialFIS_Ex3, ...   % Initial FIS
                                             paramSet_pso, ...
                                             trainData_Ex3(:, 1:5), ...
                                             trainData_Ex3(:, 6), ...
                                             psoTuneOpt);          % PSO options
    % Getting best RMSE
    fprintf('Tuning with PSO completed. Best training RMSE (fval): %.6f\n', tuningInfo_pso.tuningOutputs.fval);
    % Evaluate the PSO-tuned FIS on test data
    predictedTest_pso = evalfis(tunedFIS_pso, testInput_Ex3);
    % Calculate RMSE for the test set for the PSO-tuned model
    rmse_test_pso = sqrt(mean((actualTestOutput_Ex3 - predictedTest_pso).^2));
    fprintf('RMSE on test set after PSO tuning: %.6f\n', rmse_test_pso);
    % Plot
    figure;
    scatter(actualTestOutput_Ex3, predictedTest_pso, 'm.'); % Magenta dots
    hold on;
    mn_pso = min([actualTestOutput_Ex3; predictedTest_pso]);
    mx_pso = max([actualTestOutput_Ex3; predictedTest_pso]);
    plot([mn_pso mx_pso], [mn_pso mx_pso], 'r--');
    hold off;
    xlabel('Actual Output'); ylabel('Predicted Output (PSO Tuned)');
    title('Test Data: Actual vs. PSO-Tuned Predicted');
    legend('Predictions', 'Ideal (y=x)', 'Location', 'best');
    grid on; axis equal;
catch ME_pso
    warning('Could not complete PSO tuning. Error: %s', ME_pso.message);
    % Setting dummy values if PSO failed to ensure saving
    tunedFIS_pso = initialFIS_Ex3;
    rmse_test_pso = NaN;
end
%% Summary of Test RMSE for Exercise 3
fprintf('\n=== Summary Test RMSE (Exercise 3) ===\n');
fprintf('ANFIS: %.6f\n', rmse_test_anfis);
fprintf('GA   : %.6f\n', rmse_test_ga);
fprintf('PSO  : %.6f\n', rmse_test_pso);