clear; close all; clc;

% Load fuzzy inference system 
fisFile = 'tank.fis';                 
fis     = readfis(fisFile);

fprintf('Loaded FIS: %s\n', fis.Name);

%% 1 | Generate random input data
levelRange = fis.Inputs(1).Range;     % e.g. [0 10]
rateRange  = fis.Inputs(2).Range;     % e.g. [-10 10]

N = 1000;                             % number of synthetic samples
rng default;                          % reproducible randomness
level = rand(N,1).*diff(levelRange) + levelRange(1);
rate  = rand(N,1).*diff(rateRange)  + rateRange(1);
inputData = [level rate];

%% 2 | Obtain target valve signal from original FIS
valveSignal = evalfis(fis, inputData);
dataset = [inputData valveSignal];

%% 3 | Train / test split
trainRatio = 0.7;                     % 70 % training, 30 % test
cv = cvpartition(N, 'HoldOut', 1 - trainRatio);
trainIdx = training(cv);
testIdx  = test(cv);

trainData = dataset(trainIdx,:);
testData  = dataset(testIdx,:);

%% 4 | Create + train the ANFIS model
% --- Use the SAME five rules from tank.fis


% Convert original Mamdani FIS to Sugeno FIS (preserves 5 rules)
fisSug = convertToSugeno(fis);
fisSug.DefuzzificationMethod = 'wtaver'; % required by ANFIS

% 2) Use this 5‑rule Sugeno system as initial structure for ANFIS.
initFIS = fisSug;                                 

% 3) Train ANFIS 
opt = anfisOptions( ...
    'InitialFIS',            initFIS, ...
    'EpochNumber',           100, ...
    'ErrorGoal',             0, ...
    'DisplayANFISInformation', true, ...
    'DisplayErrorValues',      true, ...
    'ValidationData',          testData);

[anfisFIS, trainError, ~, ~, chkError] = anfis(trainData, opt);

% Plot learning curves
figure('Name', 'ANFIS Training Curves');
plot(1:numel(trainError), trainError, '-', ...
     1:numel(chkError),  chkError,  '-.', 'LineWidth', 1.2);
xlabel('Epoch'); ylabel('RMSE'); grid on;
legend({'Training','Validation'}, 'Location', 'northEast');
title('Learning Curves – ANFIS vs Epochs');

% Surface view of the trained ANFIS only -- Neuro‑Fuzzy inference system
figure('Name','ANFIS Surface');
gensurf(anfisFIS, [1 2]); % plot membership surface for inputs 1 & 2
xlabel('Level'); ylabel('Rate'); zlabel('Valve signal');
title('Surface view of trained Neuro‑Fuzzy (ANFIS)');
view([45 35]);

%% 5 | Predict valve signal for test data
Xtest   = testData(:,1:2);
Ytarget = testData(:,3);
Yanfis  = evalfis(anfisFIS, Xtest);

%% 6 | Compute error metrics
err     = Ytarget - Yanfis;
MSE     = mean(err.^2);
RMSE    = sqrt(MSE);
meanErr = mean(err);
stdErr  = std(err);

fprintf('\n---- Test set performance -----\n');
fprintf('MSE   : %.4f\n', MSE);
fprintf('RMSE  : %.4f\n', RMSE);
fprintf('Mean  : %.4f\n', meanErr);
fprintf('Std   : %.4f\n', stdErr);

%% 7 | Plot target vs ANFIS output
figure('Name', 'Target vs ANFIS Output (Test)');
plot(Ytarget, 'ko', 'MarkerSize', 4, 'DisplayName', 'Target (FIS)');
hold on;
plot(Yanfis,  'b*', 'MarkerSize', 4, 'DisplayName', 'ANFIS Output');
legend('Location', 'best');
xlabel('Sample #'); ylabel('Valve opening [%]'); grid on;
title('Comparison on Test Dataset');

%% 8 | Extrapolation beyond universe of discourse
overScale = 0.5;
excLevel = [levelRange(1)-overScale*diff(levelRange), levelRange(2)+overScale*diff(levelRange)];
excRate  = [rateRange(1)-overScale*diff(rateRange),  rateRange(2)+overScale*diff(rateRange)];

% Define the points
oodPts = [ excLevel(2), 0;
           excLevel(1), 0;
           0,           excRate(2);
           0,           excRate(1) ];

% Temporarily disable warnings
warnState = warning('off','fuzzy:evalfis:InputOutOfRange');

% Evaluate original FIS and ANFIS for all points
nPts = size(oodPts,1);
results = zeros(nPts,4);
for i = 1:nPts
    lvl = oodPts(i,1);
    rt  = oodPts(i,2);
    results(i,1:2) = [lvl rt];
    results(i,3)   = evalfis(fis,       [lvl rt]);
    results(i,4)   = evalfis(anfisFIS, [lvl rt]);
end

% Re-enable warnings
warning(warnState);

% Table of results
fprintf('\nOutside UoD [level rate] -> Mamdani FIS | ANFIS\n');
fprintf('%8s %8s %10s %10s\n','Level','Rate','FIS','ANFIS');
for i = 1:nPts
    fprintf('%8.2f %8.2f %10.4f %10.4f\n', results(i,:));
end



