%% This script will help you test out your kNN code

%% Select which data to use:

% 1 = dot cloud 1
% 2 = dot cloud 2
% 3 = dot cloud 3
% 4 = OCR data

dataSetNr = 3; % Change this to load new data 

[X, D, L] = loadDataSet( dataSetNr );

% You can plot and study dataset 1 to 3 by running:
% plotCase(X,D)

%% Select a subset of the training features

numBins = 3; % Number of Bins you want to devide your data into
numSamplesPerLabelPerBin = 100; % Number of samples per label per bin, set to inf for max number (total number is numLabels*numSamplesPerBin)
selectAtRandom = true; % true = select features at random, false = select the first features

[ Xt, Dt, Lt ] = selectTrainingSamples(X, D, L, numSamplesPerLabelPerBin, numBins, selectAtRandom );

% Note: Xt, Dt, Lt will be cell arrays, to extract a bin from them use i.e.
% XBin1 = Xt{1};

%% Use c.v-kNN to classify data


%Find optimal k using n-fold cross validation

accValues = zeros(1,6);

for k = 1:6
    sumAcc = 0;
    for i=1:numBins
        xtest = Xt{i};
        xtrain = Xt;
        % Remove cell corresponding to testdata.
        xtrain(i) = [];
        % Merge the remaining cells
        xtrain = cat(2, xtrain{:});
        
        ltest = Lt{i};
        ltrain = Lt;
        % Remove cell corresponding to testdata.
        ltrain(i) = [];
        % Merge remaining cells
        ltrain = cat(2, ltrain{:});
        
        %Run kNN.
        LkNN = kNN(xtest, k, xtrain, ltrain);
        
        %Get accuracy for this k.
        cM = calcConfusionMatrix( LkNN, ltest);
        acc = calcAccuracy(cM);
        sumAcc = sumAcc + acc;
          
    end
    accValues(k) = (sumAcc/numBins);
end

[M, I] = max(accValues);
%Display best accuray and for which K.
M
I



%% Plot classifications
% Note: You do not need to change this code.
if dataSetNr < 4
    plotkNNResultDots(Xt{2},LkNN,k,Lt{2},Xt{1},Lt{1});
else
    plotResultsOCR( Xt{2}, Lt{2}, LkNN )
end
