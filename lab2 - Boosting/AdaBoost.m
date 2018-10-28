% Load face and non-face data and plot a few examples
load faces;
load nonfaces;
faces = double(faces);
nonfaces = double(nonfaces);

% Generate Haar feature masks
nbrHaarFeatures = 200;
haarFeatureMasks = GenerateHaarFeatureMasks(nbrHaarFeatures);


%{
figure(1);
colormap gray;
for c=1:25
    subplot(5,5,c), imagesc(faces(:,:,10*c));
    axis image;
    axis off;
end

figure(2);
colormap gray;
for c=1:25
    subplot(5,5,c), imagesc(nonfaces(:,:,10*c));
    axis image;
    axis off;
end

figure(3);
colormap gray;
for c = 1:25
    subplot(5,5,c),imagesc(haarFeatureMasks(:,:,c),[-1 2]);
    axis image;
    axis off;
end
%}


% Create a training data set with a number of training data examples
% from each class. Non-faces = class label y=-1, faces = class label y=1
nbrTrainExamples = 300;
trainImages = cat(3,faces(:,:,1:nbrTrainExamples),nonfaces(:,:,1:nbrTrainExamples));
xTrain = ExtractHaarFeatures(trainImages,haarFeatureMasks);
yTrain = [ones(1,nbrTrainExamples), -ones(1,nbrTrainExamples)];


%% Implement the AdaBoost training here
%  Use your implementation of WeakClassifier and WeakClassifierError

tic
r = size(xTrain, 2);
d = ones(1, r) / r;
nClf = 60; % Number of weak classifiers.
clfs = ones(4, nClf); %For best model variables.

for clfRange = 1:nClf
    alpha = 0;
    errMin = 1;
    polarity = 0;
    threshold = 0;
    featureIndex = 0;
    h_min = 0;
    for haarFt = 1:nbrHaarFeatures
        for trainEx = 1:nbrTrainExamples*2
            tao = xTrain(haarFt, trainEx);
            p = 1;            
            %Classify image.
            h = WeakClassifier(tao, p, xTrain(haarFt,:));           
            %Classification error.
            err = WeakClassifierError(h, d, yTrain);
            
            if err > 0.5
                p = -1;
                err = 1 - err;
            end
            
            % Optimal clasifier.
            if err < errMin
                polarity = p;
                threshold = tao;
                featureIndex = haarFt;
                errMin = err;
                h_min = p*h;
                % Alpha formula -> lecture slides.
                alpha = 0.5*log((1 - errMin) / (errMin));
            end
        end
    end
    
    clfs(1:4, clfRange) = [threshold, polarity, featureIndex, alpha];
    % Update the weights
    d = d.*exp(-alpha * yTrain .* h_min);
    % Check to avoid outlier problem.
    d(d>0.5) = 0.5;
    d = d ./ sum(d);
end
runningTime = toc;

%% Extract test data

nbrTestExamples = 3000;

testImages  = cat(3,faces(:,:,(nbrTrainExamples+1):(nbrTrainExamples+nbrTestExamples)),...
                    nonfaces(:,:,(nbrTrainExamples+1):(nbrTrainExamples+nbrTestExamples)));
xTest = ExtractHaarFeatures(testImages,haarFeatureMasks);
yTest = [ones(1,nbrTestExamples), -ones(1,nbrTestExamples)];

%% Evaluate your strong classifier here
%  You can evaluate on the training data if you want, but you CANNOT use
%  this as a performance metric since it is biased. You MUST use the test
%  data to truly evaluate the strong classifier.

singleStrongTrainClassifier = zeros(nClf, nbrTrainExamples*2);
finalStrongTrainClassifier = zeros(1, nbrTrainExamples*2);
trainAccuracy = zeros(1,nClf);

singleStrongTestClassifier = zeros(nClf, nbrTestExamples*2);
finalStrongTestClassifier = zeros(1, nbrTestExamples*2);
testAccuracy = zeros(1,nClf);

weakClassification = zeros(nClf, nbrTestExamples*2);

for clf = 1:nClf % test using different amount of classifiers
    for c = 1:clf % for each classifier
        % use only best feature for classification
        threshold = clfs(1, c);
        polarity = clfs(2, c);
        featureIndex = clfs(3, c);
        alpha = clfs(4, c);

        x = xTest(featureIndex,:);
        x2 = xTrain(featureIndex,:);
        
        %Strong classifier formula. Sum next for loop.
        singleStrongTestClassifier(c,:) = alpha*WeakClassifier(threshold, polarity, x);
        singleStrongTrainClassifier(c,:) = alpha*WeakClassifier(threshold, polarity, x2);
        %Used to observe missclassified images. (discrete output)
        weakClassification(c,:) = WeakClassifier(threshold, polarity, x);
    end
    for i = 1:nbrTestExamples*2 
        finalStrongTestClassifier(i) = sign(sum(singleStrongTestClassifier(:,i)));
    end
    for i = 1:nbrTrainExamples*2
        finalStrongTrainClassifier(i) = sign(sum(singleStrongTrainClassifier(:,i)));
    end
    
    testMissClassification = (finalStrongTestClassifier ~= yTest);
    testAccuracy(clf) = 1- sum(testMissClassification/(nbrTestExamples*2));
    
    trainMissClassification = (finalStrongTrainClassifier ~= yTrain);
    trainAccuracy(clf) = 1 - sum(trainMissClassification/(nbrTrainExamples*2));     
end

% Fetch optimal accuracy and for which amount of weak classifiers.
[optTest,optTestClassifier] = max(testAccuracy);
[optTrain,optTrainClassifier] = max(trainAccuracy);

% Print optimal amount of Weak Classifiers for test data.
optTest
optTestClassifier

missclassified = zeros(nClf,nbrTestExamples*2);
for i = 1:nClf
   missclassified(i,:) = (weakClassification(i,:) ~= yTest); 
end

% Descend order to fetch 
[missClassifications, missclassifiedIndex] = sort(sum(missclassified),'descend'); 


%% Plot the error of the strong classifier as function of the number of weak classifiers.
%  Note: you can find this error without re-training with a different
%  number of weak classifiers.

% Plot of accuracies
clfRange = 1:nClf;
figure(4);
plot(clfRange, testAccuracy);
hold on
plot(clfRange, trainAccuracy);
title('Training & test accuracy with various number of weak classifiers.')
xlabel('Number of weak classifiers')
ylabel('Accuracy')
str = {'Number of training data (faces + nonfaces):';
    nbrTrainExamples;
    'Number of Haar-Features:';
    nbrHaarFeatures};
dim = [.5 .5 .3 .3];
annotation('textbox',dim,'String',str,'FitBoxToText','on');
legend('Test Accuracy', 'Training Accuracy')


%Plots of missclassified images.
figure(5)
colormap gray
counter = 1;
doubleCheck = [];
for i = 1:nbrTestExamples*2
    misclassifiedImg = missclassifiedIndex(i);
    if(misclassifiedImg < nbrTestExamples && ~ismember(misclassifiedImg, doubleCheck(:)))
        face = faces(:,:,misclassifiedImg);
        subplot(5,5,counter), imagesc(face), axis image, axis off
        counter = counter + 1;
        doubleCheck = [doubleCheck, misclassifiedImg];
        if(counter == 26)
            break;
        end 
    end
end

figure(6)
colormap gray
counter2 = 1;
doubleCheck2 = [];
for i = 1:nbrTestExamples*2
    misclassifiedImg = missclassifiedIndex(i);
    if(misclassifiedImg > nbrTestExamples && ~ismember(misclassifiedImg, doubleCheck2(:)))    
        nonface = nonfaces(:,:,misclassifiedImg);
        subplot(5,5,counter2), imagesc(nonface), axis image, axis off
        counter2 = counter2 + 1;
        doubleCheck2 = [doubleCheck2, misclassifiedImg];
        if(counter2 == 26)
            break;
        end 
    end
end