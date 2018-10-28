function [ labelsOut ] = kNN(X, k, Xt, Lt)
%KNN Your implementation of the kNN algorithm
%   Inputs:
%               X  - Features to be classified
%               k  - Number of neighbors
%               Xt - Training features
%               LT - Correct labels of each feature vector [1 2 ...]'
%
%   Output:
%               LabelsOut = Vector with the classified labels

labelsOut  = zeros(size(X,2),1);
classes = unique(Lt);
numClasses = length(classes);

% Loop through the features which need classification
% 2 rows, 200 cols
for sample=1:size(X,2)
    %Used for storing.
    Xt1 = size(Xt);
    % Measure the euclidian distance between chosen feature to
    % every other feature in the feature space among the observed data.
    % Save the distance and label of each observed sample
    for i=1:size(Xt,2)
        dist = pdist([transpose(X(:,sample));transpose(Xt(:,i))], 'euclidean');
        label = Lt(i);
        Xt1(i, 1) = dist;
        Xt1(i, 2) = label;
    end
    % Sort the list by distance.
    Xt1 = sortrows(Xt1,1);
    % Fetch the K-nearest neighbour(s).
    knn = Xt1(1:k,:);
    % Decide the vote.
    class = -1;
    maxcount = -1;
    % Loop through each unique class.
    for j= 1:size(classes)
        count = sum(knn(:,2) == classes(j));
        % If count > maxcount we have a majority.
        if count > maxcount
            class = classes(j);
            maxcount = count;
        end
    end
    labelsOut(sample) = class;
end
       

end

