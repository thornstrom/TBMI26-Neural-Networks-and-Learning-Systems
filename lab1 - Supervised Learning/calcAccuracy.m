function [ acc ] = calcAccuracy( cM )
%CALCACCURACY Takes a confusion matrix and calculates the accuracy

acc = (sum(diag(cM))/ sum(sum(cM)))

end

