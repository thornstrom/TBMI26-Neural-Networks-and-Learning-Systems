function [ cM ] = calcConfusionMatrix( Lclass, Ltrue )
classes = unique(Ltrue);
numClasses = length(classes);
cM = zeros(numClasses);

% Add 1 for each coord.
for i = 1:size(Lclass,1)
    cM(Lclass(i),Ltrue(i)) = cM(Lclass(i),Ltrue(i)) + 1;

end


% Lclass = vector [1,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,3]
% Ltrue = vector  [1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3,3]
%
%
% 0 0 0
% 0 0 0
% 0 0 0