% Training DNSBM
% 
% Ertunc Erdil, 01.01.2017

clear all
close all
clc

imageIds = 1:29;% Traning image ids: 1:29 for standing_person data set, 1:24 for walking_silhouettes data set;
numberOfImages = length(imageIds);
p = 6; % number of polytopes in each image
numh1 = 2000; % number of units in the first hidden unit that corresponds to a single polytope
numh2 = 500; % number of units in the second hidden unit
directoryName = 'standing_person';
outputFolder = 'samples';

%% to learn image sizes
temp = double(imread(sprintf('%s/image%d_%d.png', directoryName, 1, 1)) > 0);
[sz_x, sz_y] = size(temp);

%% read training images
images = zeros(sz_x * sz_y, p, numberOfImages);
counter = 0;
for i = imageIds
    counter = counter + 1;
    for j = 1:p
        temp = double(imread(sprintf('%s/image%d_%d.png', directoryName, i, j)) > 0);
        images(:, j, counter) = temp(:);
    end
end

%% training v-h1 for each polytope
alpha = 0.01; % learning rate
numv = sz_x * sz_y; % number of input units for each polytope
maxIter = 7000; % maximum number of iterations for RBMTrain
W1 = zeros((numv + 1), (numh1 + 1), p); % model parameters. +1 is for the bias unit
for i = 1:p
    data = squeeze(images(:, i, :)); % each column of data is a polytope from different training shapes
    W1(:, :, i) = RBMTrain(data, numv, numh1, alpha, maxIter); % train RBMs betwewn v-h1 for each polytope 
end

% exclude bias weights from W1
b1 = zeros(numh1, p); % bias weights from v to h1 for each polytope
b2 = zeros(numv, p); % bias weights from h1 to v for each polytope

b1 = squeeze(W1(1, 2:end, :));
b2 = squeeze(W1(2:end, 1, :));

W1 = W1(2:end, 2:end, :);

%% training h1-h2
W2 = zeros(numh1 * p + 1, numh2 + 1); % model parameters. +1 is for the bias unit
datah1 = zeros(numh1 * p, numberOfImages); % outputs in h1

for i = 1:numberOfImages % find outputs in h1 based on the trained RBMs above
    temp = [];
    for j = 1:p
        data = images(:, j, i);
        dummy = RBMTest(data, W1(:, :, j), b1(:, j));
        temp = [temp; dummy];
    end
    datah1(:, i) = temp;
end

maxIter = 50000; % maximum number of iterations for RBMTrain
alpha = 0.01; % learning rate
W2 = RBMTrain(datah1, numh1*p, numh2, alpha, maxIter); % find the model parameters between h1-h2

b3 = zeros(numh2, 1); % bias weights from h1 to h2
b4 = zeros(numh1 * p, 1); % bias weights from h2 to h1

b3 = W2(1, 2:end)';
b4 = W2(2:end, 1);

W2 = W2(2:end, 2:end);

% Save model parameters
save(sprintf('%s/parameters.mat', directoryName), 'W1', 'W2', 'b1', 'b2', 'b3', 'b4', 'sz_x', 'sz_y', 'outputFolder', 'imageIds', '-v7.3');
disp(sprintf('Parameters have been saved to the folder %s', directoryName));

