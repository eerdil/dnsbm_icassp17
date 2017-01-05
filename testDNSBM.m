% Testing DNSBM
% 
% Ertunc Erdil, 01.01.2017

clear all
close all
clc

directoryName = 'walking_silhouettes'; % or standing_person
load(sprintf('%s/parameters', directoryName));
numberOfSamples = 100; % number of samples to be generated
sampling_iter = 20; % Gibbs sampling iterations
outputFolder = 'test1'; % test image is read from and output is written to this folder.
numberOfPolytopes = 6;

initial_states = zeros(sz_x * sz_y, numberOfPolytopes);
numberOfTrainingImages = length(imageIds);
missingRegion = double(imread(sprintf('%s/%s/missingRegion.png', directoryName, outputFolder)) > 0); % binary image that indicate missing region.

missingPolytopeIds = [];
completelyMissingPolytopeIds = [];
inputImage = 0;

for i = 1:numberOfPolytopes
    temp = double(imread(sprintf('%s/%s/%s_%d.png', directoryName, outputFolder, outputFolder, i)) > 0); % read corresponding polytope of the test image
    inputImage = inputImage + double(temp) .* double(missingRegion); % construct whole test image with known parts
    initial_states(:, i) = double(temp(:)) .* double(missingRegion(:)); % each polytope is initialized with its current data before sampling from DNSBM.
    
    % find polytopes ids that includes missing region. samples are
    % generated from that polytopes.
    temp_initial_states = double(temp(:)) .* double(missingRegion(:));
    if(sum(temp_initial_states) == 0)
        completelyMissingPolytopeIds = [completelyMissingPolytopeIds, i];
    elseif(sum(temp_initial_states) < sum(temp(:))) % finds partially missing polytopes
        missingPolytopeIds = [missingPolytopeIds, i]; 
    end

end
inputImage = double(inputImage > 0);

rng('shuffle');
for i = 1:numberOfSamples
    
    disp(sprintf('sample %d', i));
    allMissingPolytopes = [completelyMissingPolytopeIds, missingPolytopeIds];
    
    sample = 0; % as we sampled from each polytope, add to variable sample to form the final sample.
    for j = 1:numberOfPolytopes
        if(~isempty(find(allMissingPolytopes == j)))
            states_v = RBMSample(initial_states, W1, W2, b1, b2, b3, b4, sampling_iter);
            temp = states_v(:, j);
            temp = reshape(temp, [sz_x sz_y]);
            sample = sample + temp;
        else
            temp = initial_states(:, j);
            temp = reshape(temp, [sz_x sz_y]);
            sample = sample + temp;
        end
    end
    inputImage(find(missingRegion == 0)) = sample(find(missingRegion == 0));
    imwrite(inputImage > 0, sprintf('%s/%s/sample_%d.png', directoryName, outputFolder, i));
end




