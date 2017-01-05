% W2 : dnsbm parameter
% states_h1: numh1 x p dimensional matrix
% states_h2: numh2 x 1 dimensional matrix indicating hidden state h2
function [state_h1 probability_h1] = dnsbm_sample_h2(W2, b3, states_h1)
rng('shuffle');
numh2 = size(W2, 2);

temp = states_h1(:);
dummy = temp' * W2;
dummy = dummy' + b3;

probability_h1 = 1 ./ (1 + exp(-dummy)); % sigmoid function

state_h1 = double(probability_h1 > rand(numh2, 1));

end