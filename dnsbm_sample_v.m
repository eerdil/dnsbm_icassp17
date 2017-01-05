% W1 : dnsbm parameter
% states_h1: numh1 x p dimensional matrix
% states_v: numv x p dimensional matrix
function [state_v probability_v] = dnsbm_sample_v(W1, b2, states_h1)
rng('shuffle');
numv = size(W1, 1);
[numh1, p] = size(states_h1);
state_v = zeros(numv, p);

for i = 1:p
    temp = states_h1(:, i);
    state_v(:, i) = W1(:, :, i) * double(temp) + b2(:, i);
end

probability_v = 1 ./ (1 + exp(-state_v));

state_v = double(probability_v > rand(numv, p));


end