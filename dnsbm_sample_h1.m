% W1 and W2 are the parameters of DNSBM
% states_v: numv x p dimensional matrix indicating visible state
% states_h2: numh2 x 1 dimensional matrix indicating hidden state h2
% states_h1: numh1 x p dimensional matrix
function [state_h1, probability_h1] = dnsbm_sample_h1(W1, W2, b1, b4, states_v, states_h2)
rng('shuffle');
numh1 = size(W1, 2);
p = size(W1, 3);

% computation of h1 conditioned on v
temp_state_h1 = zeros(numh1, p); 
for i = 1:p
    dummy_sv = states_v(:, i);
    dummy = dummy_sv' * W1(:, :, i);
    temp_state_h1(:, i) = dummy' + b1(:, i); % add bias unit
end

% computation of h1 conditioned on h2
dummy = W2 * states_h2 + b4;
dummy = reshape(dummy, [numh1 p]);

state_h1 = temp_state_h1 + dummy;

probability_h1 = 1 ./ (1 + exp(-state_h1)); % sigmoid function

state_h1 = double(probability_h1 > rand(numh1, p));

end