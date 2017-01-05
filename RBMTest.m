% data: data vector corresponding to visible units. Should be a column
% vector.
% weights: weight vector
% hiddenStates: a column vector representing the states of the hidden
% units
% bias: weights come from bias unit
function hiddenStates = RBMTest(newData, weights, bias)

numh = size(weights, 2);
hiddenStates = ones(1, numh);

hidden_activations = newData' * weights + bias';
hidden_probs = 1 ./ (1 + exp(-hidden_activations));
hiddenStates = hidden_probs > rand(1, numh);

hiddenStates = hiddenStates';

end