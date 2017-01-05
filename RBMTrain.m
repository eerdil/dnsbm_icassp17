% numv: number of layers in the visible unit
% numh: number of layers in the hidden unit
% alpha: learning rate
function weights = RBMTrain(data, numv, numh, alpha, maxIter)

%% initializing weights
weights = 0.1 * randn(numv, numh); % create a matrix that stores the weights between v and h

% insert weights for the bias unit
weights = [ones(1, numh); weights];
temp = [ones(1, numv + 1); weights'];
weights = temp';

%% training starts

num_examples = size(data, 2); % number of images in the training set

data = [ones(1, num_examples); data]; % add bias units of 1 into each column for each image

for iter = 1:maxIter
    pos_hidden_activations = data' * weights;
    pos_hidden_probs = 1 ./ (1 + exp(-pos_hidden_activations));
    pos_hidden_states = pos_hidden_probs > rand(num_examples, numh + 1);
    pos_associations = data * pos_hidden_probs;
    
    neg_visible_activations = pos_hidden_states * weights';
    neg_visible_probs = 1 ./ (1 + exp(-neg_visible_activations));
    neg_visible_probs(:, 1) = 1; % Fix the bias unit.
    neg_hidden_activations = neg_visible_probs * weights;
    neg_hidden_probs = 1 ./ (1 + exp(-neg_hidden_activations));
    neg_associations = neg_visible_probs' * neg_hidden_probs;
    
    weights = weights + alpha * ((pos_associations - neg_associations) / num_examples);
    if(mod(iter, 1000) == 0)
        disp(sprintf('iter: %d, err = %.3f', iter, sum(sum(data - neg_visible_probs') .^ 2)));
    end
end




end