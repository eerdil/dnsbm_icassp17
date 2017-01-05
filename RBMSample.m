% data: numv x p matrix which corresponds to states_v
% W1 and W2 are the parameters of DNSBM
% sampling_iter : number of Gibbs sampling iteration
% b1 (v to h1), b2 (h1 to v), b3 (h1 to h2), and b4(h2 to h1) are bias
% units whose dimensions are numh1 x p, numv x p, numh2 x 1, and (numh1*p)
% x 1, respectively.
% numv: number of visible units for each polytope
% numh2: number of hidden units in hidden later layer h2
% p: number of polytopes
% states_v: numv x p dimensional matrix

function states_v = RBMSample(data, W1, W2, b1, b2, b3, b4, sampling_iter)

%% initializations
% no need to initialize states_h1 since it is first computed
numh1 = size(W1, 2);
numv = size(W1, 1);
p = size(W1, 3);
numh2 = size(W2, 2);

rng('shuffle');

states_v = data;
states_h2 = double(rand(numh2, 1) > rand);

for iter = 1:sampling_iter
    [states_h1 prob_h1] = dnsbm_sample_h1(W1, W2, b1, b4, states_v, states_h2);
    [states_h2 prob_h2]= dnsbm_sample_h2(W2, b3, states_h1);
    [states_h1 prob_h1]= dnsbm_sample_h1(W1, W2, b1, b4, states_v, states_h2);
    [states_v probability_v] = dnsbm_sample_v(W1, b2, states_h1);
end

end





