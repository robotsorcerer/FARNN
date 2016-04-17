% backprop a per-epoch backpropagation training for a multilayer feedforward
%          neural network.
%   Network = bbackprop(Layers,N,M,SatisfactoryMSE,Input,Desired) 
%   returns Network, a structure having the fields:
%     structure - vector of number of nodes at each layer
%     weights - a cell array specifiying the final weight matrices computed
%     epochs - the epochs required for training
%     mse - the mean squared error at termination
%   weights is a cell array specifying the final weight matrices computed by 
%   minimizing the mean squared error between the Desired output and the 
%   actual output of the network given a set of training samples: Input and 
%   the SatisfactoryMSE (satisfactory mean squared error) 
%
%   Input:
%    Layers - a vector of integers specifying the number of nodes at each
%     layer, i.e for all i, Layers(i) = number of nodes at layer i, there
%     must be at least three layers and the input layer Layers(1) must
%     equal the dimension of each vector in Input, likewise, Layers(end) 
%     must be equal to the dimension of each vector in Desired
%     N - training rate for network learning [0.1 - 0.9]
%     M - momentum for the weight update rule [0.0 - 0.9]
%     SatisfactoryMSE - the mse at which to terminate computation
%     Input - the training samples, a P-by-N matrix, where each Input[p] is
%      a training vector
%     Desired - the desired outputs, a P-by-M matrix where each Desired[p]
%      is the desired output for the corresponding input Input[p]
%
%   This algorithm uses the hyperbolic tangent node function 
%   2/(1+e^(-net)) - 1, suitable for use with bipolar data
%   
%   NOTE: due to its generality this algorithm is not as efficient as a 
%   one designed for a specific problem. If the number of desired layers is 
%   known ahead of time, it is better to a) remove the use of cells for the
%   weight matrices and explicitly create the number of desired weight matrices
%   and 'unfold' the loops inside the presentation loop. That is, explicitly 
%   calculate the input and output of each each layer one by one and subsequently 
%   the modified error and weight matrix modifications b) remove calculation of 
%   momentum if it is not required
%
% Author: Dale Patterson
% $Version: 3.1.2 $ $Date: 2.25.06 $
% 
function Network = bbackprop(L,n,m,smse,X,D)
%%%%% VERIFICATION PHASE %%%%%
% determine number of input samples, desired output and their dimensions
[P,N] = size(X);
[Pd,M] = size(D);

% make user that each input vector has a corresponding desired output
if P ~= Pd 
    error('backprop:invalidTrainingAndDesired', ...
          'The number of input vectors and desired ouput do not match');
end

% make sure that at least 3 layers have been specified and that the 
% the dimensions of the specified input layer and output layer are
% equivalent to the dimensions of the input vectors and desired output
if length(L) < 3 
    error('backprop:invalidNetworkStructure','The network must have at least 3 layers');
else
    if N ~= L(1) || M ~= L(end)
        e = sprintf('Dimensions of input (%d) does not match input layer (%d)',N,L(1));
        error('backprop:invalidLayerSize', e);
    elseif M ~= L(end)
        e = sprintf('Dimensions of output (%d) does not match output layer (%d)',M,L(end));
        error('backprop:invalidLayerSize', e);    
    end
end

%%%%% INITIALIZATION PHASE %%%%%
nLayers = length(L); % we'll use the number of layers often  

% randomize the weight matrices (uniform random values in [-1 1], there
% is a weight matrix between each layer of nodes. Each layer (exclusive the 
% output layer) has a bias node whose activation is always 1, that is, the 
% node function is C(net) = 1. Furthermore, there is a link from each node
% in layer i to the bias node in layer j (the last row of each matrix)
% because it is less computationally expensive then the alternative. The 
% weights of all links to bias nodes are irrelevant and are defined as 0
w = cell(nLayers-1,1); % a weight matrix between each layer
for i=1:nLayers-2        
    w{i} = [1 - 2.*rand(L(i+1),L(i)+1) ; zeros(1,L(i)+1)];
end
w{end} = 1 - 2.*rand(L(end),L(end-1)+1);

% initialize stopping conditions
mse = Inf;  % assuming the intial weight matrices are bad
epochs = 0;

%%%%% PREALLOCATION PHASE %%%%%
% for faster computation preallocate activation,net,prev_w and sum_w

% Activation: there is an activation matrix a{i} for each layer in the 
% network such that a{1} = the network input and a{end} = network output
% Since we're doing batch mode, each activation matrix a{i} is a 
% P-by-K (P=num of samples,K=nodes at layer i) matrix such that 
% a{i}(j) denotes the activation vector of layer i for the jth input and 
% a{i}(j,k) is the activation(output) of the kth node in layer i for the jth 
% input
a = cell(nLayers,1);  % one activation matrix for each layer
a{1} = [X ones(P,1)]; % a{1} is the input + '1' for the bias node activation
                      % a{1} remains the same throught the computation
for i=2:nLayers-1
    a{i} = ones(P,L(i)+1); % inner layers include a bias node (P-by-Nodes+1) 
end
a{end} = ones(P,L(end));   % no bias node at output layer

% Net: like activation, there is a net matrix net{i} for each layer
% exclusive the input such that net{i} = sum(w(i,j) * a(j)) for j = i-1
% and each net matrix net{i} is a P-by-K matrix such that net{i}(j) denotes 
% the net vector at layer i for the jth sample and net{i}(j,k) denotes the
% net input at node k of the ith layer for the jth sample
net = cell(nLayers-1,1); % one net matrix for each layer exclusive input
for i=1:nLayers-2;
    net{i} = ones(P,L(i+1)+1); % affix bias node 
end
net{end} = ones(P,L(end));

% Since we're using batch mode and momentum, two additional matrices are
% needed: prev_dw is the delta weight matrices at time (t-1) and sum_dw
% is the sum of the delta weights for each presentation of the input
% the notation here is the same as net and activation, that is prev_dw{i} 
% is P-by-K matrix where prev_dw{i} is the delta weight matrix for all samples 
% at time (t-1) and sum_dw{i} is a P-by-K matrix where sum_dw{i} is the
% the sum of the weight matrix at layer i for all samples
prev_dw = cell(nLayers-1,1);
sum_dw = cell(nLayers-1,1);
for i=1:nLayers-1
    prev_dw{i} = zeros(size(w{i})); % prev_dw starts at 0
    sum_dw{i} = zeros(size(w{i}));
end    

% loop until computational bounds are exceeded or the network has converged
% to a satisfactory condition. We allow for 30000 epochs here, it may be
% necessary to increase or decrease this bound depending on the number of 
% training
while mse > smse && epochs < 30000
    % FEEDFORWARD PHASE: calculate input/output off each layer for all samples
    for i=1:nLayers-1
        net{i} = a{i} * w{i}'; % compute inputs to current layer
        
        % compute activation(output of current layer, for all layers
        % exclusive the output, the last node is the bias node and
        % its activation is 1
        if i < nLayers-1 % inner layers
            a{i+1} = [2./(1+exp(-net{i}(:,1:end-1)))-1 ones(P,1)];
        else             % output layers
            a{i+1} = 2 ./ (1 + exp(-net{i})) - 1;
        end
    end
    
    % calculate sum squared error of all samples
    err = (D-a{end});       % save this for later
    sse = sum(sum(err.^2)); % sum of the error for all samples, and all nodes
    
    % BACKPROPAGATION PHASE: calculate the modified error at the output layer: 
    % S'(Output) * (D-Output) in this case S'(Output) = (1+Output)*(1-Output)
    % then starting at the output layer, calculate the sum of the weight 
    % matrices for all samples: LearningRate * ModifiedError * Activation
    % then backpropagate the error such that the modified error for this
    % layer is: S'(Activation) * ModifiedError * weight matrix
    delta = err .* (1 + a{end}) .* (1 - a{end});
    for i=nLayers-1:-1:1
        sum_dw{i} = n * delta' * a{i};
        if i > 1
            delta = (1+a{i}) .* (1-a{i}) .* (delta*w{i});
        end
    end
    
    % update the prev_w, weight matrices, epoch count and mse
    for i=1:nLayers-1
        % we have the sum of the delta weights, divide through by the 
        % number of samples and add momentum * delta weight at (t-1)
        % finally, update the weight matrices
        prev_dw{i} = (sum_dw{i} ./ P) + (m * prev_dw{i});
        w{i} = w{i} + prev_dw{i};
    end   
    epochs = epochs + 1;
    mse = sse/(P*M); % mse = 1/P * 1/M * summed squared error
end

% return the trained network
Network.structure = L;
Network.weights = w;
Network.epochs = epochs;
Network.mse = mse;