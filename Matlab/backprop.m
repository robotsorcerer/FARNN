% backprop a per-period backpropagation training for a multilayer feedforward
%          neural network.
%   Network = backprop(Layers,N,M,SatisfactoryMSE,Input,Desired) returns 
%   Network, a two field structure of the form Network.structure = Layers 
%   and Network.weights where weights is a cell array specifying the final 
%   weight matrices computed by minimizing the mean squared error between 
%   the Desired output and the actual output of the network given a set of 
%   training samples: Input and the SatisfactoryMSE (satisfactory mean 
%   squared error)
%
%   Input:
%    Layers - a vector of integers specifying the number of nodes at each
%     layer, i.e for all i, Layers(i) = number of nodes at layer i, there
%     must be at least three layers and the input layer Layers(1) must
%     equal the dimension of each vector in Input, likewise, Layers(end) 
%     must be equal to the dimension of each vector in Desired
%     N - training rate for network learning (0.1 - 0.9)
%     M - momentum for the weight update rule [0.1 - 0.9)
%     SatisfactoryMSE - the mse at which to terminate computation
%     Input - the training samples, a P-by-N matrix, where each Input[p] is
%      a training vector
%     Desired - the desired outputs, a P-by-M matrix where each Desired[p]
%      is the desired output for the corresponding input Input[p]
%
%   This algorithm uses the hyperbolic tangent node function 
%   2/(1+e^(-net)) - 1, for use with bipolar data
%   
%   NOTE: due to its generality this algorithm is not as efficient as a 
%   one designed for a specific problem if the number of desired layers is 
%   known ahead of time, it is better to a) 'unfold' the loops inside the 
%   loop presenting the data. That is, calculate the input and output of each 
%   layer explicitly one by one and subsequently the modified error and weight 
%   matrix modifications b) remove momentum and training rate as parameters
%   if they are known
%
% Author: Dale Patterson
% $Version: 2.2.1 $ $Date: 2.25.06 $
% 
function Network = backprop(L,n,m,smse,X,D)

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

% will use the number of layers often, so save the number here
nLayers = length(L); 

% randomize the weight matrices (uniform random values in [-.5 .5], there
% is a weight matrix between each layer of nodes. Each layer (exclusive the 
% output layer) has a bias node whose activation is always 1, that is, the 
% node function is C(net) = 1. Furthermore, there is a link from each node
% in layer i to the bias node in layer j (the last row of each matrix)
% because this is less computationally expensive then the alternative.
% NOTE: below that the wieghts of all links to bias nodes are defined as
% zero
w = cell(nLayers-1,1); % a weight matrix between each layer
for i=1:nLayers-2
    w{i} = [.5 - rand(L(i+1),L(i)+1) ; zeros(1,L(i)+1)];
end
w{end} = .5 - rand(L(end),L(end-1)+1);

X = [X ones(P,1)]; % affix the column of bias activations to the input layer

% preallocate activation,net vectors and delta weight matrices for faster 
% computation
% activation vectors, all but output layer include bias activation
a = cell(nLayers,1);
for i=1:nLayers-1
    a{i} = ones(L(i)+1,1);
end
a{end} = ones(L(end),1);

% net vectors, one for each node in that layer but there is 
% no net for input layer
net = cell(nLayers-1,1);
for i=1:nLayers-2;
    net{i} = ones(L(i+1)+1,1);
end
net{end} = ones(L(end),1);

% delta weight matrices
dw = cell(nLayers-1,1);
for i=1:nLayers-1
    dw{i} = zeros(size(w{i}));
end

% initialize stopping conditions
mse = Inf;  % assuming the intial weight matrices are bad
presentations = 0; % we'll measure by epoch instead of presentation

% loop until computational bounds are exceeded or the network has converged
% to a satisfactory condition. We allow for 30000 epochs, it may be
% necessary to reduce this if the number of training samples is large
while mse > smse && presentations < P * 10000
    sse = 0; % running total of squared error
    for p=1:P 
        % get the current input vector and desired output
        a{1} = X(p,:)';
        Dp = D(p,:)';

        % compute the inputs and outputs to each layer
        for i=1:nLayers-1
            % compute inputs to this layer
            net{i} = w{i} * a{i}; 

            % compute outputs of this layer
            % for all layers but the output layer, the last node is the 
            % bias node and its activation is 1
            if i < nLayers-1        
                a{i+1} = [2./(1+exp(-net{i}(1:end-1)))-1 ; 1];
            else
                a{i+1} = 2./(1+exp(-net{i})) - 1;
            end
        end
        
        % accumlate the squared error
        sse = sse + sum((Dp-a{end}).^2);
        
        % calculate the modified error at each layer and update the weight 
        % matrices accordingly. first calculate delta, the modified error
        % for the output nodes (S'(Output[net])*(Dp-Output[Activation])
        % then for each weight matrix, add n * delta * activation and
        % propagate delta to the previous layer
        delta = (Dp-a{end}) .* (1+a{end}) .* (1-a{end});
        for i=nLayers-1:-1:1
            dw{i} = n * delta * a{i}' + (m .* dw{i});
            w{i} = w{i} + dw{i};
            if i > 1 % dont compute mod err for input layer 
                delta = (1+a{i}).*(1-a{i}).*(delta'*w{i})';
            end
        end
    end
    presentations = presentations + P;
    mse = sse/(P*M); % mse = 1/P * 1/M * summed squared error
end

% return the trained network
Network.structure = L;
Network.weights = w;
Network.mse = mse;
Network.presentations = presentations;