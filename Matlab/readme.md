Description

This page lists two programs backpropagation written in MATLAB take from chapter 3 of . the textbook, "Elements of Artificial Neural Networks".
Please note that they are generalizations, including momentum and the option to include as many layers of hidden nodes as desired. If you plan on using either of these, it is recommended to hard code the number of layers and nodes at each layer (remembering that more than two is usually undesirable, unrecemmonded, unnessary, and inefficient). Likewise, if you are not using momentum it is recommended to remove the momentum code as this will result in a faster algorithm. I have left these as generalizations because they are easy to use initially when the values of parameters are unknown and you want to determine appropriated parameters.
Source Code

per-epoch backpropagation in MATLAB
per-period backpropagation in MATLAB
Both of these files use the hyperbolic tangent function, for bipolar data. If you want to use a binary sigmoid function, replace the following lines
For the feedforward phase
line 146 in bbackprop.m with a{i+1} = [1./(1+exp(-net{i}(:,1:end-1))) ones(P,1)];
line 148 in bbackprop.m with a{i+1} = 1 ./ (1 + exp(-net{i}));
and
line 131 in backprop.m with a{i+1} = [1./(1+exp(-net{i}(1:end-1))) ; 1];
line 133 in backprop.m with a{i+1} = 1./(1+exp(-net{i}));
For the error propagation phase
line 162 in bbackprop.m with delta = err .* a{end} .* (1 - a{end});
line 166 in bbackprop.m with delta = a{i} .* (1-a{i}) .* (delta*w{i});
and
line 145 in backprop.m with delta = (Dp-a{end}) .* a{end} .* (1-a{end});
line 150 in backprop.m with delta = (1+a{i}) .* a{i}.*(delta'*w{i})';
