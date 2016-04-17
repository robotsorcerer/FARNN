p = [-1 -1 2 2 ;
      0  5 0 5];

t = [-1 -1 1 1];

net = feedforwardnet(3, 'traingd');
net.divideFcn = '';

%modify some of the default training parameters
net.trainParam.show = 50;
net.trainParam.lr = 0.05;
net.trainParam.epochs = 300;
net.trainParam.goal = 1e-5;

%now train the network
[net, tr] = train(net, p, t);

disp( tr);

%% Create, configure, and initialize multilayer neural networks
house = load('house_dataset')

net = feedforwardnet;
net = configure(net, houseInputs, houseTagets);
