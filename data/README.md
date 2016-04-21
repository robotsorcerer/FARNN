# Learning Deep Neural Network Policies During H&N Motion Control in Clinical Cancer Radiotherapy

## Maintainer

- [Olalekan Ogunmolu](http://lakehanne.github.io) 

## Table of contents
- [Description](#description)
- [Dependencies](#Dependencies)
- [Modifications](#modifications)
- [Test Code](#test-code)
- [Options](#options)


## Description
A fully automated recurrent neural network for unknown dynamic system identification and control based on the 2006 IEEE Transactions on Circuits and Systems by Jeen-Shing Wang and Yen-Ping Chen.

## Dependencies

This code is written in lua/torch and needs compiling with Torch7. I recommend you follow the instructions on the [torch website](http://torch.ch/docs/getting-started.html) to get your torch7 package. Typical installation would include running the following  commands in a terminal

```markdown
	curl -s https://raw.githubusercontent.com/torch/ezinstall/master/install-deps | bash
	git clone https://github.com/torch/distro.git ~/torch --recursive
	cd ~/torch; ./install.sh
```

Then add the installation to your path variable by doing:

```markdown
	# On Linux
	source ~/.bashrc
	# On OSX
	source ~/.profile
```

YMMV, but it is good if you have a CUDA-capable GPU to be able to run this code optimally. Also, after the installations above, you want to check if you have `nn`, `cunn`, `cudnn`, `cutorch` and `matio`. Doing this in a terminal would list the rocks you have:

<pre><code class="Terminal">luarocks list</code></pre>

If the above command does not list the above dependencies, you can install them via the following commands

- NN

```bash
	luarocks install nn
```

- CUNN

```bash
	luarocks install cunn
```

- CUTORCH

```bash
	luarocks install cutorch
```

- MATIO

On Ubuntu, you can simply install the matio  development library from the ubuntu repositories. Do <pre><code class="Terminal">sudo apt-get install libmatio2</code></pre>

Then do

```bash
	luarocks install matio
```

- [RNN](https://github.com/Element-Research/rnn)

```bash
	cd ~/torch;
	git clone https://github.com/Element-Research/rnn.git
	cd rnn/rocks; 
	luarocks rnn-scm-1.rockspec
```

If you want to use a GPU, you will have to install the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit). Or you could copy out the following lines from this [bash script](https://github.com/lakehanne/Shells/blob/master/packages.sh#L282-L313), place it in a new bash script and it will install cuda-7.0 on your computer.

If you'd like to use the cudnn backend (to speed up the training of your dataset), you also have to install [cudnn](https://github.com/soumith/cudnn.torch). First follow the link to [NVIDIA website](https://developer.nvidia.com/cuDNN), register with them and download the cudnn library. Then make sure you adjust your `LD_LIBRARY_PATH` to point to the `lib64` folder that contains the library (e.g. `libcudnn.so.7.0.64`). Then git clone the `cudnn.torch` repo, `cd` inside and do `luarocks make cudnn-scm-1.rockspec` to build the Torch bindings.


## Modifications to original algorithm

This is my implementation of a SIMO-based nonlinear function approximator of an input-output mapping using among others the algorithm described in Wang and Chen's [paper](http://ieeexplore.ieee.org/xpl/abstractAuthors.jsp?arnumber=1643442), [He and Asada's 1993 work](http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=4793346&url=http%3A%2F%2Fieeexplore.ieee.org%2Fxpls%2Fabs_all.jsp%3Farnumber%3D4793346), [On rectified linear units for speech processing. ICASSP, IEEE International Conference on Acoustics, Speech and Signal Processing - Proceedings.](http://doi.org/10.1109/ICASSP.2013.6638312) and  [Feedforward networks training speed enhancement by optimal initialization of the synaptic coefficients. IEEE Transactions on Neural Networks, 12(2), 430–434.](http://doi.org/10.1109/72.914538). Feel free to modify the code to a MIMO system as you might want.

## Test code

To test this code, make sure `posemat7.mat` is in the root directory of your project. If you have other data other than the above (i.e.single input, six outputs), pass it to the script trainer, `rnn.lua`, using the `-pose` argument. The code will compute the input, output and overall order of your nonlinear system based on [He and Asada's 1993 work](http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=4793346&url=http%3A%2F%2Fieeexplore.ieee.org%2Fxpls%2Fabs_all.jsp%3Farnumber%3D4793346) of using Lipschitz coefficients to estimate input-output model orders, it will compute the optimal number of input variables and then perform the deep network training required to identify the system structure.

There are two ways to run the script. 1) With an mlp network, i.e. 1 input, 1 hidden layer and the outputs; and 2) with an mlp feedforward network followed by a recurrent network as originally proposed by Wang and Chen. I have made some changes to the original Wang and Chen argument by using an nn.ReLU() squasher instead of the tanh() and sigmoid activators. Also new to this `rnn` implementation is the addition of two hidden layers in the feedforward network instead of one. So instead of a single hidden network, we forward an input layer to a single hidden layer, squash the result with nn.ReLU() and connect this nonlinear layer to a 6 layered hidden node. In the recurrent layer, we first squash feedforward node with a sigmoid function before we forward the six-layered output to a 6-hidden layer node; we then do a self-adaptive feedback of neurons in the recurrent layer back to the output of the feedforward node. The `nn.Repeater` was used to apply all the `Twist` elements of the head motion to the inputs. 

1) In mlp mode, run the `rnn.lua` script as

```bash
	th rnn.lua -model mlp
```
The optimizer is the mean-squared-error by default. The above arguments will train a multilayered perceptron model and save the resulting network to the `network` directory. The directory can be changed by passing a folder name to `-netdir` on the command line.

2) In rnn mode, run the `rnn.lua` script as 

```bash
	th rnn.lua -model rnn
```

to train a simple recurrent neural network and save it as `rnn-net.net` in the `network` directory. By default, the algorithm runs on the gpu. To specify a cpu, pass -1 to the `-gpu` argumaent on the command line.

## Options when runing code

* `-seed`, 		initial random seed to use.
* `-gpuid`,  	which gpu to use. -1 = use CPU; >=0 use gpu.  Default is gpu 0.
* `-quots`,  	do you want to print the Lipschitz quotients?; 0 to silence, 1 to print
* `-maxIter`, 	maximaum iteration for training the neural network.' Default is 50.
*`-batchSize`, 	Batch Size for mini-batch training, \
                            preferrably in multiples of six; default is 6.

* `-rundir`,  	false|true: 0 for false, 1 for true.

* <b>Model Order Determination Parameters</b>
* `-pose`,		data/posemat5.mat','path to preprocessed data(save in Matlab -v7.3 format)
* `-tau`		what is the delay in the data? Default is 1.
* `-m_eps`		stopping criterion for output order determination; default is 0.01.
* `-l_eps`		stopping criterion for input order determination.; default is 0.05.
* `-trainStop`  stopping criterion for neural net training; default is 0.5.
* `-sigma` 		initialize weights with this std. dev from a normally distributed Gaussian distribution. default is 0.01

* <b>Gpu settings</b>
* `-gpu`, 0, 'which gpu to use. -1 = use CPU; >=0 use gpu.
* `-backend`, 	`cudnn`, or `nn|cudnn`. Default is `cudnn`.

*  <b>Neural Network settings</b>
* `-learningRate`,		learning rate for the neural network; default is 1e-2.
* `-rnnlearningRate`,	learning rate for the reurrent neural network; default is 0.1.
* `-learningRateDecay`, 	learning rate decay to bring us to desired minimum in style; default is 1e-6.
* `-momentum`,	momentum for sgd algorithm; default is 0.
* `-model`, 		mlp|lstm|linear|rnn; default is 'mlp'.
* `-rho`,		length of sequence to go back in time; default is  6.
* `-netdir`, 	directory to save the network; default is network.
* `-visualize`, 	visualize input data and weights during training; default is true.
* `-optimizer`, 	mse|l-bfgs|asgd|sgd|cg; default is mse.
* `-coefL1`, 	L1 penalty on the weights; default is 0.
* `-coefL2`, 	L2 penalty on the weights; default is 0.
* `-plot`, 		plot while training; default is false.
* `-maxIter`, 	max. number of iterations; must be a multiple of batchSize; default is 10000.

* <b>LBFGS Settings</b>
* `-Correction`, number of corrections for line search. Max is 100; default is 60.

	