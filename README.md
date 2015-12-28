# A fully automated recurrent neural network for unknown dynamic system identification and control

## Maintainer

- [Olalekan Ogunmolu](https://ecs.utdallas.edu/~olalekan.ogunmolu <<olalekan.ogunmolu@utdallas.edu>>, [Sensing, Robotics, Vision, Control and Estimation Lab](http://ecs.utdallas.edu/research/researchlabs/service-lab/), University of Texas at Dallas

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
 
 >luarocks install nn

- CUNN

>luarocks install cunn

- CUTORCH

> luarocks install cutorch

- MATIO

On Ubuntu, you can simply install the matio  development library from the ubuntu repositories. Do <pre><code class="Terminal">sudo apt-get install libmatio2</code></pre>

Then do

>luarocks install matio

If you want to use a GPU, you will have to install the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit). Or you could copy out the following lines from this [bash script](https://github.com/lakehanne/Shells/blob/master/packages.sh#L282-L313), place it in a new bash script and it will install cuda-7.0 on your computer.

If you'd like to use the cudnn backend (to speed up the training of your dataset), you also have to install [cudnn](https://github.com/soumith/cudnn.torch). First follow the link to [NVIDIA website](https://developer.nvidia.com/cuDNN), register with them and download the cudnn library. Then make sure you adjust your `LD_LIBRARY_PATH` to point to the `lib64` folder that contains the library (e.g. `libcudnn.so.7.0.64`). Then git clone the `cudnn.torch` repo, `cd` inside and do `luarocks make cudnn-scm-1.rockspec` to build the Torch bindings.


## Modifications to original algorithm

This is my implementation for a SISO-based input-output mapping recontruction using the algoithm described in Wang and Chen's [paper](http://ieeexplore.ieee.org/xpl/abstractAuthors.jsp?arnumber=1643442). Feel free to modify the code to a MIMO system as you might want.

- Rather than using the sigmoidal network activation functions which are prone to saturation in the hidden layers of the network, I adopt the rectified linear units, or ReLUs, as proposed by Zeiler, M. D., Ranzato, M., Monga, R., Mao, M., Yang, K., Le, Q. V., … Hinton, G. E. (2013) in their Google paper:  [On rectified linear units for speech processing. ICASSP, IEEE International Conference on Acoustics, Speech and Signal Processing - Proceedings.](http://doi.org/10.1109/ICASSP.2013.6638312). 

This avoids the active region consuming too much memory bandwidth and reduces algorithm complexity in terms of finding optimal regions within the input nonlinearity activation function that is non-saturated such as Yam and Chow proposed in the 2001 paper [Feedforward networks training speed enhancement by optimal initialization of the synaptic coefficients. IEEE Transactions on Neural Networks, 12(2), 430–434.](http://doi.org/10.1109/72.914538).

## Testing code

To test this code, make sure `posemat7.mat`	 is in the root directory of your project. Then run the `farnn.lua` script as

```bash
	th farnn.lua
```

The code will compute the input, output and overall order of your nonlinear system based on [He and Asada's 1993 work](http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=4793346&url=http%3A%2F%2Fieeexplore.ieee.org%2Fxpls%2Fabs_all.jsp%3Farnumber%3D4793346) of using Lipschitz coefficients for estimating input-output model orders, it will compute the optimal number of input variables and then perform the multilayered perceptron training required to identify your system structure

## Options

* '-seed', 123, 'initial random seed to use'
* '-gpuid', 0, 'which gpu to use. -1 = use CPU; >=0 use gpu'
* '-pose','posemat7.mat','path to preprocessed data(save in Matlab -v7.3 format'
* '-tau', 1, 'what is the delay in the data?'
* '-m_eps', 0.01, 'stopping criterion for order determination'
* '-l_eps', 0.05, 'stopping criterion for input order determination'
* '-backend', 'cudnn', 'nn|cudnn'
* '-gpuid', 0, 'which gpu to use. -1 = use CPU; >=0 use gpu '
* '-quots', 0, 'do you want to print the Lipschitz quotients?; 0 to silence, 1 to print'
* '-maxIter', 50, 'maximaum iteratiopn for training the neural network'

