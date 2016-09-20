# Nonlinear Systems Identification Using Deep Dynamic Neural Networks

Author: Olalekan Ogunmolu         

This repo contains the code for reproducing the results introduced in the paper, [Nonlinear Systems Identification Using Deep Dynamic Neural Networks](http://ecs.utdallas.edu/~opo140030/vitae.html).

## Maintainer

- [Olalekan Ogunmolu](http://ecs.utdallas.edu/~olalekan.ogunmolu) 

## Table of contents
- [Description](#description)
- [Dependencies](#Dependencies)
- [Modifications](#modifications)
- [Test Code](#test-code)
- [Options](#options)


## Description
Nonlinear Systems Identification Using Deep Dynamic Neural Networks

## Dependencies

This code is written in lua/torch and compiles with Torch7. I recommend you follow the instructions on the [torch website](http://torch.ch/docs/getting-started.html) to get the torch7 package installed. Typical installation would include running the following  commands in a terminal

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

By default, the code runs on GPU 0 and to get things running on CUDA we need to add a few more dependencies namely `cunn`, `cudnn`, and `cutorch`. 

First, you will have to install the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) by downloading the debian from NVIDIA's website and `dpkg` installing. Otherwise, this [bash script](/cuda.sh) will fetch the source files for CUDA 7.0 and install it on your computer.

If you'd like to use the cudnn backend (this is enabled by default), you also have to install [cudnn's torch wrapper](https://github.com/soumith/cudnn.torch). First follow the link to [NVIDIA website](https://developer.nvidia.com/cuDNN), register with them and download the cudnn library. Then make sure you adjust your `LD_LIBRARY_PATH` to point to the `lib64` folder that contains the library (e.g. `libcudnn.so.7.0.64`). Then git clone the `cudnn.torch` repo, `cd` inside and do `luarocks make cudnn-scm-1.rockspec` to build the Torch bindings.

We would also need other Torch packages including `nn` and `matio`. Doing this in a terminal would list the rocks you have:

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

To run the Hammerstein models described in the paper, we do

```bash
	luarocks install rnn
```

## Test code

To test this code, make sure `posemat7.mat`	 is in the root directory of your project. Then run the `farnn.lua` script as

```bash
	th main.lua
```

## Training Models
By default, this trains with the Hammerstein LSTM architecture described in the paper. 
To use a different model such as `mlp`, `fastlstm`, `rnn` or `gru`, do the following. To train on a specific dataset that was mentioned in the paper, pass the name of the dataset as a command line argument before training (e.g., `-data softRobot` for soft-robot dataset or `-data glasssurface` for glassfurnace dataset).

### MLP Models

```bash
	th main.lua -model mlp
```

### RNN Models
```bash
	th main.lua -model rnn
```

### FastLSTM models
```
	th main.lua -fastlstm
```

### GRU Models

```
	th main.lua -gru
```

## Options
	
* `-seed`, 123, 'initial seed for random number generator'
* `-silent`, true, 'false|true: 0 for false, 1 for true'
* `-dir`, 'outputs', 'directory to log training data'

-- Model Order Determination Parameters
* `-data`,'glassfurnace','path to -v7.3 Matlab data e.g. robotArm | glassfurnace | ballbeam | soft_robot'
* `-tau`, 5, 'what is the delay in the data?'
* `-m_eps`, 0.01, 'stopping criterion for output order determination'
* `-l_eps`, 0.05, 'stopping criterion for input order determination'
* `-trainStop`, 0.5, 'stopping criterion for neural net training'
* `-sigma`, 0.01, 'initialize weights with this std. dev from a normally distributed Gaussian distribution'

--Gpu settings
* `-gpu`, 0, 'which gpu to use. -1 = use CPU; >=0 use gpu'
* `-backend`, 'cudnn', 'nn|cudnn'

-- Neural Network settings
* `-learningRate`,1e-3, 'learning rate for the neural network'
* `-learningRateDecay`,1e-3, 'learning rate decay to bring us to desired minimum in style'
* `-momentum`, 0.9, 'momentum for sgd algorithm'
* `-model`, 'lstm', 'mlp|lstm|linear|rnn'
* `-gru`, false, 'use Gated Recurrent Units (nn.GRU instead of nn.Recurrent)'
* `-fastlstm`, false, 'use LSTMS without peephole connections?'
* `-netdir`, 'network', 'directory to save the network'
* `-optimizer`, 'mse', 'mse|sgd'
* `-coefL1`,   0.1, 'L1 penalty on the weights'
* `-coefL2`,  0.2, 'L2 penalty on the weights'
* `-plot`, true, 'true|false'
* `-maxIter`, 10000, 'max. number of iterations; must be a multiple of batchSize'

-- RNN/LSTM Settings 
* `-rho`, 5, 'length of sequence to go back in time'
* `-dropout`, true, 'apply dropout with this probability after each rnn layer. dropout <= 0 disables it.'
* `-dropoutProb`, 0.35, 'probability of zeroing a neuron (dropout probability)'
* `-rnnlearningRate`,1e-3, 'learning rate for the reurrent neural network'
* `-decay`, 0, 'rnn learning rate decay for rnn'
* `-batchNorm`, false, 'apply szegedy and Ioffe\'s batch norm?'
* `-hiddenSize`, {1, 10, 100}, 'number of hidden units used at output of each recurrent layer. When more thanone is specified, RNN/LSTMs/GRUs are stacked')
* `-batchSize`, 100, 'Batch Size for mini-batch training'