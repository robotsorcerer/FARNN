# MLP Training
## Training with mse on soft robots data

With a learning rate of 1e-4 and using the mlp model on soft robot, mI achieved convergence but training was slow.; important to notice there is no dropout or momentum in this implementation; batchNorm seems pretty useless for this training as the error reduction is not clearly discernible

### Learning rate = 1e-2
Changed learning rate to 1e-2 to see if I could get better convergence but the minimization was zagging round +/- 20 decrease in error

### Learning rate = 1e-3 [th main.lua -data softRobot -model mlp -plot -optimizer mse -batchNorm
]
brought error in first epoch down from 48716.261719 to 129.988235. Sounds like 1e-3 is a good learning rate.

### Number of epochs set to 25

#RNN Model Training
##Sept 2, 6:03pm
By some magic of luck or cleaning up of this code, I got the model to work well with (i) a standalone feedforward mlp and (ii) a feedforward mlp followed by an rnn using a learning rate of 1e-3/1e-4.

I've separated the model constructions to a separate file/module to make the errors easier to spot during training. Changingthe set-up to a the twist motion means changing the number of outputs in the dataparserr file now. Given my previous strugges in Reading on this project, I've gotta say this is a remarkable achievement.

### Parameters
If you changed to a SISO model under rnn for the soft Robots data, remember to set the following parameters in dataparser

```lua
	  ninputs     = 1; noutputs    = 1; nhiddens = 1; nhiddens_rnn = 1
```
I found the learning rate of 1e-3 to be optimal for the rnn model

And in `model.lua`, remember to do `neunet = nn.Sequencer(neunet)` rather than `neunet    = nn.Repeater(neunet, noutputs)`

# LSTM Training
## Fast LSTM --Sep 03, 2016
Use fast lstm with a dropout probability of .35. Three hidden layers each with 1, 10 and 100 neurons respectively.  

Trained for 50 epochs each of `softRobot_lstm-net.t7`, 
`softRobot_fastlstm-net.t7`, `softRobot_gru-net.t7`, `softRobotrnn-net.t7` and 

## LSTMs

Recurrent networks use their feedback connections to store representations of recent input events in the form of activations (i.e. <i>short-term memory</i> compared against "long-term memory by slowly changing weights.") Such is important for tasks such as speech processingm non-Markovian Control and music composition.

###RNN problem: With conventional "Back-propagation Through Time" (e.g. Werbos 1998) or Real Time Recurrent Learning (RTRL) error signals flowing backwards in time tend to either (i) explode or (ii) vanish such that the temporal evolution of the backpropagated error exponentially depend on the size if the weights. With exploding gradients, you could have oscillating weights while for vanishing gradients, learning to bridge long time delays is prohibitively costly in time or never works.

###LSTM Remedy: Learning to store information over a long period of time using recurrent backpropagation takes a long period of time, mostly due to lack of adequate and decaying error back floww. LSTM is a gadient-based method that truncates the gradient where it does no harm in the network. An LSTM can learn to bridge minimal time lags in excess of 1000 discrete time steps by enforcing <i>constant error flow </i> through "constant error carousels" within special units. [HochReiter1997](LSTM). Multiplicative gate units learn to open and close access to the constant error flow. LSTM is local in space and time and its computational complexity per time step and weight is $\mathcal{O}(1)$. For long-term delay-infested (:D) systems, LSTMs solves recurrent neural network algorithms faster and better.

LSTMs are designed to avoid the long-term dependency problem that recurrent neural networks are known to have. They are specialists in remembering information for a long period of time. 

LSTMS are recurrent neural network architecture variants that overcome the error back-flow problems by bridging time intervals exceeding 1000 steps even for noisy incomprehensible input sequences without loss if short time lag capabilities. The LSTM architecture enforces constant error flow through the internal states of the special units iff the gradient computation is truncated at certain architecture specific points. LSTMs use multiplicative units as second order nets to protect error flow from unwanted perturbations. To avoid long-term problems of gradient-based learning algorithms, LSTMs are initialized using simple weight-guessing to randomly initialize all network weights until the resulting net is able to correctly classify all the training sequences.While weight-guessing solves many simple tasks, it is not a fool-proof way of solving more complicated tasks which may require many free parameters (e.g. input weights) or high weight precision (e.g. for continuous-valued parameters).

#### Constant error flow: A naive approach
To avoid vanishing error signals, constant error flow through a single unit $j$ with a single connection to itself can be achieved vy setting 

\begin{align}
	{f_j}^'(net_j(k))w_{jj} = 1.0 
\end{align}

in the error back flow if $\theta_j$ given by 

\begin{align}
\theta_j(k) = {f_j}^'(net_j(k)) \theta_j(k+1) w_{jj}
\end{align}

<b>The constant error carousel: </b> Integrating the differential equation above, we have $f_j(net_j(k)) = \frac{net_j(k)}{w_{jj}} for arbitrary net_j(k). This means that $f_j$ has to be linear and unit j's activation has to remain constant:

\begin{align}
y_j(k+1) = f_j(net_j(k+1)) = f_j(w_{jj}y^j(k)) = y^j(k)
\end{align}

In experiments, this is ensured by using the identity function $f_j: f_j(x= = x, \forall x, and by setting w_jj = 1.0. We refer to this as the constant error carousek (CEC). CEC will be LSTM's central feature. )

#### Memory cells and gate units: To 



### System integration

Install boost with chrono and thread and then install torch-ros
e.g. First copy bjam from tools/build to /usr/bin and then do the following in order

```bash
	lex@lex:/usr/local/boost_1_61_0$ bjam install libs/chrono libs/thread
	lex@lex:/usr/local/boost_1_61_0$ sudo bjam install --prefix=/usr/local/boost_1_61_0 --with-chrono --with-thread
```
