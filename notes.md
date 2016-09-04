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

