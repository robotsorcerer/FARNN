--[[Source Code that implements the algorithm described in the paper:
   A Fully Automated Recurrent Neural Network for System Identification and Control
   Jeen-Shing Wang, and Yen-Ping Chen. IEEE Transactions on Circuits and Systems June 2006

   Author: Olalekan Ogunmolu, SeRViCE Lab, UT Dallas, December 2015
   MIT License
   ]]

-- needed dependencies
require 'torch'
require 'nn'
require 'optim'
require 'image'
matio   	= require 'matio'  
orderdet 	= require 'order.order_det'     

--[[modified native Torch Linear class to allow random weight initializations
 and avoid local minima issues ]]
do
    local Linear, parent = torch.class('nn.CustomLinear', 'nn.Linear')    
    -- override the constructor to have the additional range of initialization
    function Linear:__init(inputSize, outputSize, mean, std)
        parent.__init(self,inputSize,outputSize)                
        self:reset(mean,std)
    end    
    -- override the :reset method to use custom weight initialization.        
    function Linear:reset(mean,stdv)        
        if mean and stdv then
            self.weight:normal(mean,stdv)
            self.bias:normal(mean,stdv)
        else
            self.weight:normal(0,1)
            self.bias:normal(0,1)
        end
    end
end

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('========================================================================')
cmd:text('A Fully Automated Dynamic Neural Network for System Identification')
cmd:text('Based on the IEEE Transactions on Circuits and Systems article by ')
cmd:text()
cmd:text('           Jeen-Shing Wang, and Yen-Ping Chen. June 2006          ')
cmd:text()
cmd:text()
cmd:text('Code by Olalekan Ogunmolu: FirstName [dot] LastName _at_ utdallas [dot] edu')
cmd:text('========================================================================')
cmd:text()
cmd:text()
cmd:text('Options')
cmd:option('-seed', 123, 'initial random seed to use')
cmd:option('-rundir', 0, 'false|true: 0 for false, 1 for true')

-- Model Order Determination Parameters
cmd:option('-pose','data/posemat5.mat','path to preprocessed data(save in Matlab -v7.3 format)')
cmd:option('-tau', 1, 'what is the delay in the data?')
cmd:option('-quots', 0, 'do you want to print the Lipschitz quotients?; 0 to silence, 1 to print')
cmd:option('-m_eps', 0.01, 'stopping criterion for output order determination')
cmd:option('-l_eps', 0.05, 'stopping criterion for input order determination')
cmd:option('-trainStop', 0.5, 'stopping criterion for neural net training')
cmd:option('-sigma', 0.01, 'initialize weights with this std. dev from a normally distributed Gaussian distribution')

--Gpu settings
cmd:option('-gpu', 0, 'which gpu to use. -1 = use CPU; >=0 use gpu')
cmd:option('-backend', 'cudnn', 'nn|cudnn')

-- Neural Network settings
cmd:option('-learningRate',1e-2, 'learning rate for the neural network')
cmd:option('-learningRateDecay',1e-6, 'learning rate decay to bring us to desired minimum in style')
cmd:option('-maxIter', 200000, 'maximum iteration for training the neural network')
cmd:option('-optimizer', 'mse', 'mse|l-bfgs|adam')

-- LBFGS Settings
cmd:option('-Correction', 60, 'number of corrections for linesearch. Max is 100')
cmd:option('-linesearch', 0.5, 'Line Search')

-- Print options
cmd:option('-print', 0, 'false = 0 | true = 1 : Option to make code print neural net parameters')  -- print System order/Lipschitz parameters



-- misc
local opt = cmd:parse(arg)
torch.manualSeed(opt.seed)

-- create log file if user specifies true for rundir
if(opt.rundir==1) then
	opt.rundir = cmd:string('experiment', opt, {dir=false})
	paths.mkdir(opt.rundir)
	cmd:log(opt.rundir .. '/log', opt)
end

cmd:addTime('FARNN Identification', '%F %T')
cmd:text('Code initiated on CPU')
cmd:text()
cmd:text()

-------------------------------------------------------------------------------
-- Fundamental initializations
-------------------------------------------------------------------------------
--torch.setdefaulttensortype('torch.FloatTensor')            -- for CPU
data        = opt.pose 
if opt.gpu >= 0 then
  require 'cutorch'
  require 'cunn'
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.gpu + 1)                         -- +1 because lua is 1-indexed
  idx       = cutorch.getDevice()
  print('System has', cutorch.getDeviceCount(), 'gpu(s).', 'Code is running on GPU:', idx)
end

if opt.backend == 'cudnn' then
 require 'cudnn'
else
  opt.backend = 'nn'
end

----------------------------------------------------------------------------------------
-- Parsing Raw Data
----------------------------------------------------------------------------------------
input       = matio.load(data, 'in')						--SIMO System
out         = matio.load(data, {'xn', 'yn', 'zn', 'rolln', 'pitchn',  'yawn' })

local   k   = input:size()[1]

y           = {out.xn, out.yn, 
               out.zn, out.rolln, 
               out.pitchn, out.yawn}

--Determine training data               
off         = torch.ceil( torch.abs(0.6*k))
u_off       = input[{{1, off}, {1}}]     --offline data
y_off       = {
               out.xn[{{1, off}, {1}}], out.yn[{{1, off}, {1}}], 
               out.zn[{{1, off}, {1}}], out.rolln[{{1, off}, {1}}], 
               out.pitchn[{{1, off}, {1}}], out.yawn[{{1, off}, {1}}] 
              }

--print('k', input:size()[1], 'off', off, '\nout\n', out, '\ny_off\n', y_off)
--print('xn', out.xn[{{1, off}, {1}}])

u_on        = input[{{off + 1, k}, {1}}]	--online data
y_on       = {
               out.xn[{{off+1, k}, {1}}], out.yn[{{off+1, k}, {1}}], 
               out.zn[{{off+1, k}, {1}}], out.rolln[{{off+1, k}, {1}}], 
               out.pitchn[{{off+1, k}, {1}}], out.yawn[{{off+1, k}, {1}}] 
              }

print('u_on', u_on:size(), 'y_on', y_on)              


--[[Determine input-output order using He and Asada's prerogative
    See Code order_det.lua in folder "order"]]

--find optimal # of input variables from data
qn  = order_det.computeqn(u_off, y_off[3])

--compute actual system order
utils = require 'order.utils'
inorder, outorder, q =  order_det.computeq(u_off, y_off[3], opt)


--[[Set up the network, add layers in place as we add more abstraction]]
local function contruct_net()
  input = 1 	 output = 1 	HUs = 1;
  local neunet 	  = {}
        neunet        	= nn.Sequential()
        neunet:add(nn.Linear(input, HUs))
        neunet:add(nn.ReLU())                       	
        neunet:add(nn.Linear(HUs, output))	
  return neunet			
end

neunet          = contruct_net()
neunetnll       = neunet:clone('weight', bias);
neunetlbfgs     = neunet:clone('weight', bias);collectgarbage()

local function perhaps_print(q, qn, inorder, outorder, input, out, off, y_off)
  print('\nqn:' , qn)
  print('Optimal number of input variables is: ', torch.ceil(qn))
  print('inorder: ', inorder, 'outorder: ', outorder)
  print('system order:', inorder + outorder)

  print('\ninput head', input[{ {1,5}, {1}} ]) 
  print('k', input:size()[1], 'off', off, '\nout\n', out, '\ny_off\n', y_off)
  print('\noutput head\n\n', out[{ {1,5}, {1}} ])
  --Print out some Lipschitz quotients (first 5) for user
  if opt.quots == 1 then
   for k, v in pairs( q ) do
    print(k, v)
    if k == 5 then break end
    end
  end
  --print neural net parameters
  print('neunet biases Linear', neunet.bias)
  print('\nneunet biases\n', neunet:get(1).bias, '\tneunet weights: ', neunet:get(1).weights)
end

if (opt.print==1) then perhaps_print(q, qn, inorder, outorder, input, out, off, y_off) end
--[[ Run optimization: User has three options:
We could train usin the genral mean squared error, Limited- Broyden-Fletcher-GoldFarb and Shanno or the
Negative Log Likelihood Function ]]

--[[Declare states for limited BFGS
 See: https://github.com/torch/optim/blob/master/lbfgs.lua
]]
local state = nil
local config = nil

if opt.optimizer == 'l-bfgs' then
  state = 
  {
    maxIter = opt.maxIter,  --Maximum number of iterations allowed
    verbose=true,
    maxEval = 1000,      --Maximum number of function evaluations
    tolFun  = 1e-1      --Termination tol on progress in terms of func/param changes
  }

  config =   
  {
    nCorrection = opt.Correction,    
    lineSearch  = opt.linesearch,
    lineSearchOpts = {},
    learningRate = opt.learningRate
  }

elseif opt.optimizer == 'adam' then
  state = {
    learningRate = opt.learningRate
  }

elseif opt.optimizer == 'nll' then
  state = {
    learningRate = opt.learningRate
  }

elseif opt.optimizer == 'mse' then
  state = {
    learningRate = opt.learningRate
  }

else
  error(string.format('Unrecognized optimizer "%s"', opt.optimizer))
end

--[[ Function to evaluate loss and gradient.
-- optim.lbfgs internally handles iteration and calls this fucntion many
-- times]]

local num_calls = 0
local function grad(x, y)
  num_calls = num_calls + 1
  neunetlbfgs:forward(x, y)
  local grad = neunetlbfgs:backward(x, y)
  local loss = 0
  for _, mod in ipairs(content_losses) do
    loss = loss + mod.loss
  end
  for _, mod in ipairs(style_losses) do
    loss = loss + mod.loss
  end
  maybe_print(num_calls, loss)
  maybe_save(num_calls)

  collectgarbage()
  -- optim.lbfgs expects a vector for gradients
  return loss, grad:view(grad:nElement())
end

--Train using the Negative Log Likelihood Criterion
function nllOptim(neunetnll, u_off, y_off, learningRate)
  local x = u_off   local y = y_off	
   local mse_crit      	= nn.MSECriterion()
   local trainer        = nn.StochasticGradient(neunet2, mse_crit)
   trainer.learningRate = learningRate
   trainer:train({u_off, y_off})
   return neunet2
end

--Training using the MSE criterion
local i = 0
local function msetrain(neunet, x, y, learningRate)
  --https://github.com/torch/nn/blob/master/doc/containers.md#Parallel
    pred      = neunet:forward(x)
    --print ('pred', pred)

    cost      = nn.MSECriterion()           -- Loss function
    err       = cost:forward(pred, y)
    gradcrit  = cost:backward(pred, y)


    --https://github.com/torch/nn/blob/master/doc/module.md
    --neunet:accGradParameters(x, pred, 1)    --https://github.com/torch/nn/blob/master/doc/module.md#nn.Module.accGradParameters
    neunet:zeroGradParameters();
    neunet:backward(x, gradcrit);
    neunet:updateParameters(learningRate);
   -- print(err)
return pred, err
end
collectgarbage()                           --yeah, sure. come in and argue :)


local i = {}
if opt.optimizer == 'mse' then
	print('Running optimization with mean-squared error')
	for i = 0, opt.maxIter do
		pred, mse_error = msetrain(neunet, u_off, y_off[3], opt.learningRate)
		if mse_error > 150 then learningRate = opt.learningRate
		elseif mse_error <= 150 then learningRate = opt.learningRateDecay end
    i = i + 1   
		print('MSE iteration', i, '\tMSE error: ', mse_error)
    --'\tPrediction', pred, 
		-- print('neunet gradient weights', neunet.gradWeight)
		-- print('neunet gradient biases', neunet.gradBias)
	end

elseif opt.optimizer == 'l-bfgs' then
  print('Running optimization with L-BFGS')
  for i = 0, opt.maxIter do
  	local losses = optim.lbfgs(grad, u_off, config, state)
  	i = i + 1
  	if losses > 150 then learningRate = opt.learningRate
  	elseif losses <= 150 then learningRate = opt.learningRateDecay end
  	print('lbfgs iter', i,  'error', losses)
  end
 
elseif opt.optimizer == 'nll' then
	print('Running optimization with negative log likelihood criterion')
  	for i = 0, opt.maxIter do
  		delta = nllOptim(neunetnll, u_off, y_off[3], opt.learningRate)
  		i = i + 1
  		if delta > 150 then learningRate = opt.learningRate
  		elseif delta <= 150 then learningRate = opt.learningRateDecay end
  		print('nll iter', i, 'bkwd error', t, 'fwd error', delta )
  	end
end