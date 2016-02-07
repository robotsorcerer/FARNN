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
optim_    = require 'optima._optim'  

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
cmd:option('-momentum', 0, 'momentum for sgd algorithm')
cmd:option('-optimizer', 'mse', 'mse|l-bfgs|adam')

-- LBFGS Settings
cmd:option('-Correction', 60, 'number of corrections for linesearch. Max is 100')
cmd:option('-batchSize', 10, 'Batch Size')

-- Print options
cmd:option('-print', 0, 'false = 0 | true = 1 : Option to make code print neural net parameters')  -- print System order/Lipschitz parameters



-- misc
local opt = cmd:parse(arg)
torch.manualSeed(opt.seed)

torch.setnumthreads(4)

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
k           = input:size()[1]

--geometry of input
geometry    = {k, input:size()[2]}

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
y_on        = {
               out.xn[{{off+1, k}, {1}}], out.yn[{{off+1, k}, {1}}], 
               out.zn[{{off+1, k}, {1}}], out.rolln[{{off+1, k}, {1}}], 
               out.pitchn[{{off+1, k}, {1}}], out.yawn[{{off+1, k}, {1}}] 
              }

off_data = {u_off, y_off}
on_data  = {u_on,  y_on}
 


--[[Determine input-output order using He and Asada's prerogative
    See Code order_det.lua in folder "order"]]

--find optimal # of input variables from data
qn  = order_det.computeqn(u_off, y_off[3])

--compute actual system order
utils = require 'order.utils'
inorder, outorder, q =  order_det.computeq(u_off, y_off[3], opt)


--[[Set up the network, add layers in place as we add more abstraction]]
local function contruct_net()
  local input = 1 	 output = 1 	HUs = 1;
  local neunet 	  = {}
        neunet        	= nn.Sequential()
        neunet:add(nn.Linear(input, HUs))
        neunet:add(nn.ReLU())                       	
        neunet:add(nn.Linear(HUs, output))	
  return neunet			
end

neunet          = contruct_net()
neunetnll       = neunet:clone('weight', bias);
neunetlbfgs     = neunet:clone('weight', bias);

-- retrieve parameters and gradients
parameters, gradParameters = neunet:getParameters()

collectgarbage()

local function perhaps_print(q, qn, inorder, outorder, input, out, off, y_off, off_data)
  
  print('training_data', off_data)
  print('\ntesting_data', on_data)    

  --random checks to be sure data is consistent
  print('train_data_input', off_data[1]:size())  
  print('train_data_output', off_data[2])        
  print('\ntrain_xn', off_data[2][1]:size())  
  print('\ntrain_yn', off_data[2][2]:size()) 
  print('\ntrain_zn', off_data[2][3]:size())  
  print('\ntrain_roll', off_data[2][4]:size()) 
  print('\ntrain_pitch', off_data[2][5]:size())  
  print('\ntrain_yaw', off_data[2][6]:size()) 

  print('\ninput head', input[{ {1,5}, {1}} ]) 
  print('k', input:size()[1], 'off', off, '\nout\n', out, '\ttrain_output\n', y_off)
  print('\npitch head\n\n', out.zn[{ {1,5}, {1}} ])

  print('\nqn:' , qn)
  print('Optimal number of input variables is: ', torch.ceil(qn))
  print('inorder: ', inorder, 'outorder: ', outorder)
  print('system order:', inorder + outorder)

  --Print out some Lipschitz quotients (first 5) for user
  if opt.quots == 1 then
   for ii, v in pairs( q ) do
    print(ii, v)
    if ii == 5 then break end
    end
  end
  --print neural net parameters
  print('neunet biases Linear', neunet.bias)
  print('\nneunet biases\n', neunet:get(1).bias, '\tneunet weights: ', neunet:get(1).weights)
end

if (opt.print==1) then perhaps_print(q, qn, inorder, outorder, input, out, off, y_off, off_data) end

cost      = nn.MSECriterion()           -- Loss function

--training function
function train(data)
  --track the epochs
  epoch = epoch or 1
  local time = sys.clock()

  --do one epoch
  print('<trainer> on training set: ')
  print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
  for t = 1,data:size(),opt.batchSize do
     -- create mini batch
     local inputs = torch.Tensor(opt.batchSize,1,geometry[1],geometry[2])
     local targets = torch.Tensor(opt.batchSize)
     local k = 1
     for i = t,math.min(t+opt.batchSize-1,dataset:size()) do
        -- load new sample
        local sample = dataset[i]
        local input = sample[1]:clone()
        local _,target = sample[2]:clone():max(1)
        target = target:squeeze()
        inputs[k] = input
        targets[k] = target
        k = k + 1
     end

    --create closure to evaluate f(x)
    local feval = function(x)
      collectgarbage()

      --retrieve new params
      if x~=parameters then
        parameters:copy(x)
      end

      --reset grads
      gradParameters:zero()

      --eval function for complete mini-batch
      local outputs   = model:forward(inputs)
      local f         = cost:forward(outputs, targets)

      --find df/dw
      local df_do     = cost:backward(outouts, targets)
      model:backward(inputs, df_do)

      --L1 and L2 regularization (penalties )
      if opt.coefL1  ~= 0 or opt.coefL2 ~=0 then
        --locals
        local norm, sign = torch.norm, torch.sign

        --Loss:
        f = f + opt.coefL1 * norm(parameters, 1)
        f = f + opt.coefL2 * norm(parameters, 2)^2/2

        --Gradients
        gradParameters:add( sign(parameters):mul(opt.coefL1) + parametsrs:clone():mul(opt.coefL2))
      end

      --update confusion
      for i = 1, opt.batchSize do
        confusion:add(outputs[i], targets[i])
      end

      --retrun f and df/dx
      return f, gradParameters
    end

    local state = nil
    local config = nil

    --[[Declare states for limited BFGS
     See: https://github.com/torch/optim/blob/master/lbfgs.lua]]
    if opt.optimizer == 'l-bfgs' then
      print('Running optimization with L-BFGS')
      state = 
      {
        maxIter = opt.maxIter,  --Maximum number of iterations allowed
        verbose=true,
        maxEval = 1000,      --Maximum number of function evaluations
        tolFun  = 1e-1,      --Termination tol on progress in terms of func/param changes
      }

      config =   
      {
        nCorrection = opt.Correction,    
        lineSearch  = optim.lswolfe,
        lineSearchOpts = {},
        learningRate = opt.learningRate
      }
      for i_l = 0, opt.maxIter do
        local losses = optim.lbfgs(feval, parameters, config, state)
        i_l = i_l + 1
        if losses > 150 then learningRate = opt.learningRate
        elseif losses <= 150 then learningRate = opt.learningRateDecay end
        print('lbfgs iter', i_l,  'error', losses)
      end
      -- disp report:
      print('LBFGS step')
      print(' - progress in batch: ' .. t .. '/' .. data:size())
      print(' - nb of iterations: ' .. state.nIter)
      print(' - nb of function evalutions: ' .. state.funcEval)

    elseif opt.optimization == 'SGD' then

      -- Perform SGD step:
      sgdState = sgdState or {
        learningRate = opt.learningRate,
        momentum = opt.momentum,
        learningRateDecay = 5e-7
      }
      optim.sgd(feval, parameters, sgdState)
      
      -- disp progress
      xlua.progress(t, data:size())


    elseif opt.optimizer == 'adam' then
      state = {
        learningRate = opt.learningRate
      }

    elseif opt.optimizer == 'nll' then
      state = {
        learningRate = opt.learningRate
      }
    print('Running optimization with negative log likelihood criterion')
      for i_nll = 0, opt.maxIter do
        delta = optim_.nllOptim(neunetnll, u_off, y_off[3], opt.learningRate)
        i_nll  = i_nll  + 1
        if delta > 150 then learningRate = opt.learningRate
        elseif delta <= 150 then learningRate = opt.learningRateDecay end
        print('nll iter', i_nll , 'bkwd error', t, 'fwd error', delta )
      end
      

    elseif opt.optimizer == 'mse' then
      state = {
        learningRate = opt.learningRate
      }
      print('Running optimization with mean-squared error')
      local i_mse = {}
      for i_mse = 0, opt.maxIter do
        pred, mse_error = optim_.msetrain(neunet, parameters, targets[3], opt.learningRate)
          --  pred, mse_error = optim_.msetrain(neunet, u_off, y_off[3], opt.learningRate)
        if mse_error > 150 then learningRate = opt.learningRate
        elseif mse_error <= 150 then learningRate = opt.learningRateDecay end
        i_mse = i_mse + 1   
        print('MSE iteration', i_mse, '\tMSE error: ', mse_error)
          --'\tPrediction', pred, 
          -- print('neunet gradient weights', neunet.gradWeight)
          -- print('neunet gradient biases', neunet.gradBias)
      end

    else  error(string.format('Unrecognized optimizer "%s"', opt.optimizer))  end
  end
end

while true do
  train(off_data)
end