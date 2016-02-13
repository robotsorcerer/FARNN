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
require 'order.order_det'   
matio   	= require 'matio'  
--optim_    = 
require 'optima.optim_'  

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
cmd:text('===========================================================================')
cmd:text('          A Convoluted Dynamic Neural Network for System Identification    ')
cmd:text(                                                                             )
cmd:text('             Olalekan Ogunmolu. March 2016                                 ')
cmd:text(                                                                             )
cmd:text('Code by Olalekan Ogunmolu: FirstName [dot] LastName _at_ utdallas [dot] edu')
cmd:text('===========================================================================')
cmd:text(                                                                             )
cmd:text(                                                                             )
cmd:text('Options')
cmd:option('-seed', 123, 'initial random seed to use')
cmd:option('-rundir', 0, 'false|true: 0 for false, 1 for true')

-- Model Order Determination Parameters
cmd:option('-pose','data/posemat5.mat','path to preprocessed data(save in Matlab -v7.3 format)')
cmd:option('-tau', 1, 'what is the delay in the data?')
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
cmd:option('-momentum', 0, 'momentum for sgd algorithm')

cmd:option('-optimizer', 'mse', 'mse|l-bfgs|adam')

cmd:option('-model', 'mlp', 'mlp|convnet|linear')
cmd:option('-visualize', true, 'visualize input data and weights during training')
cmd:option('-optimizer', 'mse', 'mse|l-bfgs|adam|sgd')

cmd:option('-coefL1',   0, 'L1 penalty on the weights')
cmd:option('-coefL2',  0, 'L2 penalty on the weights')
cmd:option('-plot', false, 'plot while training')

-- LBFGS Settings
cmd:option('-Correction', 60, 'number of corrections for line search. Max is 100')
cmd:option('-batchSize', 6, 'Batch Size for mini-batch training, \
                            preferrably in multiples of six')

-- Print options
cmd:option('-print', false, 'false = 0 | true = 1 : Option to make code print neural net parameters')  -- print System order/Lipschitz parameters

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
print('==> fundamental initializations')

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
print '==> Parsing raw data'

input       = matio.load(data, 'in')						--SIMO System
out         = matio.load(data, {'xn', 'yn', 'zn', 'rolln', 'pitchn',  'yawn' })

y           = {out.xn, out.yn, 
               out.zn/10, out.rolln, 
               out.pitchn, out.yawn}

k           = input:size()[1]
--Determine training data               
off         = torch.ceil( torch.abs(0.6*k))
train_input = input[{{1, off}, {1}}]     
train_out   = {
               out.xn[{{1, off}, {1}}], out.yn[{{1, off}, {1}}], 
               (out.zn[{{1, off}, {1}}])/10, out.rolln[{{1, off}, {1}}], 
               out.pitchn[{{1, off}, {1}}], out.yawn[{{1, off}, {1}}] 
              }

--create testing data
test_input      = input[{{off + 1, k}, {1}}]  
test_out        = {
               out.xn[{{off+1, k}, {1}}], out.yn[{{off+1, k}, {1}}], 
               (out.zn[{{off+1, k}, {1}}])/10, out.rolln[{{off+1, k}, {1}}], 
               out.pitchn[{{off+1, k}, {1}}], out.yawn[{{off+1, k}, {1}}] 
              }              

kk          = train_input:size()[1]

--geometry of input
geometry    = {kk, train_input:size()[2]}

trainData     = {train_input, train_out}
testData     = {test_input,  test_out}
--===========================================================================================
--[[Determine input-output order using He and Asada's prerogative
    See Code order_det.lua in folder "order"]]
print '==> Determining input-output model order parameters'    

--find optimal # of input variables from data
qn  = order_det.computeqn(train_input, train_out[3])

--compute actual system order
utils = require 'order.utils'
inorder, outorder, q =  order_det.computeq(train_input, (train_out[3])/10, opt)
----------------------------------------------------------------------------------------------
--Set up each network properties
----------------------------------------------------------------------------------------------
-- dimension of my feature bank (each input is a 1D array)
nfeats      = 1   

--dimension of training input
width       = trainData[1]:size()[2]
height      = trainData[1]:size()[1]
ninputs     = 1  --nfeats * width * height
noutput     = 1

--number of hidden layers (for mlp network)
nhiddens    = 1     --you may try ninputs / 2

--hidden units, filter kernel (for ConvNet)
nstates     = {10, 10, 20}
filtsize    = 3
poolsize    = 2                   --LP norm work best with P = 2 or P = inf. This results in a reduced-resolution output feature map which is robust to small variations in the location of features in the previous layer
normkernel = image.gaussian1D(7)

--[[Set up the network, add layers in place as we add more abstraction]]
function contruct_net()
  if opt.model  == 'mlp' then
          neunet        	= nn.Sequential()
          neunet:add(nn.Linear(ninputs, nhiddens))
          neunet:add(nn.ReLU())                       	
          neunet:add(nn.Linear(nhiddens, noutput))	
  elseif opt.model == 'convnet' then

    if opt.backend == 'cudnn' then
      --typical convnet (convolution + relu + pool)
      neunet  = nn.Sequential()

      --stage 1: filter bank -> squashing - L2 pooling - > normalization
      --[[The first layer applies 10 filters to the input map choosing randomly
      among its different layers ech being a 3x3 kernel. The receptive field of the 
      first layer is 3x3 and the maps produced are therefore]]
      neunet:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize, filtsize))
      neunet:add(nn.ReLU())
      neunet:add(nn.SpatialMaxPooling(poolsize, poolsize, poolsize, poolsize))

      -- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
      neunet:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize, filtsize))
      neunet:add(nn.ReLU())
      neunet:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))

      -- stage 3 : standard 2-layer neural network
      neunet:add(nn.View(nstates[2]*filtsize*filtsize))
      neunet:add(nn.Dropout(0.5))
      neunet:add(nn.Linear(nstates[2]*filtsize*filtsize, nstates[3]))
      neunet:add(nn.ReLU())
      neunet:add(nn.Linear(nstates[3], noutputs))

    else

      -- a typical convolutional network, with locally-normalized hidden
      -- units, and L2-pooling

      -- Note: the architecture of this convnet is loosely based on Pierre Sermanet's
      -- work on this dataset (http://arxiv.org/abs/1204.3968). In particular
      -- the use of LP-pooling (with P=2) has a very positive impact on
      -- generalization. Normalization is not done exactly as proposed in
      -- the paper, and low-level (first layer) features are not fed to
      -- the classifier.  

      neunet    = nn.Sequential()        

      --stage 1: filter bank -> squashing -> L2 pooling -> normalization
      neunet:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize, filtsize))
      neunet:add(nn.Tanh())
      neunet:add(nn.SpatialLPooling(nStates[1], 2, poolsize, poolsize, poolsize, poolsize))
      neunet:add(nn.SpatialSubtractiveNormalization(nstates[1], normkernel))

      -- stage 2: filter bank -> squashing -> L2 poolong - > normalization
      neunet:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize, filtsize))
      neunet:add(nn.Tanh())
      neunet:add(nn.SpatialLPooling(nstates[2], 2, poolsize, poolsize, poolsize, poolsize))
      neunet:add(nn.SpatialSubtractiveNormalization(nstates[2], normkernel))

      -- stage 3: standard 2-layer neural network
      neunet:add(nn.Reshape(nstates[2] * filtsize * filtsize))
      neunet:add(nn.Linear(nstates[2] * filtsize * filtsize, nstates[3]))
      neunet:add(nn.Tanh())
      neunet:add(nn.Linear(nstates[3], noutputs))
    end
    print('neunet biases Linear', neunet.bias)
    print('\nneunet biases\n', neunet:get(1).bias, '\tneunet weights: ', neunet:get(1).weights)
  else
    
      error('you have specified an unknown model')
    
  end

  return neunet			
end

neunet          = contruct_net()
neunety         = neunet:clone(weight, bias);
neunetz         = neunet:clone(weight, bias);
neunetr         = neunet:clone(weight, bias);
neunetp         = neunet:clone(weight, bias);
neunetyw         = neunet:clone(weight, bias);
--===================================================================================
-- Visualization is quite easy, using itorch.image().
--===================================================================================

if opt.visualize then
   if opt.model == 'convnet' then
      if itorch then
   print '==> visualizing ConvNet filters'
   print('Layer 1 filters:')
   itorch.image(neunet:get(1).weight)
   print('Layer 2 filters:')
   itorch.image(neunet:get(5).weight)
      else
   print '==> To visualize filters, start the script in itorch notebook'
      end
   end
end

-- retrieve parameters and gradients
parameters, gradParameters = neunet:getParameters()

--=====================================================================================================

cost      = nn.MSECriterion()           -- Loss function

--training function
function train(data)
  --track the epochs
  epoch = epoch or 1
  local time = sys.clock()

  --do one epoch
  print('<trainer> on training set: ')
  print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
  
  local targets_X = {} local targets_Y = {} local targets_Z = {}
  local targets_R = {} local targets_P = {} local targets_YW = {}

  for t = 1, data[1]:size()[1], opt.batchSize do
     -- create mini batch

     --local inputs = torch.Tensor(opt.batchSize,1,geometry[1],geometry[2])
     local inputs = torch.Tensor(opt.batchSize,1,data[1]:size()[1],data[1]:size()[2])
     --print('inputs[k]', inputs:size())
     local targets = torch.Tensor(opt.batchSize)
     local k = 1
     for i = t,math.min(t+opt.batchSize-1,data[1]:size()[1]) do
        -- load new sample
        local sample = {data[1], data[2][1], data[2][2], (data[2][3])/10, data[2][4], data[2][5], data[2][6]}       --use pitch 1st; we are dividing pitch values by 10 because it was incorrectly loaded from vicon
        --for ii, kk in ipairs(sample) do print(ii, kk) end
        local input = sample[1]:clone()
        local _,target = sample[4]:clone():max(1)
        target = target:squeeze()
        inputs[k] = input
        targets[k] = target
        k = k + 1
     end
     print(targets)
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
      local outputs   = neunetlbfgs:forward(input)
      local f         = cost:forward(outputs, targets)

      --find df/dw
      local df_do     = cost:backward(outouts, targets)
      neunetlbfgs:backward(inputs, df_do)

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

    parameters = data[1]

    local inputs = {}
    local targets = {}
    for i = t,math.min(t+opt.batchSize-1,data[1]:size()[1]) do
      -- load new sample
      local sample = {data[1], data[2][1], data[2][2], data[2][3], data[2][4], data[2][5], data[2][6]}       --use pitch 1st; we are dividing pitch values by 10 because it was incorrectly loaded from vicon
      local input = sample[1]:clone()[i]
      local target = {sample[2]:clone()[i], sample[3]:clone()[i], sample[4]:clone()[i], sample[5]:clone()[i], sample[6]:clone()[i], sample[7]:clone()[i]}
      table.insert(inputs, input)
      table.insert(targets, target) 
    end
    
    targets_X   = {targets[1][1], targets[2][1], targets[3][1], targets[4][1], targets[5][1], targets[6][1]}
    targets_Y   = {targets[1][2], targets[2][2], targets[3][2], targets[4][2], targets[5][2], targets[6][2]}
    targets_Z   = {targets[1][3], targets[2][3], targets[3][3], targets[4][3], targets[5][3], targets[6][3]}
    targets_R   = {targets[1][4], targets[2][4], targets[3][4], targets[4][4], targets[5][4], targets[6][4]}
    targets_P   = {targets[1][5], targets[2][5], targets[3][5], targets[4][5], targets[5][5], targets[6][5]}
    targets_YW   = {targets[1][6], targets[2][6], targets[3][6], targets[4][6], targets[5][6], targets[6][6]}
    -- table.insert(targets_Y, targets[2]) 
    -- table.insert(targets_Z, targets[3]) 
    -- table.insert(targets_R, targets[4]) 
    -- table.insert(targets_P, targets[5])
    -- table.insert(targets_YW, targets[6])

    -- classes
    classes = target

    -- This matrix records the current confusion across classes
    --confusion = optim.ConfusionMatrix(classes)

    --print('classes', classes)
    --confusion = optim.ConfusionMatrix(classes)

    --create closure to evaluate f(x): https://github.com/torch/tutorials/blob/master/2_supervised/4_train.lua
    local feval = function(x)
                    collectgarbage()

                    --retrieve new params
                    if x~=parameters then
                      parameters:copy(x)
                    end

                    --reset grads
                    gradParameters:zero()
          
                    -- f is the average of all criterions
                    local f = 0

                    -- evaluate function for complete mini batch
                    for i_f = 2,#inputs do
                        print('#inputs', #inputs)
                        -- estimate f
                        local output = neunet:forward(inputs[i_f])
                        local err = cost:forward(output, targets[i_f])
                        f = f + err
                        print('feval error', err)

                        -- estimate df/dW
                        local df_do = cost:backward(output, targets[i_f])
                        neunet:backward(inputs[i_f], df_do)

                        -- penalties (L1 and L2):
                        if opt.coefL1 ~= 0 or opt.coefL2 ~= 0 then
                           -- locals:
                           local norm,sign= torch.norm,torch.sign

                           -- Loss:
                           f = f + opt.coefL1 * norm(parameters,1)
                           f = f + opt.coefL2 * norm(parameters,2)^2/2

                           -- Gradients:
                           gradParameters:add( sign(parameters):mul(opt.coefL1) + parameters:clone():mul(opt.coefL2) )
                        
                        else
                          -- normalize gradients and f(X)
                          gradParameters:div(#inputs)
                        end

                        -- update confusion
                        --confusion:add(output, targets[i_f])
                    end

                    -- normalize gradients and f(X)
                    gradParameters:div(#inputs)
                    f = f/#inputs

                    --retrun f and df/dx
                    return f, gradParameters
                  end

    --[[Declare states for limited BFGS
     See: https://github.com/torch/optim/blob/master/lbfgs.lua]]

     if opt.optimizer == 'mse' then
       state = {
         learningRate = opt.learningRate
       }
       --we do a SISO from input to each of the six outputs in each iteration
       --For SIMO data, it seems best to run a different network from input to output.
       print('Running optimization with mean-squared error')
           pred_x, error_x = optim_.msetrain(neunet, cost, inputs, targets_X, opt.learningRate, opt)
           pred_y, error_y = optim_.msetrain(neunety, cost, inputs, targets_Y, opt.learningRate, opt)
           pred_z, error_z = optim_.msetrain(neunetz, cost, inputs, targets_Z, opt.learningRate, opt)
           pred_r, error_r = optim_.msetrain(neunetr, cost, inputs, targets_R, opt.learningRate, opt)
           pred_p, error_p = optim_.msetrain(neunetp, cost, inputs, targets_P, opt.learningRate, opt)
           pred_yw, error_yw = optim_.msetrain(neunety, cost, inputs, targets_YW, opt.learningRate, opt)
      print('Error Table at epoch:', epoch)
      print('\t', error_x, '\n', error_y, '\n', error_z, '\n', error_r, '\n', error_p, '\n', error_yw)
      local state = nil      local config = nil      parameters = train_input

    elseif opt.optimizer == 'l-bfgs' then
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
      -- disp report:
      print('LBFGS step')
      print(' - progress in batch: ' .. t .. '/' .. data[1]:size()[1])
      print(' - nb of iterations: ' .. state.maxIter)
      print(' - nb of function evaluations: ' .. state.maxEval)

    --  for i_l = 0, opt.maxIter do

        local u, losses = optim.lbfgs(feval, parameters, config, state)

        local u, losses = optim.lbfgs(feval, inputs[t], config, state)
   --    i_l = i_l + 1
        if losses > 150 then learningRate = opt.learningRate
        elseif losses <= 150 then learningRate = opt.learningRateDecay end
        print('losses', losses, 'optimal u', u)
    --  end
 
    elseif opt.optimization == 'SGD' then
    elseif opt.optimizer == 'ASGD' then
       optimState = {
          eta0 = opt.learningRate,
          t0 = data[1]:size()[1] * 1
       }
      optim.asgd(feval, parameters, sgdState)

    elseif opt.optimizer == 'sgd' then

      -- Perform SGD step:
      sgdState = sgdState or {
        learningRate = opt.learningRate,
        momentum = opt.momentum,
        learningRateDecay = 5e-7
      }
      optim.sgd(feval, parameters, sgdState)
      
      -- disp progress
      xlua.progress(t, data:size())


    elseif opt.optimizer == 'nll' then
      state = {
        learningRate = opt.learningRate
      }
    print('Running optimization with negative log likelihood criterion')
      for i_nll = 0, opt.maxIter do
        delta = optim_.nllOptim(neunetnll, u_off, targets, opt.learningRate)
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
        pred, mse_error = optim_.msetrain(neunet, parameters, targets, opt.learningRate)
          --  pred, mse_error = optim_.msetrain(neunet, u_off, y_off[3], opt.learningRate)
        if mse_error > 150 then learningRate = opt.learningRate
        elseif mse_error <= 150 then learningRate = opt.learningRateDecay end
        i_mse = i_mse + 1   
        print('MSE iteration', i_mse, '\tMSE error: ', mse_error)
          --'\tPrediction', pred, 
          -- print('neunet gradient weights', neunet.gradWeight)
          -- print('neunet gradient biases', neunet.gradBias)
      end

    elseif opt.optimizer == 'asgd' then
       _,_,average = optimMethod(feval, parameters, optimState)
    
    else  
      error(string.format('Unrecognized optimizer "%s"', opt.optimizer))  
    end

    
    -- time taken
    time = sys.clock() - time
    print("train data size", trainData[1]:size()[1])
    time = time / trainData[1]:size()[1]
    print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

    -- print confusion matrix
    --print(confusion)

    -- update logger/plot
    --trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
    if opt.plot then
       trainLogger:style{['% mean class accuracy (train set)'] = '-'}
       trainLogger:plot()
    end

    -- save/log current net
    local filename = paths.concat('results', 'neunet.net')
    os.execute('mkdir -p ' .. sys.dirname(filename))
    print('==> saving model to '..filename)
    torch.save(filename, neunet)

    -- next epoch
    --confusion:zero()
    epoch = epoch + 1
  end
end

while true do
  train(trainData)
  train(testData)
end

--print a bunch of stuff if user enables print option
local function perhaps_print(q, qn, inorder, outorder, input, out, off, train_out, trainData)
  
  print('training_data', trainData)
  print('\ntesting_data', test_data)    

  --random checks to be sure data is consistent
  print('train_data_input', trainData[1]:size())  
  print('train_data_output', trainData[2])        
  print('\ntrain_xn', trainData[2][1]:size())  
  print('\ntrain_yn', trainData[2][2]:size()) 
  print('\ntrain_zn', trainData[2][3]:size())  
  print('\ntrain_roll', trainData[2][4]:size()) 
  print('\ntrain_pitch', trainData[2][5]:size())  
  print('\ntrain_yaw', trainData[2][6]:size()) 

  print('\ninput head', input[{ {1,5}, {1}} ]) 
  print('k', input:size()[1], 'off', off, '\nout\n', out, '\ttrain_output\n', train_out)
  print('\npitch head\n\n', out.zn[{ {1,5}, {1}} ])

  print('\nqn:' , qn)
  print('Optimal number of input variables is: ', torch.ceil(qn))
  print('inorder: ', inorder, 'outorder: ', outorder)
  print('system order:', inorder + outorder)

  --Print out some Lipschitz quotients (first 5) for user
  for ii, v in pairs( q ) do
    print('Lipschitz quotients head', ii, v)
    if ii == 5 then break end
  end
  --print neural net parameters
  print('neunet biases Linear', neunet.bias)
  print('\nneunet biases\n', neunet:get(1).bias, '\tneunet weights: ', neunet:get(1).weights)

  
  print('inputs: ', inputs, '#inputs', #inputs)
  print('targets: ', targets)
  print('targets_X', targets_X)
  print('targets_Y', targets_Y)
  print('targets_Z', targets_Z)
  print('targets_R', targets_R)
  print('targets_P', targets_P)
  print('targets_YW', targets_YW)
end

if (opt.print) then perhaps_print(q, qn, inorder, outorder, input, out, off, train_out, trainData) end
