--[[Source Code that implements the the learning controller described in my IEEE T-Ro Journal:
   Learning deep neural network policies for head motion control in maskless cancer RT
   Olalekan Ogunmolu. IEEE International Conference on Robotics and Automation (ICRA), 2017

   Author: Olalekan Ogunmolu, December 2015 - May 2016
   MIT License
   ]]

-- needed dependencies
require 'torch'
require 'nn'
require 'optim'
require 'image'
require 'order.order_det'   
matio     = require 'matio'  
require 'utils.utils'
require 'utils.train'
require 'xlua'

--[[modified native Torch Linear class to allow random weight initializations
 and avoid local minima issues ]]
do
    local Linear, parent = torch.class('nn.CustomLinear', 'nn.Linear')    
    -- override the constructor to have the additional range of initialization
    function Linear:__initutils(inputSize, outputSize, mean, std)
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
cmd:text('Learning Deep Neural Network Policies During H&N Motion Control in         ')
cmd:text('                      Clinical Cancer Radiotherapy                         ')
cmd:text(                                                                             )
cmd:text('             Olalekan Ogunmolu. March 2016                                 ')
cmd:text(                                                                             )
cmd:text('Code by Olalekan Ogunmolu: lexilighty [at] gmail [dot] com')
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
cmd:option('-rnnlearningRate',0.1, 'learning rate for the reurrent neural network')
cmd:option('-learningRateDecay',1e-6, 'learning rate decay to bring us to desired minimum in style')
cmd:option('-momentum', 0, 'momentum for sgd algorithm')
cmd:option('-model', 'mlp', 'mlp|lstm|linear|rnn')
cmd:option('-rho', 6, 'length of sequence to go back in time')
cmd:option('-netdir', 'network', 'directory to save the network')
cmd:option('-visualize', true, 'visualize input data and weights during training')
cmd:option('-optimizer', 'mse', 'mse|l-bfgs|asgd|sgd|cg')
cmd:option('-coefL1',   0, 'L1 penalty on the weights')
cmd:option('-coefL2',  0, 'L2 penalty on the weights')
cmd:option('-plot', false, 'plot while training')
cmd:option('-maxIter', 10000, 'max. number of iterations; must be a multiple of batchSize')

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

cmd:addTime('Deep Head Motion Control', '%F %T')
cmd:text()

-------------------------------------------------------------------------------
-- Fundamental initializations
-------------------------------------------------------------------------------
--torch.setdefaulttensortype('torch.FloatTensor')            -- for CPU
print('==> fundamental initializations')

data        = opt.pose 
use_cuda = false
if opt.gpu >= 0 then
  require 'cutorch'
  require 'cunn'
  if opt.backend == 'cudnn' then require 'cudnn' end
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.gpu + 1)                         -- +1 because lua is 1-indexed
  idx       = cutorch.getDevice()
  print('System has', cutorch.getDeviceCount(), 'gpu(s).', 'Code is running on GPU:', idx)
  use_cuda = true  
end

-- Log results to files
trainLogger = optim.Logger(paths.concat(opt.netdir, 'train.log'))
testLogger  = optim.Logger(paths.concat(opt.netdir, 'test.log'))
----------------------------------------------------------------------------------------
-- Parsing Raw Data
----------------------------------------------------------------------------------------
print '==> Parsing raw data'

input       = matio.load(data, 'in')            --SIMO System
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

if use_cuda then   
  train_input = train_input:cuda()
  train_out   = {train_out[1]:cuda(), train_out[2]:cuda(), train_out[3]:cuda(),
                train_out[4]:cuda(), train_out[5]:cuda(),  train_out[6]:cuda()}
  test_input  = test_input:cuda()
  test_out    = {test_out[1]:cuda(), test_out[2]:cuda(), test_out[3]:cuda(),
                  test_out[4]:cuda(), test_out[5]:cuda(), test_out[6]:cuda()}
end

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
qn  = computeqn(train_input, train_out[3])

--compute actual system order
--utils = require 'order.utils'
inorder, outorder, q =  computeq(train_input, (train_out[3])/10, opt)

--------------------utils--------------------------------------------------------------------------
print '==> Setting up neural network parameters'
----------------------------------------------------------------------------------------------
-- dimension of my feature bank (each input is a 1D array)
local nfeats      = 1   

--dimension of training input
local width       = trainData[1]:size()[2]
height      = trainData[1]:size()[1]
local ninputs     = 1
local noutputs    = 6

--number of hidden layers (for mlp network)
local nhiddens    = 1
local transfer    =  nn.ReLU()   --

--[[Set up the network, add layers in place as we add more abstraction]]
function contruct_net()
  if opt.model  == 'mlp' then
          neunet          = nn.Sequential()
          neunet:add(nn.Linear(ninputs, nhiddens))
          neunet:add(transfer)                         
          neunet:add(nn.Linear(nhiddens, noutputs)) 

  elseif opt.model == 'rnn' then    
-------------------------------------------------------
--  Recurrent Neural Net Initializations 
    require 'rnn'
    local nhiddens_rnn = 6 
------------------------------------------------------
  --[[m1 = batchSize X hiddenSize; m2 = inputSize X start]]
    --we first model the inputs to states
    local ffwd  =   nn.Sequential()
                  :add(nn.Linear(ninputs, nhiddens))
                  :add(nn.ReLU())      --very critical; changing this to tanh() or ReLU() leads to explosive gradients
                  :add(nn.Linear(nhiddens, nhiddens_rnn))


    local rho         = opt.rho                   -- the max amount of bacprop steos to take back in time
    local start       = 6                       -- the size of the output (excluding the batch dimension)        
    local rnnInput    = nn.Linear(start, nhiddens_rnn)     --the size of the output
    local feedback    = nn.Linear(nhiddens_rnn, start)           --module that feeds back prev/output to transfer module

    --then do a self-adaptive feedback of neurons 
   r = nn.Recurrent(start, 
                     rnnInput,  --input module from inputs to outs
                     feedback,
                     transfer,
                     rho             
                     )

    --we then join the feedforward with the recurrent net
    neunet     = nn.Sequential()
                  :add(ffwd)
                  :add(nn.Sigmoid())
                  :add(r)
                  :add(nn.Linear(start, 1))
                --  :add(nn.Linear(1, noutputs))

    --neunet    = nn.Sequencer(neunet)
    neunet    = nn.Repeater(neunet, noutputs)
    --======================================================================================
    --Nested LSTM Recurrence
  elseif opt.model == 'lstm' then   
    require 'rnn'
    local nIndex, hiddenSize = 10, 7 
    local rm = nn.Sequential()
              :add(nn.ParallelTable()
              :add(nn.LookupTable(nIndex, hiddenSize)) 
              :add(nn.Linear(hiddenSize, noutputs))) 
              :add(nn.CAddTable())
              :add(nn.Sigmoid())
              :add(nn.FastLSTM(ninputs, noutputs)) -- an AbstractRecurrent instance
              :add(nn.Linear(noutputs,hiddenSize))
              :add(nn.Sigmoid())  

    neunet = nn.Sequential()
       :add(nn.Recurrence(rm, hiddenSize, 0)) -- another AbstractRecurrent instance
       :add(nn.Linear(hiddenSize, nIndex))
       :add(nn.LogSoftMax())

       neunet = nn.Sequencer(neunet)


   --print('Network Table'); print(neunet)
--===========================================================================================
--Convnet
  elseif opt.model == 'convnet' then
    --hidden units, filter kernel (for Temporal ConvNet)
    local nstates     = {1, 1, 2}
    local kW          = 5           --kernel width
    local dW          = 1           --convolution step
    local poolsize    = 2                   --LP norm work best with P = 2 or P = inf. This results in a reduced-resolution output feature map which is robust to small variations in the location of features in the previous layer
    local normkernel = image.gaussian1D(7)

    if use_cuda then
      --typical convnet (convolution + relu + pool)
      neunet  = nn.Sequential()

      --stage 1: filter bank -> squashing - L2 pooling - > normalization
      --[[The first layer applies 10 filters to the input map choosing randomly
      among its different layers ech being a 3x3 kernel. The receptive field of the 
      first layer is 3x3 and the maps produced are therefore]]
      --neunet:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize, filtsize))
      neunet:add(nn.TemporalConvolution(ninputs, noutputs, kW, dW))
      neunet:add(transfer)
      neunet:add(nn.SpatialMaxPooling(poolsize, poolsize, poolsize, poolsize))

      -- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
      neunet:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize, filtsize))
      neunet:add(transfer)
      neunet:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))

      -- stage 3 : standard 2-layer neural network
      neunet:add(nn.View(nstates[2]*filtsize*filtsize))
      neunet:add(nn.Dropout(0.5))
      neunet:add(nn.Linear(nstates[2]*filtsize*filtsize, nstates[3]))
      neunet:add(transfer)
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

print('Network Table'); print(neunet)
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
if opt.model == 'rnn' then
  cost    = nn.SequencerCriterion(nn.MSECriterion())
  --cost    = nn.RepeaterCriterion(nn.MSECriterion())
else
  cost      = nn.MSECriterion()           -- Loss function
end
--======================================================================================

if use_cuda then
  neunet = neunet:cuda() 
  --neunet = cudnn.convert(neunet, cudnn)
  cost = cost:cuda()
end
print '==> configuring optimizer\n'

 --[[Declare states for limited BFGS
  See: https://github.com/torch/optim/blob/master/lbfgs.lua]]

 if opt.optimizer == 'mse' then
    state = {
     learningRate = opt.learningRate
   }
   optimMethod = msetrain

 elseif opt.optimizer == 'sgd' then      
   -- Perform SGD step:
   sgdState = sgdState or {
   learningRate = opt.learningRate,
   momentum = opt.momentum,
   learningRateDecay = 5e-7
   }
   optimMethod = optim.sgd

elseif opt.optimizer == 'asgd' then
   optimState = {
      eta0 = opt.learningRate,
      t0 = height * 1
   }
   optimMethod = optim.asgd

elseif opt.optimization == 'cg' then
   optimState = {
      maxIter = opt.maxIter
   }
   optimMethod = optim.cg

elseif opt.optimization == 'l-bfgs' then
   optimState = {
      learningRate = opt.learningRate,
      maxIter = opt.maxIter,
      nCorrection = 10
   }
   optimMethod = optim.lbfgs

 else  
   error(string.format('Unrecognized optimizer "%s"', opt.optimizer))  
 end
----------------------------------------------------------------------

print '==> defining training procedure'

function train(data)  
  if opt.model == 'rnn' then 
    train_rnn(opt)      

  elseif opt.model == 'lstm' then
    train_lstm(opt)

  elseif  opt.model == 'mlp'  then
    train_mlp(opt)

  end   
end     -- end train function


--test function
function test(data)
   -- local vars
   local time = sys.clock()

   -- averaged param use?
   if average then
      cachedparams = parameters:clone()
      parameters:copy(average)
   end

   -- test over given dataset
   print('<trainer> on testing Set:')
   for t = 1, math.min(opt.maxIter, height), opt.batchSize do
      -- disp progress
      xlua.progress(t, height)

    -- create mini batch
    local inputs = {}
    local targets = {}
    for i = t,math.min(t+opt.batchSize-1,height) do
      -- load new sample
      local sample = {data[1], data[2][1], data[2][2], data[2][3], data[2][4], data[2][5], data[2][6]}       --use pitch 1st; we are dividing pitch values by 10 because it was incorrectly loaded from vicon
      input = sample[1]:clone()[i]
      target = {sample[2]:clone()[i], sample[3]:clone()[i], sample[4]:clone()[i], sample[5]:clone()[i], sample[6]:clone()[i], sample[7]:clone()[i]}
      table.insert(inputs, input)
      table.insert(targets, target) 
    end    
    
    -- test samples
    for j = 1, #inputs do
      local preds = neunet:forward(inputs[j])
    end

    -- timing
    time = sys.clock() - time
    time = time / height
    print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')

    end
end

function saveNet(epoch, time)
  -- time taken
  time = sys.clock() - time
  time = time / height
  print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

  -- save/log current net
  if opt.model == 'rnn' then
    netname = 'rnn-net.net'
  elseif opt.model == 'mlp' then
    netname = 'mlp-net.net'
  else
    netname = 'neunet.net'
  end  
  
  local filename = paths.concat(opt.netdir, netname)
  os.execute('mkdir -p ' .. sys.dirname(filename))
  print('<trainer> saving network model to '..filename)
  torch.save(filename, neunet)

  -- next epoch
    --confusion:zero()
  epoch = epoch + 1
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
end

if (opt.print) then perhaps_print(q, qn, inorder, outorder, input, out, off, train_out, trainData) end


while true do
  train(trainData)
  test(testData)


  -- update logger/plot
  --trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
  if opt.plot then
     trainLogger:style{['% mean class accuracy (train set)'] = '-'}
     trainLogger:plot()
  end
end