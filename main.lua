--[[Source Code that implements the the learning controller described in my IEEE T-Ro Journal:
   Learning deep neural network policies for head motion control in maskless cancer RT
   Olalekan Ogunmolu. IEEE International Conference on Robotics and Automation (ICRA), 2017

   Author: Olalekan Ogunmolu, December 2015 - May 2016
   MIT License
   ]]

-- needed dependencies
require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'order.order_det'   
matio     = require 'matio' 
plt = require 'gnuplot' 
require 'utils.utils'
require 'utils.train'
require 'xlua'
require 'utils.stats'

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
cmd:option('-learningRateDecay',1e-6, 'learning rate decay to bring us to desired minimum in style')
cmd:option('-momentum', 0, 'momentum for sgd algorithm')
cmd:option('-model', 'mlp', 'mlp|lstm|linear|rnn')
cmd:option('-netdir', 'network', 'directory to save the network')
cmd:option('-optimizer', 'mse', 'mse|sgd')
cmd:option('-coefL1',   0, 'L1 penalty on the weights')
cmd:option('-coefL2',  0, 'L2 penalty on the weights')
cmd:option('-plot', false, 'true|false')
cmd:option('-maxIter', 10000, 'max. number of iterations; must be a multiple of batchSize')

-- RNN/LSTM Settings 
cmd:option('-rho', 5, 'length of sequence to go back in time')
cmd:option('--dropout', false, 'apply dropout with this probability after each rnn layer. dropout <= 0 disables it.')
cmd:option('--dropoutProb', 0.5, 'probability of zeroing a neuron (dropout probability)')
cmd:option('-rnnlearningRate',0.1, 'learning rate for the reurrent neural network')
cmd:option('-hiddenSize', {1, 10, 100}, 'number of hidden units used at output of each recurrent layer. When more than one is specified, RNN/LSTMs/GRUs are stacked')


-- LBFGS Settings
cmd:option('-Correction', 60, 'number of corrections for line search. Max is 100')
cmd:option('-batchSize', 6, 'Batch Size for mini-batch training, \
                            preferrably in multiples of six')

-- Print options
cmd:option('-print', false, 'false = 0 | true = 1 : Option to make code print neural net parameters')  -- print System order/Lipschitz parameters

-- misc
opt = cmd:parse(arg)
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

function transfer_data(x)
  if use_cuda then
    return x:cuda()
  else
    return x:double()
  end
end

local function pprint(x)  print(tostring(x)); print(x); end

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
off         = torch.ceil( torch.abs(0.6*k))

print '==> Data Pre-processing'          
--Determine training data    
train_input = input[{{1, off}, {1}}]   -- order must be preserved. cuda tensor does not support csub yet
train_out   = {
               out.xn[{{1, off}, {1}}], out.yn[{{1, off}, {1}}],            -- most of the work is done here.
               (out.zn[{{1, off}, {1}}])/10, out.rolln[{{1, off}, {1}}], 
               out.pitchn[{{1, off}, {1}}], out.yawn[{{1, off}, {1}}] 
              } 

--create testing data
test_input = input[{{off + 1, k}, {1}}]
test_out   = {
               out.xn[{{off+1, k}, {1}}], out.yn[{{off+1, k}, {1}}], 
               (out.zn[{{off+1, k}, {1}}])/10, out.rolln[{{off+1, k}, {1}}], 
               out.pitchn[{{off+1, k}, {1}}], out.yawn[{{off+1, k}, {1}}] 
              }  

--preprocess data
test_input  = stats.inputnorm(test_input)
test_out    = stats.normalize(test_out)
train_input = stats.inputnorm(train_input)
train_out   = stats.normalize(train_out)
--===========================================================================================
--ship train and test data to gpu
train_input = transfer_data(train_input)
test_input = transfer_data(test_input)

for i = 1, #train_out do
  train_out[i] = transfer_data(train_out[i])
end

for i=1,#test_out do
   test_out[i] = transfer_data(test_out[i])
end
                              
kk          = train_input:size(1)
--===========================================================================================           
--geometry of input
geometry    = {train_input:size(1), train_input:size(2)}

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
local width       = train_input:size(2)
height      = train_input:size(1)
local ninputs     = 1
local noutputs    = 6
local nhiddens_rnn = 6 

--number of hidden layers (for mlp network)
local nhiddens    = 1
local transfer    =  nn.ReLU()   --

--[[Set up the network, add layers in place as we add more abstraction]]
local function contruct_net()
  if opt.model  == 'mlp' then
          neunet          = nn.Sequential()
          neunet:add(nn.Linear(ninputs, nhiddens))
          neunet:add(transfer)                         
          neunet:add(nn.Linear(nhiddens, noutputs)) 

  cost      = nn.MSECriterion() 

  elseif opt.model == 'rnn' then    
-------------------------------------------------------
--  Recurrent Neural Net Initializations 
    require 'rnn'
    cost    = nn.SequencerCriterion(nn.MSECriterion())
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
    require 'nngraph'
    -- opt.hiddenSize = loadstring(" return "..opt.hiddenSize)()
    nn.LSTM.usenngraph = true -- faster
    --cost = nn.SequencerCriterion(nn.DistKLDivCriterion())
    local crit = nn.MSECriterion()
    cost = nn.SequencerCriterion(crit)
    neunet = nn.Sequential()
    local inputSize = opt.hiddenSize[1]
    for i, inputSize in ipairs(opt.hiddenSize) do 
      local rnn = nn.LSTM(ninputs, opt.hiddenSize[1], opt.rho)
      neunet:add(rnn) 
       
      if opt.dropout then
        neunet:insert(nn.Dropout(opt.dropoutProb), 1)
      end
       inputSize = opt.hiddenSize[1]
    end

    -- output layer
    neunet:add(nn.Linear(ninputs, 1))
    --neunet:add(nn.ReLU())
    neunet:add(nn.SoftSign())

    -- will recurse a single continuous sequence
    neunet:remember('eval')
    --output layer 
    neunet = nn.Repeater(neunet, noutputs)
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
      error('you have specified an incorrect model. model must be <lstm> or <mlp> or <rnn>')    
  end

  return cost, neunet     
end

cost, neunet          = contruct_net()

print('Network Table'); print(neunet)

-- retrieve parameters and gradients
parameters, gradParameters = neunet:getParameters()
--=====================================================================================================
neunet = transfer_data(neunet)  --neunet = cudnn.convert(neunet, cudnn)
cost = transfer_data(cost)
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

 else  
   error(string.format('Unrecognized optimizer "%s"', opt.optimizer))  
 end
----------------------------------------------------------------------

print '==> defining training procedure'

local function train(data)  

  --time we started training
  local time = sys.clock()

  --track the epochs
  epoch = epoch or 1

  --do one epoch
  print('<trainer> on training set: ')
  print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']\n')

  if opt.model == 'rnn' then 
    train_rnn(opt) 
  elseif opt.model == 'lstm' then
    train_lstm(opt)
  elseif  opt.model == 'mlp'  then
    train_mlp(opt)
  end   

  -- time taken for one epoch
  time = sys.clock() - time
  time = time / height
  print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

  saveNet()

  -- next epoch
  epoch = epoch + 1
end     


--test function
local function test(data)
   -- local vars
   local time = sys.clock()

   -- averaged param use?
   if average then
      cachedparams = parameters:clone()
      parameters:copy(average)
   end

   -- test over given dataset
   print('<trainer> on testing Set:')
   for t = 1, math.max(opt.maxIter, height), opt.batchSize do
      -- disp progress
      xlua.progress(t, math.min(opt.maxIter, height))

    -- create mini batch
    local inputs = {}
    local targets = {}
    for i = t,math.min(t+opt.batchSize-1,height) do
      -- load new sample
      if i > height or opt.maxIter then  i = 1  end
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

function saveNet()
  -- save/log current net
  if opt.model == 'rnn' then
    netname = 'rnn-net.t7'
  elseif opt.model == 'mlp' then
    netname = 'mlp-net.t7'
  elseif opt.model == 'lstm' then
    netname = 'lstm-net.t7'
  else
    netname = 'neunet.t7'
  end  
  
  local filename = paths.concat(opt.netdir, netname)
  os.execute('mkdir -p ' .. sys.dirname(filename))
  print('<trainer> saving network model to '..filename)
  torch.save(filename, neunet)
end

if (opt.print) then perhaps_print(q, qn, inorder, outorder, input, out, off, train_out, trainData) end

local function main()
  while true do
    train(trainData)
    test(testData)
  end
end


main()