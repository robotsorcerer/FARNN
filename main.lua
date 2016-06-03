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

--[[modified native Torch Linear class to allow random weight initializations]]
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
            self.weight:normal(0.5,1)
            self.bias:normal(0.5,1)
        end
    end
end

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
local cmd = torch.CmdLine()
cmd:text()
cmd:text('===========================================================================')
cmd:text('         Identification and Control of Nonlinear Systems Using Deep        ')
cmd:text('                      Neural Networks                                      ')
cmd:text(                                                                             )
cmd:text('             Olalekan Ogunmolu. March 2016                                 ')
cmd:text(                                                                             )
cmd:text('Code by Olalekan Ogunmolu: lexilighty [at] gmail [dot] com')
cmd:text('===========================================================================')
cmd:text(                                                                             )
cmd:text(                                                                             )
cmd:text('Options')
cmd:option('-seed', 123, 'initial seed for random number generator')
cmd:option('-silent', false, 'false|true: 0 for false, 1 for true')

-- Model Order Determination Parameters
cmd:option('-data','data/soft_robot.mat','path to preprocessed data(save in Matlab -v7.3 format)')
cmd:option('-tau', 1, 'what is the delay in the data?')
cmd:option('-m_eps', 0.01, 'stopping criterion for output order determination')
cmd:option('-l_eps', 0.05, 'stopping criterion for input order determination')
cmd:option('-trainStop', 0.5, 'stopping criterion for neural net training')
cmd:option('-sigma', 0.01, 'initialize weights with this std. dev from a normally distributed Gaussian distribution')

--Gpu settings
cmd:option('-gpu', 0, 'which gpu to use. -1 = use CPU; >=0 use gpu')
cmd:option('-backend', 'cudnn', 'nn|cudnn')

-- Neural Network settings
cmd:option('-learningRate',1e-1, 'learning rate for the neural network')
cmd:option('-learningRateDecay',1e-6, 'learning rate decay to bring us to desired minimum in style')
cmd:option('-momentum', 0, 'momentum for sgd algorithm')
cmd:option('-model', 'mlp', 'mlp|lstm|linear|rnn|bnlstm')
cmd:option('-netdir', 'network', 'directory to save the network')
cmd:option('-optimizer', 'mse', 'mse|sgd')
cmd:option('-coefL1',   0, 'L1 penalty on the weights')
cmd:option('-coefL2',  0, 'L2 penalty on the weights')
cmd:option('-plot', false, 'true|false')
cmd:option('-maxIter', 10000, 'max. number of iterations; must be a multiple of batchSize')

-- RNN/LSTM Settings 
cmd:option('-rho', 5, 'length of sequence to go back in time')
cmd:option('--dropout', true, 'apply dropout with this probability after each rnn layer. dropout <= 0 disables it.')
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
opt = cmd:parse(arg or {})
torch.manualSeed(opt.seed)

torch.setnumthreads(8)

opt.rundir = cmd:string('experiment', opt, {dir=true})
paths.mkdir(opt.rundir)

-- create log file
cmd:log(opt.rundir .. '/log', opt)
cmd:addTime('Deep Head Motion Control', '%F %T')
cmd:text()

if not opt.silent then
   print(opt)
end
-------------------------------------------------------------------------------
-- Fundamental initializations
-------------------------------------------------------------------------------
--torch.setdefaulttensortype('torch.FloatTensor')            -- for CPU
print("")
print('==> fundamental initializations')

data        = opt.data
use_cuda = false
if opt.gpu >= 0 then
  require 'cutorch'
  require 'cunn'
  if opt.backend == 'cudnn' then require 'cudnn' end
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.gpu + 1)                         -- +1 because lua is 1-indexed
  idx       = cutorch.getDevice()
  print('\nSystem has', cutorch.getDeviceCount(), 'gpu(s).', 'Code is running on GPU:', idx)
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

----------------------------------------------------------------------------------------
-- Parsing Raw Data
----------------------------------------------------------------------------------------
print '==> Parsing raw data'

if(string.find(data, 'soft_robot.mat')) then
  data = matio.load(data)
  data = data.pose
  input       = data[{{}, {1}}]     --SIMO System
  -- print(input)
  out         = { 
                  data[{{}, {2}}],       --x
                  data[{{}, {3}}],       --y
                  0.1* data[{{}, {4}}],  --z
                  data[{{}, {5}}],       --roll
                  data[{{}, {6}}],       --pitch
                  data[{{}, {7}}]       --yaw
                }
  k           = input:size(1)    
  off         = torch.ceil( torch.abs(0.6*k))

  train_input = input[{{1, off}, {1}}]   -- order must be preserved. cuda tensor does not support csub yet
  train_out   = {
                 out[1][{{1, off}, {1}}], out[2][{{1, off}, {1}}],            -- most of the work is done here              (out[{{1, off}, {1}}])/10, outlln[{{1, off}, {1}}], 
                 out[3][{{1, off}, {1}}], out[4][{{1, off}, {1}}],
                 out[5][{{1, off}, {1}}], out[6][{{1, off}, {1}}],
                } 
  --create testing data
  test_input = input[{{off + 1, k}, {1}}]
  test_out   = {
                 out[1][{{off+1, k}, {1}}], out[2][{{off+1, k}, {1}}], 
                 out[3][{{off+1, k}, {1}}], out[4][{{off+1, k}, {1}}], 
                 out[5][{{off+1, k}, {1}}], out[6][{{off+1, k}, {1}}] 
                }  
end

print '==> Data Pre-processing'              
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
--qn  = computeqn(train_input, train_out[3])

--compute actual system order
--utils = require 'order.utils'
--inorder, outorder, q =  computeq(train_input, (train_out[3])/10, opt)

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
          neunet:add(nn.Linear(nhiddens, 6)) 
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

    --neunet:remember('eval') --used by Sequencer modules only
    --output layer 
    neunet = nn.Repeater(neunet, noutputs)
--===========================================================================================
    --Nested BN_LSTM Recurrence
  elseif opt.model == 'fastlstm' then   
    require 'rnn'
    -- opt.hiddenSize = loadstring(" return "..opt.hiddenSize)()
    nn.FastLSTM.usenngraph = true -- faster
    nn.FastLSTM.bn = true
    local crit = nn.MSECriterion()
    cost = nn.SequencerCriterion(crit)
    neunet = nn.Sequential()
    local inputSize = opt.hiddenSize[1]
    for i, inputSize in ipairs(opt.hiddenSize) do 
      local rnn = nn.FastLSTM(ninputs, opt.hiddenSize[1], opt.rho)
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
  else    
      error('you have specified an incorrect model. model must be <lstm> or <mlp> or <rnn>')    
  end

  return cost, neunet     
end

cost, neunet          = contruct_net()

print('Network Table\n'); print(neunet)

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
    -- time taken for one epoch
    time = sys.clock() - time
    time = time / height
    print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

    if epoch % 10 == 0 then
      saveNet()
    end
    -- next epoch
    epoch = epoch + 1

  elseif opt.model == 'lstm' or 'fastlstm' then
    train_lstm(opt)
    -- time taken for one epoch
    time = sys.clock() - time
    time = time / height
    print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')
    saveNet()

    if epoch % 10 == 0 then
      saveNet()
    end
    -- next epoch
    epoch = epoch + 1

  elseif  opt.model == 'mlp'  then
    train_mlp(opt)
    -- time taken for one epoch
    time = sys.clock() - time
    time = time / height
    print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

    if epoch % 10 == 0 then
      saveNet()
    end
    -- next epoch
    epoch = epoch + 1
  else
    print("Incorrect model entered")
  end   

end     


--test function
local function test(data)
   -- local vars
   local time = sys.clock()
   local testHeight = test_input:size(1)
   -- averaged param use?
   if average then
      cachedparams = parameters:clone()
      parameters:copy(average)
   end
   -- test over given dataset
   print('<trainer> on testing Set:')
   local avg = 0; local predF, normedPreds = {}, {}
   local iter = 1;
   for t = 1, math.min(opt.maxIter, testHeight), opt.batchSize do
      -- disp progress
      xlua.progress(t, math.min(opt.maxIter, testHeight))
      -- create mini batch        
      local inputs, targets, offsets = {}, {}, {}
      -- load new sample
      offsets = torch.LongTensor(opt.batchSize):random(1, testHeight) 
      inputs = test_input:index(1, offsets)
      --batch of targets
      targets = {
      test_out[1]:index(1, offsets), test_out[2]:index(1, offsets), 
                  test_out[3]:index(1, offsets), test_out[4]:index(1, offsets), 
                  test_out[5]:index(1, offsets), test_out[6]:index(1, offsets)
                }  

      --pre-whiten the inputs and outputs in the mini-batch
      inputs = batchNorm(inputs)
      targets = batchNorm(targets)  
    
      -- test samples
      local preds = neunet:forward(inputs)
      for i=1,#preds do
        predF[i] = preds[i]:float()        
        normedPreds[i] = torch.norm(predF[i])
        avg = normedPreds[i] + avg
      end

      -- timing
      time = sys.clock() - time
      time = time / height

      if  (iter*opt.batchSize >= math.min(opt.maxIter, height))  then 
        print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')  
        print("avg. prediction errors on test data", avg/#normedPreds)
      end 
      iter = iter + 1
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
  elseif opt.model == 'fastlstm' then
    netname = 'fastlstm-net.t7'
  else
    netname = 'neunet.t7'
  end  
  
  local filename = paths.concat(opt.netdir, netname)
  if epoch == 0 then
    if paths.filep(filename) then
     os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
    end    
    os.execute('mkdir -p ' .. sys.dirname(filename))
  else    
    print('<trainer> saving network model to '..filename)
    torch.save(filename, neunet)
  end
end

if (opt.print) then perhaps_print(q, qn, inorder, outorder, input, out, off, train_out, trainData) end

local function main()
  while true do
    train(trainData)
    test(testData)
  end
end


main()