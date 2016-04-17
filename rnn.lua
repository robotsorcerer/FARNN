--[[Source Code that implements the algorithm described in the paper:
   A Fully Automated Recurrent Neural Network for System Identification and Control
   Jeen-Shing Wang, and Yen-Ping Chen. IEEE Transactions on Circuits and Systems June 2006

   Author: Olalekan Ogunmolu, December 2015
   MIT License
   ]]

-- needed dependencies
require 'torch'
require 'nn'
require 'optim'
require 'image'
require 'order.order_det'   
matio     = require 'matio'  
require 'optima.optim_'  
require 'utils.utils'
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
cmd:text('A Dynamic Convoluted Neural Network for System Identification and Control  ')
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
 --data = data:cuda()
else
  opt.backend = 'nn'
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
--utils = require 'order.utils'
inorder, outorder, q =  order_det.computeq(train_input, (train_out[3])/10, opt)
--------------------utils--------------------------------------------------------------------------
print '==> Setting up neural network parameters'
----------------------------------------------------------------------------------------------
-- dimension of my feature bank (each input is a 1D array)
nfeats      = 1   

--dimension of training input
width       = trainData[1]:size()[2]
height      = trainData[1]:size()[1]
ninputs     = 1
noutputs    = 6

--number of hidden layers (for mlp network)
nhiddens    = 1
transfer    =  nn.ReLU()   --

--hidden units, filter kernel (for Temporal ConvNet)
nstates     = {1, 1, 2}
kW          = 5           --kernel width
dW          = 1           --convolution step
poolsize    = 2                   --LP norm work best with P = 2 or P = inf. This results in a reduced-resolution output feature map which is robust to small variations in the location of features in the previous layer
normkernel = image.gaussian1D(7)

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
    nhiddens_rnn = 6 
------------------------------------------------------
  --[[m1 = batchSize X hiddenSize; m2 = inputSize X start]]
    --we first model the inputs to states
    ffwd       =   nn.Sequential()
                  :add(nn.Linear(ninputs, nhiddens))
                  :add(nn.ReLU())      --very critical; changing this to tanh() or ReLU() leads to explosive gradients
                  :add(nn.Linear(nhiddens, nhiddens_rnn))


    rho         = opt.rho                   -- the max amount of bacprop steos to take back in time
    start       = 6                       -- the size of the output (excluding the batch dimension)        
    rnnInput    = nn.Linear(start, nhiddens_rnn)     --the size of the output
    feedback    = nn.Linear(nhiddens_rnn, start)           --module that feeds back prev/output to transfer module

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
    print('rnn')
    print(neunet)
    --======================================================================================

  elseif opt.model == 'convnet' then

    if opt.backend == 'cudnn' then
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
      t0 = trainData[1]:size()[1] * 1
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
    --track the epochs
    epoch = epoch or 1
    --time we started training
    local time = sys.clock()

    --do one epoch
    print('<trainer> on training set: ')
    print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']\n')

    offsets = {}
    --form mini batch    
    local iter = 1
    local target, input = {}, {}                               

    for t = 1, opt.maxIter, opt.batchSize do --1, train_input:size(1), opt.batchSize do
      offsets = torch.LongTensor(opt.batchSize):random(1,train_input:size(1))    
      xlua.progress(t, train_input:size(1))
       print('\n')
       -- 1. create a sequence of rho time-steps
      local inputs, targets = {}, {}

      inputs = train_input:index(1, offsets)
      targets = {train_out[1]:index(1, offsets), train_out[2]:index(1, offsets), 
                        train_out[3]:index(1, offsets), train_out[4]:index(1, offsets), 
                        train_out[5]:index(1, offsets), train_out[6]:index(1, offsets)}
      --increase offsets indices by 1      
      offsets:add(1) -- increase indices by 1
      offsets[offsets:gt(train_input:size(1))] = 1

      offsets = torch.LongTensor():resize(offsets:size()[1]):copy(offsets)

      print('targets'); print(targets)
      
      --2. Forward sequence through rnn
      neunet:zeroGradParameters()
      neunet:forget()  --forget all past time steps

      inputs_, outputs = {}, {}
      local loss = 0
       outputs = neunet:forward(inputs)
       print('inputs', inputs)  
       print('outputs'); print(outputs)
       loss    = loss + cost:forward(outputs, targets)
      print(string.format("Step %d, Loss = %f ", iter, loss))
            
      --3. do backward propagation through time(Werbos, 1990, Rummelhart, 1986)
      local  gradOutputs = cost:backward(outputs, targets)
      local gradInputs  = neunet:backward(inputs, gradOutputs) 
        
        print('gradoutputs'); --print(gradOutputs)
        for i,v in ipairs(gradOutputs) do
          print(i,v)
        end

        print('gradInputs'); print(gradInputs)

      --4. update lr
      neunet:updateParameters(opt.rnnlearningRate)

      iter = iter + 1 

      saveNet(epoch, time)
    end    
  elseif  opt.model == 'mlp'  then
    --track the epochs
    epoch = epoch or 1
    --time we started training
    local time = sys.clock()

    --do one epoch
    print('<trainer> on training set: ')
    print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']\n')

    for t = 1, opt.maxIter, opt.batchSize do
      --print('\n\n' ..'evaluating batch [' .. t .. ' through  ' .. t+opt.batchSize .. ']')
      --disp progress
      xlua.progress(t, train_input:size(1))

       -- create mini batch
      local inputs, targets, offsets = {}, {}, {}
      for i = t,math.min(t+opt.batchSize-1, train_input:size(1)) do
        -- load new sample
        local sample = {data[1], data[2][1], data[2][2], data[2][3], data[2][4], data[2][5], data[2][6]}       --use pitch 1st; we are dividing pitch values by 10 because it was incorrectly loaded from vicon
        local input = sample[1]:clone()[i]
        local target = {sample[2]:clone()[i], sample[3]:clone()[i], sample[4]:clone()[i], sample[5]:clone()[i], sample[6]:clone()[i], sample[7]:clone()[i]}
        table.insert(inputs, input)
        table.insert(targets, target) 
        table.insert(offsets, input)      
      end

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
        for i_f = 1,#inputs do
        -- estimate f
          local output, targets_ = neunet:forward(inputs[i_f]), {}
          targets_ = torch.cat({targets[i_f][1], targets[i_f][2], targets[i_f][3],
                                targets[i_f][4], targets[i_f][5], targets[i_f][6]})
          local err = cost:forward(output, targets_)
          f = f + err

          -- estimate df/dW
          local df_do = cost:backward(output, targets_)
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

            print(' err ')
            print(err)
            print('\ndf_do')
            print(df_do)
        end       

        -- normalize gradients and f(X)
        gradParameters:div(#inputs)
        f = f/#inputs

        --retrun f and df/dx
        return f, gradParameters
      end --end feval

      -- optimization on current mini-batch
      if optimMethod == optim.sgd then
        optimMethod(feval, parameters, sgdState)

      elseif optimMethod == msetrain then
        for v = 1, #inputs do
         a, b, c, d = optimMethod(neunet, cost, inputs[v], 
           targets[v], opt, data)
         --print('epoch', epoch, 'pred.errors: ', c, 'acc err', d)
        end

      elseif optimMethod == optim.asgd then
        _, _, average = optimMethod(feval, parameters, optimState)

      else  
        optimMethod(feval, parameters, optimState)
      end

      saveNet(epoch, time)
    end  --end elseif
  end   --end batch for loop
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
   for t = 1,data[1]:size()[1],opt.batchSize do
      -- disp progress
      xlua.progress(t, data[1]:size()[1])

    -- create mini batch
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

    -- test samples
    for j = 1, #inputs do
      local preds = neunet:forward(inputs[j])
    end

    -- timing
    time = sys.clock() - time
    time = time / data[1]:size()[1]
    print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')

    end
end

function saveNet(epoch, time)
  -- time taken
  time = sys.clock() - time
  time = time / trainData[1]:size()[1]
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