--[[Train the network on glassfurnace data. User can train with 
    1) mlp (a simple feedforward network with one hidden layer)
    2) a recurrent network module stacked on a feedforward network
    3) a long short-term memory network

    Author: Olalekan Ogunmolu, December 2015 - May 2016
    Freely distributed under the MIT License.
  ]]
require 'torch'      
-- require 'optima.optim_'  
require 'data/dataparser'

function train_rnn(opt)                          
  iter = iter or 0
  for t = 1,opt.maxIter, opt.batchSize do 
     -- 1. create a sequence of rho time-steps
    local inputs, targets = {}, {}
    inputs, targets       = get_datapair(opt)  
    --2. Forward sequence through rnn
    neunet:zeroGradParameters()
    neunet:forget()  --forget all past time steps
    local outputs = neunet:forward(inputs) 

    local loss    = cost:forward(outputs, targets)

    if iter % 10  == 0 then collectgarbage() end 
          
    --3. do backward propagation through time(Werbos, 1990, Rummelhart, 1986)
    local  gradOutputs = cost:backward(outputs, targets)
    local  gradInputs  = neunet:backward(inputs, gradOutputs) 

    --4. update lr
    local lr = opt.learningRate/(1+ opt.decay*epoch) --learning Rate schedule
    neunet:updateParameters(lr)

    print(string.format("Epoch %d, iter = %d, Loss = %f, lr = %2.4f", epoch, iter, loss, lr))  
    logger:add{['RNN training error vs. #iterations'] = loss}
    logger:style{['RNN training error vs. #iterations'] = '-'}
    if opt.plot then logger:plot() end  
    iter = iter + 1 
  end 
    collectgarbage()        -- yeah, sure. why not?
end

function train_lstm(opt)
  local iter = iter or 0; 
  for t = 1, opt.maxIter, opt.batchSize do 
     -- 1. create a sequence of rho time-steps
    local inputs, targets = {}, {}
    inputs, targets = get_datapair(opt)
    --2. Forward sequence through rnn
    neunet:zeroGradParameters()
    neunet:forget()  --forget all past time steps
    local outputs = neunet:forward(inputs)
    if noutputs   == 1 then targets = {targets[3]} end
    local loss = cost:forward(outputs, targets) 

    if iter % 10  == 0 then collectgarbage() end

    --3. do backward propagation through time(Werbos, 1990, Rummelhart, 1986)
    local  gradOutputs  = cost:backward(outputs, targets)
    local gradInputs    = neunet:backward(inputs, gradOutputs) 

    --4. update lr
    if opt.fastlstm then opt.rnnlearningRate = 5e-3 end
    neunet:updateParameters(opt.rnnlearningRate)
    -- if (iter*opt.batchSize >= math.min(opt.maxIter, height)) then
    print(string.format("Epoch %d,iter = %d,  Loss = %f ", 
            epoch, iter, loss))
    if opt.model=='lstm' then 
      if opt.gru then
        logger:add{['GRU training error vs. epoch'] = loss}
        logger:style{['GRU training error vs. epoch'] = '-'}
        if opt.plot then logger:plot()  end
      elseif opt.fastlstm then
        logger:add{['FastLSTM training error vs. epoch'] = loss}
        logger:style{['FastLSTM training error vs. epoch'] = '-'}
        if opt.plot then logger:plot()  end
      else
        logger:add{['LSTM training error vs. epoch'] = loss}
        logger:style{['LSTM training error vs. epoch'] = '-'}
        if opt.plot then logger:plot()   end
      end
    end
    --reset counters
    -- loss = 0; iter = 0 ;
    collectgarbage()
    -- end    
    iter = iter +1 
  end  
end
     
function train_mlp(opt)

  local inputs, targets
  local data = (split_data(opt)).train    
  local loss, lossAcc= 0, 0; 
  --[[this is my sgd optimizer]]
  feval = function(params_new)
     if parameters ~= params_new then
        parameters:copy(params_new)
     end

     _nidx_ = (_nidx_ or 0) + 1
     if _nidx_ > (#data)[1] then _nidx_ = 1 end

     local sample = data[_nidx_]
     if opt.data == 'robotArm' or opt.data == 'ballbeam' then
      inputs = sample[{ {1} }]
      targets = sample[{ {2} }]
     elseif opt.data == 'softRobot' then
        inputs = torch.Tensor(1):zero():fill(sample[{1}])   --cuda() method does not accept numbers
        targets = sample[{ {4} }]/10
     end

     inputs =  transfer_data(inputs); targets =  transfer_data(targets)
        -- print('inputs: ', inputs, ' targets: ', targets)

     neunet:zeroGradParameters()
     gradParameters:zero()
     
     -- evaluate the loss function and its derivative wrt x, for that sample
     local outputs  = neunet:forward(inputs)
     local loss_x   = cost:forward(outputs, targets)
     local grad     = cost:backward(outputs, targets)
     local gradInput = neunet:backward(inputs, grad)

     -- normalize gradients and f(X)
     gradParameters:div(inputs:size(1))

     return loss_x, gradParameters
  end

  --[[compute gradient for batched error in closure]]
  fmseval = function(x, y)
    local y_fwd = {}
      neunet:zeroGradParameters();
      --1. predict inputs
      local pred = neunet:forward(x)
      
      --2. Compute loss
      local loss    = cost:forward(pred, y)
      lossAcc       = loss + lossAcc
      local gradOutputs = cost:backward(pred, y)
      local gradInputs  = neunet:backward(x, gradOutputs)
      --3. update the parameters
      neunet:updateParameters(opt.learningRate);

        -- normalize gradients and f(X)
      gradParameters:div(opt.batchSize)
      lossAcc = lossAcc/math.min(opt.batchSize, height)
      
      collectgarbage()      --yeah, sure. come in and argue :)
    return loss, lossAcc  
  end

  iter  = iter or 0
  for t = 1, opt.maxIter, opt.batchSize do
     -- create mini batch
    local inputs, targets = {}, {}

    inputs, targets = get_datapair(opt)    
    -- optimization on current mini-batch
    if optimMethod == msetrain then

      neunet:zeroGradParameters()
      
      loss, lossAcc = fmseval(inputs, targets) 

      if iter % 100  == 0 then collectgarbage() end 

      print(string.format("Epoch %d, Iter %d, Loss = %f ", epoch, iter, loss))
      logger:add{['mlp training error vs. #iterations'] = loss}
      logger:style{['mlp training error vs. #iterations'] = '-'}
      if opt.plot then logger:plot()  end
      iter = iter +1 

    elseif optimMethod == optim.sgd then  
      _, fs = optimMethod(feval, parameters, sgdState)

      loss = loss + fs[1]
      
      print(string.format('epoch: %2d, iter: %d, current loss: %4.12f ', epoch, 
            i,  loss))
      logger:add{['MLP training error vs. epoch'] = loss}
      logger:style{['MLP training error vs. epoch'] = '-'}
      if opt.plot then logger:plot()  end
    else  
      optimMethod(feval, parameters, optimState)
    end 
  end
  collectgarbage()    
end