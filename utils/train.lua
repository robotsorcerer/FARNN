--[[Train the network. User can train with 
    1) mlp (a simple feedforward network with one hidden layer)
    2) a recurrent network module stacked after a feedforward network
    3) a long short-term memory network
  ]]
require 'torch'      
-- require 'optima.optim_'  
require 'data/dataparser'

function train_rnn(opt)                          
  local offsets = {}  
  for t = 1, math.min(opt.maxIter, height), opt.batchSize do 
    xlua.progress(t, math.min(opt.maxIter, height))

     -- 1. create a sequence of rho time-steps
    local inputs, targets = {}, {}
    inputs, targets = get_datapair(opt)
  
    --2. Forward sequence through rnn
    neunet:zeroGradParameters()
    neunet:forget()  --forget all past time steps

    local outputs = neunet:forward(inputs)    
    local loss    = cost:forward(outputs, targets)

    if iter % 10 == 0 then collectgarbage() end -- good idea to do this once in a while, i think
          
    --3. do backward propagation through time(Werbos, 1990, Rummelhart, 1986)
    local  gradOutputs = cost:backward(outputs, targets)
    local gradInputs  = neunet:backward(inputs, gradOutputs) 

    --4. update lr
    neunet:updateParameters(opt.rnnlearningRate)
    if (iter*opt.batchSize >= math.min(opt.maxIter, height)) then
      print(string.format("Epoch %d, iter = %d, Loss = %f ", epoch, iter, loss))  
      print('neunet.weight', neunet.weight)    
      logger:add{['RNN training error'] = loss}
      --reset counters
      loss = 0; iter = 0 ;
    end    
    iter = iter +1 
  end 
    collectgarbage() 
end

function train_lstms(args)
  local offsets = {}                         
  
  for t = 1, math.min(args.maxIter, height), args.batchSize do 
    offsets = torch.LongTensor(args.batchSize):random(1,height)  

    xlua.progress(t, math.min(args.maxIter, height))

     -- 1. create a sequence of rho time-steps
    local inputs, targets = {}, {}    
    offsets = torch.LongTensor():resize(offsets:size()[1]):copy(offsets)
    inputs = nn.JoinTable(1):forward{train_input:index(1, offsets), 
                                              train_out[1]:index(1, offsets), train_out[2]:index(1, offsets), 
                                              train_out[3]:index(1, offsets), train_out[4]:index(1, offsets), 
                                              train_out[5]:index(1, offsets), train_out[6]:index(1, offsets)}
    inputs = batchNorm(inputs)
    -- print('inputs', inputs:size())

    --increase offsets indices by 1      
    offsets:add(1) -- increase indices by 1
    offsets[offsets:gt(height)] = 1 

    targets =  nn.JoinTable(1):forward{train_input:index(1, offsets), train_out[1]:index(1, offsets), train_out[2]:index(1, offsets), 
                  train_out[3]:index(1, offsets), train_out[4]:index(1, offsets), 
                  train_out[5]:index(1, offsets), train_out[6]:index(1, offsets)}
    targets = batchNorm(targets)             
    
    --2. Forward sequence through rnn
    neunet:zeroGradParameters()
    neunet:forget()  --forget all past time steps
    local outputs = neunet:forward(inputs)
    -- print('outputs--^', outputs:size())
    local loss = cost:forward(outputs, targets)
          
    --3. do backward propagation through time(Werbos, 1990, Rummelhart, 1986)
    local  gradOutputs = cost:backward(outputs, targets)
    local gradInputs  = neunet:backward(inputs, gradOutputs) 

    --4. update lr
    neunet:updateParameters(opt.rnnlearningRate)
    if (iter*opt.batchSize >= math.min(opt.maxIter, height)) then
      print(string.format("Epoch %d, Loss = %f ", epoch,  loss))
      print('neunet.weight', neunet.weight) 
      if opt.model=='lstm' then logger:add{['LSTM training error vs. epoch'] = loss}
        elseif opt.model=='fastlstm' then logger:add{['FastLSTM training error'] = loss} end
      --reset counters
      loss = 0; iter = 0 ;
      collectgarbage()
    end    
    iter = iter +1 
  end  
end

function train_lstm(args)
  local offsets = {}                         
  
  for t = 1, math.min(args.maxIter, height), args.batchSize do 

     -- 1. create a sequence of rho time-steps
    local inputs, targets = {}, {}
    inputs, targets = get_datapair(args)
    --2. Forward sequence through rnn
    neunet:zeroGradParameters()
    neunet:forget()  --forget all past time steps
    local outputs = neunet:forward(inputs)
    local loss = cost:forward(outputs, targets)
          
    --3. do backward propagation through time(Werbos, 1990, Rummelhart, 1986)
    local  gradOutputs = cost:backward(outputs, targets)
    local gradInputs  = neunet:backward(inputs, gradOutputs) 

    --4. update lr
    neunet:updateParameters(opt.rnnlearningRate)
    if (iter*opt.batchSize >= math.min(opt.maxIter, height)) then
      print(string.format("Epoch %d,  Loss = %f ", epoch,  loss))
      if opt.model=='lstm' then 
        if opt.gru then
          logger:add{['GRU training error vs. epoch'] = loss}
        elseif opt.fastlstm then
          logger:add{['FastLSTM training error vs. epoch'] = loss}
        else
          logger:add{['LSTM training error vs. epoch'] = loss}
        end
      end
      --reset counters
      loss = 0; iter = 0 ;
      collectgarbage()
    end    
    iter = iter +1 
  end  
end
     
function train_mlp(opt)

  local inputs, targets
  local data = (split_data(opt)).train
  
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
    local lossAcc = 0
    local y_fwd = {}
      neunet:zeroGradParameters();
      --1. predict inputs
      local pred = neunet:forward(x)

      if use_cuda then
        y_fwd =  y[3]:cuda()

        -- y_fwd = torch.cat({
        --                     y[1]:cuda(), y[2]:cuda(), 
        --                     y[3]:cuda(), 
        --                     y[4]:cuda(), y[5]:cuda(), y[6]:cuda()
        --                     })
      else
        y_fwd = y[3]:cuda() --torch.cat{y[1], y[2], y[3], y[4], y[5], y[6]}
      end
      
      --2. Compute loss
      local loss    = cost:forward(pred, y_fwd)
      lossAcc = loss + lossAcc
      local gradOutputs = cost:backward(pred, y_fwd)
      local gradInputs  = neunet:backward(x, gradOutputs)
      --3. update the parameters
      neunet:updateParameters(opt.learningRate);

        -- normalize gradients and f(X)
      gradParameters:div(opt.batchSize)
      lossAcc = lossAcc/math.min(opt.batchSize, height)
      
      collectgarbage()      --yeah, sure. come in and argue :)
    return loss, lossAcc  
  end

  local loss, lossAcc, iter = 0, 0, 0
  for i = 1, math.min(opt.maxIter, height) do
    local diff, dC, dC_est    
    -- optimization on current mini-batch
    if optimMethod == msetrain then
       -- create mini batch
      local inputs, targets = {}, {}
      inputs, targets = get_datapair(opt)

      neunet:zeroGradParameters()
      
      loss, lossAcc = fmseval(inputs, targets)       
      iter = iter +1 
      print(string.format("Epoch %d, Iter %d, Loss = %f ", epoch, iter, loss))
      logger:add{['mlp training error vs. #iterations'] = loss}
      logger:style{['mlp training error vs. #iterations'] = '-'}
      if opt.plot then logger:plot()  end
--[[
      if (iter*opt.batchSize >= math.min(opt.maxIter, height)) then
        -- print(string.format("Epoch % d, Iter %d, Loss = %f ", epoch, iter, loss))
        logger:add{['mlp training error vs. epoch'] = loss}
        logger:style{['mlp training error vs. epoch'] = '-'}
        if opt.plot then logger:plot()  end
        --reset counters
        -- loss = 0; iter = 0 ;
      end 
      ]]   
    elseif optimMethod == optim.sgd then  
        _, fs = optimMethod(feval, parameters, sgdState)

        -- print('sgdState: ', sgdState)
        loss = loss + fs[1]
        -- print('iter:', i, 'loss: ', fs[1])
        diff, dC, dC_est = optim.checkgrad(feval, parameters)
        
        loss = loss --/ data:size(1); 
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