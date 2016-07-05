--[[Train the network. User can train with 
    1) mlp (a simple feedforward network with one hidden layer)
    2) a recurrent network module stacked after a feedforward network
    3) a long short-term memory network
  ]]
require 'torch'      
require 'optima.optim_'  
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

    xlua.progress(t, math.min(args.maxIter, height))

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
      print('neunet.weight', neunet.weight) 
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
     end

     gradParameters:zero()
     
      inputs =  transfer_data(inputs); targets =  transfer_data(targets)
     -- evaluate the loss function and its derivative wrt x, for that sample
     local loss_x = cost:forward(neunet:forward(inputs), targets)
     neunet:backward(inputs, cost:backward(neunet.output, targets))
     -- print('loss: ', loss_x)
     return loss_x, gradParameters
  end

  local loss, lossAcc = 0, 0
  for i = 1, math.min(opt.maxIter, height) do
    local diff, dC, dC_est    
    local data = (split_data(opt)).train
    -- optimization on current mini-batch
    if optimMethod == msetrain then
       -- create mini batch
      local inputs, targets = {}, {}
      inputs, targets = get_datapair(opt)

      neunet:zeroGradParameters()
      
      loss, lossAcc = optimMethod(inputs, targets) 
      if (iter*opt.batchSize >= math.min(opt.maxIter, height)) then
        print(string.format("Epoch %d, Loss = %f ", epoch, loss))
        logger:add{['mlp training error'] = loss}
        --reset counters
        loss = 0; iter = 0 ;
      end    
      iter = iter +1 
    elseif optimMethod == optim.sgd then      
      for i = 1, data:size(1) do
        _, fs = optimMethod(feval, parameters, sgdState)
        loss = loss + fs[1]
        -- print('fs: ', fs[1], 'loss: ', loss)
        --do gradCheck to be sure grad descent is correct
        diff, dC, dC_est = optim.checkgrad(feval, parameters)
      end
        -- report average error on epoch
        loss = loss / data:size(1); 
        print(string.format('epoch: %2d, iter: %d, current loss: %4.12f: ', epoch, 
              i,  loss, diff))
        -- print('net params: ', neunet:getParameters())
        logger:add{['MLP training error vs. epoch'] = loss}
        logger:style{['MLP training error vs. epoch'] = '-'}
        if opt.plot then logger:plot()  end
    else  
      optimMethod(feval, parameters, optimState)
    end 
  end
  collectgarbage()    
end