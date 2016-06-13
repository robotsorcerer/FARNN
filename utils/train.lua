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
      if opt.model=='lstm' then logger:add{['LSTM training error'] = loss}
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
      if opt.model=='lstm' then 
        if opt.gru then
          logger:add{['GRU training error'] = loss}
        elseif opt.fastlstm then
          logger:add{['FastLSTM training error'] = loss}
        else
          logger:add{['LSTM training error'] = loss}
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
  for t = 1, math.min(opt.maxIter, height), opt.batchSize do

    xlua.progress(t, math.min(opt.maxIter, height))
     -- create mini batch
    local inputs, targets = {}, {}
      -- load new sample
      local offsets = {}
      offsets = torch.LongTensor(opt.batchSize):random(1,height) 
      inputs = train_input:index(1, offsets)
      --batch of targets
      targets = {train_out[1]:index(1, offsets), train_out[2]:index(1, offsets), 
                  train_out[3]:index(1, offsets), train_out[4]:index(1, offsets), 
                  train_out[5]:index(1, offsets), train_out[6]:index(1, offsets)
                }    
    --pre-whiten the inputs and outputs in the mini-batch
    inputs = batchNorm(inputs)
    targets = batchNorm(targets)


    neunet:zeroGradParameters()
          
    -- optimization on current mini-batch
    if optimMethod == msetrain then
      local loss, lossAcc       
      loss, lossAcc = optimMethod(inputs, targets) 
      if (iter*opt.batchSize >= math.min(opt.maxIter, height)) then
        print(string.format("Epoch %d, Loss = %f ", epoch, loss))
        logger:add{['mlp training error'] = loss}
        --reset counters
        loss = 0; iter = 0 ;
      end    
      iter = iter +1 
    elseif optimMethod == optim.sgd then
      optimMethod(feval, parameters, sgdState)

    else  
      optimMethod(feval, parameters, optimState)
    end 
  end
  collectgarbage()    
end