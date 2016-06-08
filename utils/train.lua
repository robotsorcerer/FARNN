--[[Train the network. User can train with 
    1) mlp (a simple feedforward network with one hidden layer)
    2) a recurrent network module stacked after a feedforward network
    3) a long short-term memory network
  ]]
require 'torch'      
require 'optima.optim_'  

function batchNorm(x)
  --apply batch normalization to training data 
  -- http://arxiv.org/pdf/1502.03167v3.pdf
  local eps = 1e-5
  local momentum = 1e-1
  local affine = true
  local BN = nn.BatchNormalization(1, eps, momentum, affine)

  if type(x) == 'userdata' then       --inputs
   x  = BN:forward(x)
    x  = transfer_data(x) --forward doubleTensor as CudaTensor
  elseif type(x) == 'table' then
    for i = 1, #x do
     x[i] = BN:forward(x[i])
      x[i] = transfer_data(x[i])
    end
  end  
  collectgarbage()    
  return x
end


function train_rnn(opt)                          
  local offsets = {}  
  for t = 1, math.min(opt.maxIter, height), opt.batchSize do 
    xlua.progress(t, math.min(opt.maxIter, height))
    offsets = torch.LongTensor(opt.batchSize):random(1,height) 

     -- 1. create a sequence of rho time-steps
    local inputs, targets = {}, {}
    inputs = train_input:index(1, offsets)    
    --increase offsets indices by 1      
    offsets:add(1) -- increase indices by 1
    offsets[offsets:gt(height)] = 1  
    offsets = torch.LongTensor():resize(offsets:size()[1]):copy(offsets)

    --batch of targets
    targets = {train_out[1]:index(1, offsets), train_out[2]:index(1, offsets), 
                      train_out[3]:index(1, offsets), train_out[4]:index(1, offsets), 
                      train_out[5]:index(1, offsets), train_out[6]:index(1, offsets)}

    --pre-whiten the inputs and outputs
    inputs = batchNorm(inputs)
    targets = batchNorm(targets)
  
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

      --reset counters
      loss = 0; iter = 0 ;
    end    
    iter = iter +1 
  end 
    collectgarbage() 
end

function train_lstm(args)
  local offsets = {}                         
  
  for t = 1, math.min(args.maxIter, height), args.batchSize do 
    offsets = torch.LongTensor(args.batchSize):random(1,height)  

    xlua.progress(t, math.min(opt.maxIter, height))
     -- 1. create a sequence of rho time-steps
    local inputs, targets = {}, {}
    inputs = train_input:index(1, offsets)
    --increase offsets indices by 1      
    offsets:add(1) -- increase indices by 1
    offsets[offsets:gt(height)] = 1  
    offsets = torch.LongTensor():resize(offsets:size()[1]):copy(offsets)

    --batch of targets
    targets = {train_out[1]:index(1, offsets), train_out[2]:index(1, offsets), 
                      train_out[3]:index(1, offsets), train_out[4]:index(1, offsets), 
                      train_out[5]:index(1, offsets), train_out[6]:index(1, offsets)}
    
    --pre-whiten the inputs and outputs
    -- if((opt.model=='mlp') or (opt.model=='rnn') or (opt.model == 'lstm') ) then
      inputs = batchNorm(inputs)
      targets = batchNorm(targets)
    -- else
      -- inputs = transfer_data(inputs)
    -- end

    --2. Forward sequence through rnn
    neunet:zeroGradParameters()
    neunet:forget()  --forget all past time steps
    -- print('inputs', inputs)
    local outputs = neunet:forward(inputs)
    local loss = cost:forward(outputs, targets)
          
    --3. do backward propagation through time(Werbos, 1990, Rummelhart, 1986)
    local  gradOutputs = cost:backward(outputs, targets)
    local gradInputs  = neunet:backward(inputs, gradOutputs) 

    --4. update lr
    neunet:updateParameters(opt.rnnlearningRate)
    if (iter*opt.batchSize >= math.min(opt.maxIter, height)) then
      print(string.format("Epoch %d, iter = %d, Loss = %f ", epoch, iter, loss))

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
        print(string.format("Epoch %d, iter = %d, Loss = %f ", epoch, iter, loss))

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