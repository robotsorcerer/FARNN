--[[Train the network. User can train with 
    1) mlp (a simple feedforward network with one hidden layer)
    2) a recurrent network module stacked after a feedforward network
    3) a long short-term memory network
  ]]
require 'torch'      
require 'optima.optim_'  


function train_rnn(opt)

  offsets = {}
  --form mini batch    
  local iter = 1
  local target, input = {}, {}                               

  for t = 1, math.min(opt.maxIter, height), opt.batchSize do 

    offsets = torch.LongTensor(opt.batchSize):random(1,height)   
    offsets = transfer_data(offsets) 

    xlua.progress(t, math.min(opt.maxIter, height))
     print('\n')
     -- 1. create a sequence of rho time-steps
    local inputs, targets = {}, {}
    inputs = train_input:index(1, offsets)
    
    --increase offsets indices by 1      
    offsets:add(1) -- increase indices by 1
    offsets[offsets:gt(height)] = 1  
    offsets = torch.LongTensor():resize(offsets:size()[1]):copy(offsets)
    offsets = transfer_data(offsets) 

    --batch of targets
    targets = {train_out[1]:index(1, offsets), train_out[2]:index(1, offsets), 
                      train_out[3]:index(1, offsets), train_out[4]:index(1, offsets), 
                      train_out[5]:index(1, offsets), train_out[6]:index(1, offsets)}
  
    --2. Forward sequence through rnn
    neunet:zeroGradParameters()
    neunet:forget()  --forget all past time steps

    outputs = {}
    local loss = 0
     outputs = neunet:forward(inputs)
     loss    = loss + cost:forward(outputs, targets)
    print(string.format("Epoch %d, Step %d, Loss = %f ", epoch, iter, loss))

    if iter % 10 == 0 then collectgarbage() end -- good idea to do this once in a while, i think
          
    --3. do backward propagation through time(Werbos, 1990, Rummelhart, 1986)
    local  gradOutputs = cost:backward(outputs, targets)
    local gradInputs  = neunet:backward(inputs, gradOutputs) 
      
    --print('gradInputs'); print(gradInputs)

    --4. update lr
    neunet:updateParameters(opt.rnnlearningRate)

    iter = iter + 1 
  end 
end

function train_lstm(args)

  offsets = {}
  --form mini batch    
  local iter = 1
  local target, input = {}, {}                               
  
  for t = 1, math.min(args.maxIter, height), args.batchSize do 

    offsets = torch.LongTensor(args.batchSize):random(1,height)    
    offsets = transfer_data(offsets) 

    xlua.progress(t, math.min(opt.maxIter, height))
     print('\n')
     -- 1. create a sequence of rho time-steps
    local inputs, targets = {}, {}

    inputs = train_input:index(1, offsets)

    --increase offsets indices by 1      
    offsets:add(1) -- increase indices by 1
    offsets[offsets:gt(height)] = 1  
    offsets = torch.LongTensor():resize(offsets:size()[1]):copy(offsets)
    offsets = transfer_data(offsets) 

    --batch of targets
    targets = {train_out[1]:index(1, offsets), train_out[2]:index(1, offsets), 
                      train_out[3]:index(1, offsets), train_out[4]:index(1, offsets), 
                      train_out[5]:index(1, offsets), train_out[6]:index(1, offsets)}
    
    --2. Forward sequence through rnn
    neunet:zeroGradParameters()
    --neunet:forget()  --forget all past time steps

    outputs = {}
    local loss = 0
    outputs = neunet:forward(inputs)
    --print('outputs', outputs)
    loss = cost:forward(outputs, targets)

    print(string.format("Epoch: %d, Step %d, Loss = %f ", epoch, iter, loss))

    if iter % 10 == 0 then collectgarbage() end -- good idea to do this once in a while, i think
          
    --3. do backward propagation through time(Werbos, 1990, Rummelhart, 1986)
    local  gradOutputs = cost:backward(outputs, targets)
    local gradInputs  = neunet:backward(inputs, gradOutputs) 
      
    -- print('gradInputs'); print(gradInputs)

    --4. update lr
    neunet:updateParameters(opt.rnnlearningRate)

    iter = iter + 1 
  end 
end
     
local iter = 0;
function train_mlp(opt)

  for t = 1, opt.maxIter, opt.batchSize do

    xlua.progress(t, math.min(opt.maxIter, height))

     -- create mini batch
    local inputs, targets, offsets = {}, {}, {}
    for i = t,math.min(t+opt.batchSize-1, height) do
      -- load new sample
      local input = train_input:clone()[i]
      local target = {train_out[1]:clone()[i], train_out[2]:clone()[i], train_out[3]:clone()[i], 
                      train_out[4]:clone()[i], train_out[5]:clone()[i], train_out[6]:clone()[i]}
      table.insert(inputs, input)
      table.insert(targets, target) 
      table.insert(offsets, input)      
    end

    -- print('inputs', inputs)
    -- print('targets', targets)
    neunet:zeroGradParameters()
          
    -- optimization on current mini-batch
    if optimMethod == msetrain then
      local loss, lossAcc
      for i_f = 1,#inputs do        
        loss, lossAcc = optimMethod(inputs[i_f], targets[i_f])    
        print("Epoch: ", epoch)
        print(string.format("Step %d, Loss = %f, Batch Normed Loss = %f ", iter, loss, lossAcc)) 
      end                    
    iter = iter + 1  

    elseif optimMethod == optim.sgd then
      optimMethod(feval, parameters, sgdState)

    else  
      optimMethod(feval, parameters, optimState)

    end    
  end
end

function train_rnns(opt)

  offsets = {}
  --form mini batch    
  local iter = 1
  local target, input = {}, {}                               

  for t = 1, math.min(opt.maxIter, height), opt.batchSize do 

    offsets = torch.LongTensor(opt.batchSize):random(1,height)    
    offsets = transfer_data(offsets) 

    xlua.progress(t, math.min(opt.maxIter, height))
     print('\n')
     -- 1. create a sequence of rho time-steps
    local inputs, targets = {}, {}
    for step = 1, opt.rho do
      inputs[step] = inputs[step] or train_input.new()
      inputs[step]:index(train_input, 1, offsets)

      --increase offsets indices by 1      
      offsets:add(1) -- increase indices by 1
      offsets[offsets:gt(height)] = 1  
      offsets = torch.LongTensor():resize(offsets:size()[1]):copy(offsets)
      offsets = transfer_data(offsets) 

      targets[step] = targets[step] or {train_out[1].new(), train_out[2].new(), train_out[3].new(),
                                        train_out[4].new(), train_out[5].new(), train_out[6].new()}
      for i = 1, #train_out do                                        
        targets[step][i]:index(train_out[1], 1, offsets)
      end
    end    

    --targets = nn.FlattenTable():forward(targets)
    print('inputs', '\n', inputs) 
    print('targets', '\n', targets)
  
    --2. Forward sequence through rnn
    neunet:zeroGradParameters()
    --neunet:forget()  --forget all past time steps

    --cost = nn.CriterionTable(cost)

    inputs_, outputs = {}, {}
    local loss = 0
     outputs = neunet:forward(inputs)
     print('outputs', outputs)
     loss    = loss + cost:forward(outputs, targets)
    print(string.format("Step %d, Loss = %f ", iter, loss))

    if iter % 10 == 0 then collectgarbage() end -- good idea to do this once in a while, i think
          
    --3. do backward propagation through time(Werbos, 1990, Rummelhart, 1986)
    local  gradOutputs = cost:backward(outputs, targets)
    local gradInputs  = neunet:backward(inputs, gradOutputs) 
      
    print('gradInputs'); print(gradInputs)

    --4. update lr
    neunet:updateParameters(opt.rnnlearningRate)

    iter = iter + 1 
  end 
end