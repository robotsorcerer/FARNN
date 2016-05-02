--[[Train the network. User can train with 
    1) mlp (a simple feedforward network with one hidden layer)
    2) a recurrent network module stacked after a feedforward network
    3) a long short-term memory network
  ]]
require 'torch'      
require 'optima.optim_'  


function train_rnn(opt)
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

  for t = 1, math.min(opt.maxIter, height), opt.batchSize do --1, train_input:size(1), opt.batchSize do
    offsets = torch.LongTensor(opt.batchSize):random(1,height)    

    offsets = transfer_data(offsets) 


    xlua.progress(t, height)
     print('\n')
     -- 1. create a sequence of rho time-steps
    local inputs, targets = {}, {}

    inputs = train_input:index(1, offsets)
    targets = {train_out[1]:index(1, offsets), train_out[2]:index(1, offsets), 
                      train_out[3]:index(1, offsets), train_out[4]:index(1, offsets), 
                      train_out[5]:index(1, offsets), train_out[6]:index(1, offsets)}
    --increase offsets indices by 1      
    offsets:add(1) -- increase indices by 1
    offsets[offsets:gt(height)] = 1

    offsets = torch.LongTensor():resize(offsets:size()[1]):copy(offsets)
    offsets = transfer_data(offsets)  
  
    --2. Forward sequence through rnn
    neunet:zeroGradParameters()
    neunet:forget()  --forget all past time steps

    inputs_, outputs = {}, {}
    local loss = 0
     outputs = neunet:forward(inputs)
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

    saveNet(epoch, time)
  end 
end

function train_lstm(args)
  --track the epochs
  epoch = epoch or 1
  --time we started training
  local time = sys.clock()

  --do one epoch
  print('<trainer> on training set: ')
  print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. args.batchSize .. ']\n')

  offsets = {}
  --form mini batch    
  local iter = 1
  local target, input = {}, {}                               
  
    for i=1,args.batchSize do
       table.insert(offsets, math.ceil(math.random()*args.batchSize))
    end
    offsets = torch.LongTensor(offsets)

    print('offsets', offsets)
    offsets = transfer_data(offsets)
    
    xlua.progress(epoch, height)
     print('\n')

     -- 1. create a sequence of rho time-steps

    local inputs, targets = {}, {}
    for step = 1, args.rho do
      inputs[step] = train_input:index(1, offsets) 
      --increase offsets indices by 1      
      offsets:add(1) -- increase indices by 1
      offsets[offsets:gt(height)] = 1
      offsets = transfer_data(offsets)  

      targets[step] = {train_out[1]:index(1, offsets), train_out[2]:index(1, offsets), 
                        train_out[3]:index(1, offsets), train_out[4]:index(1, offsets), 
                        train_out[5]:index(1, offsets), train_out[6]:index(1, offsets)}
      offsets = torch.LongTensor():resize(offsets:size()[1]):copy(offsets)
    end

    print('inputs', '\n', inputs) 
    print('targets', '\n', targets)

    
    --2. Forward sequence through rnn
    neunet:zeroGradParameters()

    inputs_, outputs = {}, {}
    local loss = 0
    outputs = neunet:forward(inputs)
    print('outputs', outputs)
    loss = cost:forward(outputs, targets)

    print(string.format("Step %d, Loss = %f ", iter, loss))

    if iter % 10 == 0 then collectgarbage() end -- good idea to do this once in a while, i think
          
    --3. do backward propagation through time(Werbos, 1990, Rummelhart, 1986)
    local  gradOutputs = cost:backward(outputs, targets)
    local gradInputs  = neunet:backward(inputs, gradOutputs) 
      

      print('gradInputs'); print(gradInputs)

    --4. update lr
    neunet:updateParameters(opt.rnnlearningRate)

    iter = iter + 1 

    saveNet(epoch, time)
  --end 
end

function train_mlp(opt)
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
    xlua.progress(t, height)

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
         targets[v], opt)
       --print('epoch', epoch, 'pred.errors: ', c, 'acc err', d)
      end

    elseif optimMethod == optim.asgd then
      _, _, average = optimMethod(feval, parameters, optimState)

    else  
      optimMethod(feval, parameters, optimState)
    end

    saveNet(epoch, time)
  end
end