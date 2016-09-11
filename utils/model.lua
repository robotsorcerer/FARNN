--[[
  Author: Olalekan Ogunmolu, December 2015 - May 2016
  Freely distributed under the MIT License
]]
print '==> setting up network model'

--number of hidden layers (for mlp network)
transfer    =  nn.ReLU()  

--[[Set up the network, add layers in place as we add more abstraction]]
function contruct_net()
  local bias = true
  if opt.model  == 'mlp' then
      neunet          = nn.Sequential()     
    if opt.data=='glassfurnace' then    
      neunet:add(nn.Linear(3, nhiddens, bias))
    else          
      neunet:add(nn.Linear(ninputs, nhiddens, bias))
    end
    neunet:add(nn.ReLU())                         
    neunet:add(nn.Linear(nhiddens, noutputs)) 
    cost      = nn.MSECriterion()      

  -------------------------------------------------------
  --  Recurrent Neural Net Initializations 
  elseif opt.model == 'rnn' then    
    require 'rnn'
    cost    = nn.SequencerCriterion(nn.MSECriterion())
    ------------------------------------------------------
    --[[m1 = batchSize X hiddenSize; m2 = inputSize X start]]
    --we first model the inputs to states
    local ffwd  =   nn.Sequential()
                  :add(nn.Linear(ninputs, nhiddens, bias))
                  :add(nn.ReLU())      --very critical; changing this to tanh() or ReLU() leads to explosive gradients
                  :add(nn.Linear(nhiddens, nhiddens_rnn, bias))

    local rho         = opt.rho                         -- the max amount of bacprop steos to take back in time
    local start       = noutputs                        -- the size of the output (excluding the batch dimension)        
    local rnnInput    = nn.Linear(start, nhiddens_rnn, bias)     --the size of the output
    local feedback    = nn.Linear(nhiddens_rnn, start, bias)           --module that feeds back prev/output to transfer module

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
                  :add(nn.Linear(start, 6, bias)) 
    neunet = nn.Sequencer(neunet)
--======================================================================================
--Nested LSTM Recurrence
  elseif opt.model == 'lstm' then   
    require 'rnn'
    require 'nngraph'
    nn.LSTM.usenngraph = true -- faster
    local crit = nn.MSECriterion()
    cost = nn.SequencerCriterion(crit)
    neunet = nn.Sequential()    
    local hidden = opt.hiddenSize
    if opt.data == 'glassfurnace' then
      hidden = {3, 10, 100}
    end
    local inputSize = hidden[1]
    for i, hiddenSize in ipairs(hidden) do 
      local rnn
      if opt.gru then -- Gated Recurrent Units
         rnn = nn.GRU(inputSize, hiddenSize, opt.rho, opt.dropoutProb)
      elseif opt.fastlstm then        
        nn.FastLSTM.usenngraph = false -- faster
        if(opt.batchNorm) then
          nn.FastLSTM.bn = false         --apply weights normalization
        end
        nn.FastLSTM.affine = true         
        print(sys.COLORS.magenta .. 'affine state', nn.FastLSTM.affine)
        rnn = nn.FastLSTM(inputSize, hiddenSize, opt.rho)
      else
        rnn = nn.LSTM(inputSize, hiddenSize, opt.rho)
      end
      neunet:add(rnn) 
       
      if opt.dropout and not opt.gru then   --gru comes with dropOut probs
        neunet:add(nn.Dropout(opt.dropoutProb))
      end
       inputSize = hiddenSize
    end

    -- output layer
    if opt.data=='glassfurnace' then         
      neunet:add(nn.Linear(inputSize, 6, bias))      
      -- will recurse a single continuous sequence
      neunet:remember((opt.lstm or opt.fastlstm or opt.gru) or 'eval')
      neunet = nn.Sequencer(neunet)
    elseif (opt.data =='ballbeam') or (opt.data=='robotArm') then
      neunet:add(nn.Linear(inputSize, 1, bias))    
      -- will recurse a single continuous sequence
      neunet:remember((opt.lstm or opt.fastlstm or opt.gru) or 'eval')
      neunet = nn.Sequencer(neunet)
    end
--===========================================================================================
  else    
      error('you have specified an incorrect model. model must be <lstm> or <mlp> or <rnn>')    
  end

  return cost, neunet     
end
