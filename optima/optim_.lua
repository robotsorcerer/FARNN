--[[ Run optimization: User has three options:
We could train usin the general mean squared error, Limited- Broyden-Fletcher-GoldFarb and Shanno or the
Negative Log Likelihood Function ]]
require 'torch'

if use_cuda then
  require 'cutorch'
  require 'cunn'
  if opt.backend == 'cudnn' then
    require 'cudnn'
  end
end

optim_ = {}
--Training using the MSE criterion
     
function msetrain(x, y)
  local lossAcc = 0,  
  gradParameters:zero()
  local y_fwd = {}
    neunet:zeroGradParameters();
    --1. predict inputs
    local pred = neunet:forward(x)

    if use_cuda then
      y_fwd = torch.cat({y[1]:double(), y[2]:double(), y[3]:double(), 
                           y[4]:double(), y[5]:double(), y[6]:double()})
      y_fwd = y_fwd:cuda()      
    else
      y_fwd = torch.cat{y[1], y[2], y[3], y[4], y[5], y[6]}
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

  return loss, lossAcc
end

collectgarbage()

--Train using the L-BFGS Algorithm
function lbfgs(neunet, x, y, learningRate) 
   local trainer        = nn.StochasticGradient(neunet2, cost)
   trainer.learningRate = learningRate
   trainer:train({x, y})
   return neunet2
end

collectgarbage()                           --yeah, sure. come in and argue :)