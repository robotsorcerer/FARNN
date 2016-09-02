--[[ Run optimization: User has three options:
We could train usin the general mean squared error, Limited- Broyden-Fletcher-GoldFarb and Shanno ]]
require 'torch'

optim_ = {}
--Training using the MSE criterion
     
function msetrain(x, y)
  local lossAcc = 0
  local y_fwd = {}
    neunet:zeroGradParameters();
    --1. predict inputs
    local pred = neunet:forward(x)
    --easiest way to do backprop avoid cluttered data
    if use_cuda then
      y_fwd =  y[3]:cuda()

      -- y_fwd = torch.cat({
      --                     y[1]:cuda(), y[2]:cuda(), 
      --                     y[3]:cuda(), 
      --                     y[4]:cuda(), y[5]:cuda(), y[6]:cuda()
      --                     })
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
    
    collectgarbage()
  return loss, lossAcc
end
                       --yeah, sure. come in and argue :)