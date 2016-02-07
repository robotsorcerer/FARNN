--[[ Run optimization: User has three options:
We could train usin the genral mean squared error, Limited- Broyden-Fletcher-GoldFarb and Shanno or the
Negative Log Likelihood Function ]]
require 'torch'
_optim = {}
--Training using the MSE criterion
local function msetrain(neunet, x, y, learningRate)
  --https://github.com/torch/nn/blob/master/doc/containers.md#Parallel
    pred      = neunet:forward(x)
    --print ('pred', pred)
    err       = cost:forward(pred, y)
    gradcrit  = cost:backward(pred, y)

    --https://github.com/torch/nn/blob/master/doc/module.md
    --neunet:accGradParameters(x, pred, 1)    --https://github.com/torch/nn/blob/master/doc/module.md#nn.Module.accGradParameters
    neunet:zeroGradParameters();
    neunet:backward(x, gradcrit);
    neunet:updateParameters(learningRate);
   -- print(err)
  return pred, err
end

--Train using the L-BFGS Algorithm
function lbfgs(neunetnll, u_off, y_off, learningRate)
  local x = u_off   local y = y_off 
   local trainer        = nn.StochasticGradient(neunet2, cost)
   trainer.learningRate = learningRate
   trainer:train({u_off, y_off})
   return neunet2
end

--Train using the Negative Log Likelihood Criterion
function nllOptim(neunetnll, u_off, y_off, learningRate)
  local x = u_off   local y = y_off 
   local mse_crit       = nn.MSECriterion()
   local trainer        = nn.StochasticGradient(neunet2, mse_crit)
   trainer.learningRate = learningRate
   trainer:train({u_off, y_off})
   return neunet2
end
collectgarbage()                           --yeah, sure. come in and argue :)
