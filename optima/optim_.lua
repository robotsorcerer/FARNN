--[[ Run optimization: User has three options:
We could train usin the genral mean squared error, Limited- Broyden-Fletcher-GoldFarb and Shanno or the
Negative Log Likelihood Function ]]
require 'torch'
optim_ = {}
--Training using the MSE criterion
function optim_.msetrain(neunet, cost, x, y, learningRate, opt)

  --https://github.com/torch/nn/blob/master/doc/containers.md#Parallel

  --reset grads
  gradParameters:zero()

  local fm = 0
  for i_mse = 1,#x do
    pred      = neunet:forward(x[i_mse])
    --print ('pred', pred)
    errm       = cost:forward(pred, y[i_mse])
    fm = errm + fm
    -- print('epoch', epoch, '\tMSE error: ', fm)
    if fm > 150 then learningRate = opt.learningRate
    elseif fm <= 150 then learningRate = opt.learningRateDecay end

    gradcrit  = cost:backward(pred, y[i_mse])
    neunet:backward(x[i_mse], gradcrit)

    --https://github.com/torch/nn/blob/master/doc/module.md
   -- normalize gradients and f(X)
    gradParameters:div(#x)
    fm = fm/#x

    neunet:zeroGradParameters();
    neunet:updateParameters(learningRate);
  end

  return gradParameters, fm
end

--Train using the L-BFGS Algorithm
function optim_.lbfgs(neunet, x, y, learningRate) 
   local trainer        = nn.StochasticGradient(neunet2, cost)
   trainer.learningRate = learningRate
   trainer:train({x, y})
   return neunet2
end

collectgarbage()                           --yeah, sure. come in and argue :)
