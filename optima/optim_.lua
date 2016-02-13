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
    print('epoch', epoch, '\tMSE error: ', errm)
    -- if fm > 150 then learningRate = opt.learningRate
    -- elseif fm <= 150 then learningRate = opt.learningRateDecay end

    gradcrit  = cost:backward(pred, y[i_mse])

    --https://github.com/torch/nn/blob/master/doc/module.md
    neunet:zeroGradParameters();
    neunet:backward(x[i_mse], gradcrit)
    neunet:updateParameters(learningRate);

   -- normalize gradients and f(X)
    gradParameters:div(#x)
    fm = fm/#x

  end

  return gradParameters, errm
end

--Train using the L-BFGS Algorithm
function optim_.lbfgs(neunet, x, y, learningRate) 
   local trainer        = nn.StochasticGradient(neunet2, cost)
   trainer.learningRate = learningRate
   trainer:train({x, y})
   return neunet2
end

collectgarbage()                           --yeah, sure. come in and argue :)
