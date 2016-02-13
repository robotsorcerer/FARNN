--[[ Run optimization: User has three options:
We could train usin the genral mean squared error, Limited- Broyden-Fletcher-GoldFarb and Shanno or the
Negative Log Likelihood Function ]]
require 'torch'
optim_ = {}
--Training using the MSE criterion
function optim_.msetrain(neunet, cost, x, y, opt, data)

  --https://github.com/torch/nn/blob/master/doc/containers.md#Parallel

  --reset grads
  gradParameters:zero()

  local fm = 0     local pred = {}     local error = {}     local fm = {}
  local gradcrit = {}

  for j_mse = 1, data[1]:size()[1] do
    for i_mse             = 1,#x do
        pred[j_mse]         = neunet:forward(x[j_mse])
        error[j_mse][i_mse]  = cost:forward(pred[j_mse], y[j_mse][i_mse])
        fm[j_mse][i_mse]    = error[j_mse][i_mse] + fm[j_mse][i_mse]

        --print('epoch', epoch, '\tMSE error: ', error)

        if fm[j_mse][i_mse] > 150 then learningRate = opt.learningRate
        elseif fm[j_mse][i_mse] <= 150 then learningRate = opt.learningRateDecay end

        gradcrit[j_mse][i_mse]   = cost:backward(pred[j_mse], y[j_mse][i_mse])

        --https://github.com/torch/nn/blob/master/doc/module.md
        neunet:zeroGradParameters();
        neunet:backward(x[j_mse], gradcrit[j_mse][i_mse] )
        neunet:updateParameters(learningRate);

        -- normalize gradients and f(X)
        gradParameters:div(#x)
        fm[j_mse][i_mse] = fm[j_mse][i_mse]/#x
    end
  end

  return gradParameters, error
end

--Train using the L-BFGS Algorithm
function optim_.lbfgs(neunet, x, y, learningRate) 
   local trainer        = nn.StochasticGradient(neunet2, cost)
   trainer.learningRate = learningRate
   trainer:train({x, y})
   return neunet2
end

collectgarbage()                           --yeah, sure. come in and argue :)
