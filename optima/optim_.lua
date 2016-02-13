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

  local fm = 0     local pred = {}     local errm = {} 
  local gradcrit = {}   local error_acc = {}   local y_fwd = {}--torch.Tensor(1,6)

  for j_mse                 = 1, data[1]:size()[1] do
    for i_mse               = 1,#x do
        print('x: ', x[j_mse])
        --dirty hack to retrieve elements of input vectors
        --x_fwd               = torch.cat({})
        pred                = neunet:forward(x[j_mse])
        --dirty hack to retrieve elements of target vectors
        y_fwd               = torch.cat({y[j_mse][1], y[j_mse][2], y[j_mse][3], y[j_mse][4], y[j_mse][5], y[j_mse][6]})
        --calculate errors
        errm                = cost:forward(pred, y_fwd)
        --show what you got
        --[[ print('y_fwd', y_fwd, 'y[j_mse]', y[j_mse], 'pred', pred, 'x', x[j_mse])]]
        fm                  = errm + fm
        table.insert(error_acc, errm)
        print('==>Printing errors in each minibatch of ', opt.batchSize)
        --print('epoch', epoch, 'errm', error_acc)
        learningRate = opt.learningRate
        --[[if fm > 150 then learningRate = opt.learningRate
        elseif fm <= 150 then learningRate = opt.learningRateDecay end]]

        gradcrit         = cost:backward(pred, y_fwd)

        --https://github.com/torch/nn/blob/master/doc/module.md
        neunet:zeroGradParameters();
        neunet:backward(x[j_mse], gradcrit)
        neunet:updateParameters(learningRate);

        -- normalize gradients and f(X)
        gradParameters:div(#x)
        fm = fm/#x
    end
  end

  return gradParameters, fm, errm, error_acc
end

collectgarbage()

--Train using the L-BFGS Algorithm
function optim_.lbfgs(neunet, x, y, learningRate) 
   local trainer        = nn.StochasticGradient(neunet2, cost)
   trainer.learningRate = learningRate
   trainer:train({x, y})
   return neunet2
end

collectgarbage()                           --yeah, sure. come in and argue :)
