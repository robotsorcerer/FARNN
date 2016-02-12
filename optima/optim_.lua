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
    print('epoch', epoch, '\tMSE error: ', fm)

    gradcrit  = cost:backward(pred, y[i_mse])
    neunet:backward(x[i_mse], gradcrit)

    -- penalties (L1 and L2):
    if opt.coefL1 ~= 0 or opt.coefL2 ~= 0 then
       -- locals:
       local norm,sign= torch.norm,torch.sign

       -- Loss:
       fm = fm + opt.coefL1 * norm(parameters,1)
       fm = fm + opt.coefL2 * norm(parameters,2)^2/2

       -- Gradients:
       gradParameters:add( sign(parameters):mul(opt.coefL1) + parameters:clone():mul(opt.coefL2) )
    else
      -- normalize gradients and f(X)
      gradParameters:div(#x)
    end

    --https://github.com/torch/nn/blob/master/doc/module.md
    --neunet:accGradParameters(x, pred, 1)    --https://github.com/torch/nn/blob/master/doc/module.md#nn.Module.accGradParameters
    neunet:zeroGradParameters();
    --neunet:backward(x, gradcrit);
    neunet:updateParameters(learningRate);
   -- print(err)
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

--Train using the Negative Log Likelihood Criterion
function optim_.nllOptim(neunetnll, u_off, y_off, learningRate)
  local x = u_off   local y = y_off 
   local mse_crit       = nn.MSECriterion()
   local trainer        = nn.StochasticGradient(neunet2, mse_crit)
   trainer.learningRate = learningRate
   trainer:train({u_off, y_off})
   return neunet2
end

collectgarbage()                           --yeah, sure. come in and argue :)
