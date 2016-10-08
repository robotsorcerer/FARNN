require 'torch'
require 'nn'
require 'rnn'
require 'nngraph'

local cmd = torch.CmdLine()
--Gpu settings
cmd:option('-gpu', 0, 'which gpu to use. -1 = use CPU; >=0 use gpu')
cmd:option('-checkpoint', 'sr-net/softRobot_fastlstm-net.t7', 'load the trained network e.g. <lstm-net.t7| rnn-net.t7|mlp-net.t7>')
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-verbose', 0)

local opt = cmd:parse(arg)
torch.setnumthreads(8)

local use_cuda = false

local msg
if opt.gpu >= 0  and opt.backend == 'cudnn' then
	require 'cunn'
	require 'cutorch'
	require 'cudnn'
	use_cuda = true
	cutorch.setDevice(opt.gpu + 1)
	msg = string.format('Code deployed on GPU %d with cudnn backend', opt.gpu+1)
else
  msg = 'Running in CPU mode'
  require 'nn'
end
if opt.verbose == 1 then print(msg) end

local checkpoint, model

if use_cuda then
	model = torch.load(opt.checkpoint)
	model:cuda()
else
	model = torch.load(opt.checkpoint)
end

netmods = model.modules;
print('netmods: \n', netmods)

weights,biases = {}, {};
netparams = {}

--[[This if .. then below takes care of recurrent modules ]]
-- if (function(model) return model:find('nn.Recursor') end) then --recurrent nngraph module
if #netmods == 1 then   		--recurrent modules
	local modules 	= netmods[1].recurrentModule.modules
	local length 	= #netmods[1].recurrentModule.modules
	for i = 1, length do
		netparams[i] 	= {['weight']=modules[i].weight, ['bias']=modules[i].bias}
	end

	print('\nnetparams\n:'); 
	print(netparams)
elseif #netmods > 1 then   --mlp modules
	for i = 1, #netmods do		
		netparams[i] 	= {['weight']=netmods[i].weight, ['bias']=netmods[i].bias}
	end

	print('\nnetparams\n:'); 
	print(netparams)
end


model:evaluate()
