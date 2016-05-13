require 'torch'
require 'nn'
require 'rnn'

local cmd = torch.CmdLine()
--Gpu settings
cmd:option('-gpu', 0, 'which gpu to use. -1 = use CPU; >=0 use gpu')
cmd:option('-checkpoint', 'network/lstm-net.t7', 'load the trained network e.g. <lstm-net.t7| rnn-net.t7|mlp-net.t7>')
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
print('model', model)

model:evaluate()

local sample = model:sample(opt)
print('sample', '\n', sample)