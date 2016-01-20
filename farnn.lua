--[[Source Codes that implement the algorithm described in the paper:
   A Fully Automated Recurrent Neural Network for System Identification and Control
   Jeen-Shing Wang, and Yen-Ping Chen. IEEE Transactions on Circuits and Systems June 2006

   Author: Olalekan Ogunmolu, SeRViCE Lab, UT Dallas, December 2015
   MIT License
   ]]

require 'torch'
require 'nn'
matio   = require 'matio'       --to load .mat files for training


-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('========================================================================')
cmd:text('A Fully Automated Dynamic Neural Network for System Identification')
cmd:text('Based on the IEEE Transactions on Circuits and Systems article by ')
cmd:text()
cmd:text('           Jeen-Shing Wang, and Yen-Ping Chen. June 2006          ')
cmd:text()
cmd:text()
cmd:text('Code by Olalekan Ogunmolu: FirstName [dot] LastName _at_ utdallas [dot] edu')
cmd:text('========================================================================')
cmd:text()
cmd:text()
cmd:text('Options')
cmd:option('-seed', 123, 'initial random seed to use')
cmd:option('-rundir', false, 'log output to file in a directory? Default is false')

-- Model Order Determination Parameters
cmd:option('-pose','posemat7.mat','path to preprocessed data(save in Matlab -v7.3 format)')
cmd:option('-tau', 1, 'what is the delay in the data?')
cmd:option('-quots', 0, 'do you want to print the Lipschitz quotients?; 0 to silence, 1 to print')
cmd:option('-m_eps', 0.01, 'stopping criterion for output order determination')
cmd:option('-l_eps', 0.05, 'stopping criterion for input order determination')

--Gpu settings
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU; >=0 use gpu')
cmd:option('-backend', 'cudnn', 'nn|cudnn')

-- Neural Network settings
cmd:option('-learningRate', 0.0055, 'learning rate for the neural network')
cmd:option('-maxIter', 1000, 'maximaum iteratiopn for training the neural network')

--parse input params
params = cmd:parse(arg)

params.rundir = cmd:string('experiment', params, {dir=false})
paths.mkdir(params.rundir)

-- create log file
cmd:log(params.rundir .. '/log', params)
cmd:addTime('FARNN Identification', '%F %T')
cmd:text('Attaboy, I am running!')
cmd:text()
cmd:text()

-- misc
local opt = cmd:parse(arg)
torch.manualSeed(opt.seed)

-------------------------------------------------------------------------------
-- Basic Torch initializations
-------------------------------------------------------------------------------
--torch.setdefaulttensortype('torch.FloatTensor')            -- for CPU
if opt.gpuid >= 0 then
  require 'cutorch'
  require 'cunn'
  if opt.backend == 'cudnn' then require 'cudnn' end
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.gpuid + 1)                         -- +1 because lua is 1-indexed
  idx = cutorch.getDevice()
  print('System has', cutorch.getDeviceCount(), 'gpu(s).', 'Code is running on GPU:', idx)
end

----------------------------------------------------------------------------------------
--Data Preprocessing
----------------------------------------------------------------------------------------
if opt.gpuid >=0 then
	data    = opt.pose       --ship raw data to gpu
	--print(data)
end

input      = matio.load(data, 'in')						--SIMO System
-- print('\ninput head\n\n', input[{ {1,5}, {1}} ])
trans     = matio.load(data, {'xn', 'yn', 'zn'})
roll      = matio.load(data, {'rolln'})
pitch     = matio.load(data, {'pitchn'})
yaw       = matio.load(data, {'yawn'})
rot       = {roll, pitch, yaw}
--print('\nFull output head\n\n', trans.xn:size()[1], trans.yn:size()[1], trans.zn:size()[1])
--,	 output.rolln:size()[1], output.pitchn:size()[1], output.yawn.size()[1])

out        = matio.load(data, 'zn')
out        = out/10;							--because zn was erroneosly multipled by 10 in the LabVIEW Code.
--print('\nSISO output head\n\n', out[{ {1,5}, {1}} ])

u 		   = input[{ {}, {1}}]
y 		   = out  [{ {}, {1}}]

local   k  = u:size()[1]
-- print('k', k)

off  = torch.ceil( torch.abs(0.6*k))
-- print('off', off)

dataset    = {input, out}

u_off      = input[{{1, off}, {1}}]     --offline data
y_off      = out  [{{1, off}, {1}}]

u_on       = input[{{off + 1, k}, {1}}]	--online data
y_on       = out  [{{off + 1, k}, {1}}]

-- matrix1 = torch.CudaTensor(10):fill(1)
-- print(matrix1) 

unorm      = torch.norm(u_off)
ynorm      = torch.norm(y_off)
q          = ynorm/unorm;

--------------------------------------------------------------------------------------------------------------
--Section 3.2 Find Optimal number of input parameters 
--------------------------------------------------------------------------------------------------------------
function computeqn(u_off, y_off)
	local p = torch.ceil(0.02 * off)       --parameter p that determines number of iterations
	local qinner = {}
	for i = 1, p do
		local qk = {}
		qk[i]		 = torch.abs(y_off[i] - y_off[i + 1])  / torch.norm(u_off[i] - u_off[i + 1]) 
		--print('qk[i', qk[i], 'sqrt(i)', torch.sqrt(i), 'p', p)
		qinner[i] 	 = qk[i] * torch.sqrt(i) 
		qbrace       = torch.cumprod(qinner[i])		
	end		
		print('qbrace', qbrace)
		qn         	 = torch.pow(qbrace, 1/p)
		--print('qbrace', qbrace:size())
		--print('off', off)
	return qn
end

--------------------------------------------------------------------------------------------------------------
--[[Compute the Lipschitz quotients and Estimate Model order

This function implements He and Asada's order selection algorithm as enumerated in their MIT 1993 paper:
"A New Method for Identifying Orders of Input-Output Models for Nonlinear Dynamic Systems"]]
--------------------------------------------------------------------------------------------------------------

function computeq (u_off, y_off)

	tau   = opt.tau                 		    -- Assume a delay of 1 sample. Feel free to change as you might require 
	m_eps  = opt.m_eps						 	-- stopping criterion for output order; can be changed from command line args

	x = {}  	   q = {}			 
	n = {}         m = {}        l = {};  		-- Initialize the input and output orders with an empty array to iterate thru 

	--initialize n,m,l
	i = 1
	n[i] = 1  	m[i] = 1 	l[i]  = 0	
	x[i] = y_off[i]   	q[i] = y_off[i] / 0	   	     	-- becaue q[1] = inf and lua is not fancy for inf values
	--print('\nx ', x[i], 'q ', q[i])
	--[[Determine lagged input output model orders at each time step]]
	i = i + 1
	n[i]	   = i;        m[i]  = n[i] - m[i - 1]  ; l[i] = n[i] - m[i]  ;
	--initialize x'es  		
	x[2]       = u_off[2 - l[2]]

	q[m[2] + l[2]]    = torch.abs(y_off[2] - y_off[2 - tau]) / torch.norm(x[2] - x[2 - tau])  --this is q(1 + 1)
	repeat
		i = i + 1
		--print('i', i)
		n[i]   = i;        m[i]  = n[i] - m[i - 1]  ; l[i] = n[i] - m[i]  ;
		x[i]   = y_off[i - m[i]*tau]		
		q[i]   = torch.abs(y_off[i] - y_off[i - m[i]*tau]) / torch.norm(x[i] - x[i - m[i]*tau])     	  -- this is q(2 + 1)	

		absdiff = torch.abs(q[i] - q[i - tau])
		--print('absdiff = ', absdiff )
		m_epstensor = torch.Tensor({m_eps + opt.l_eps})        --hack to convert stopping criterion to torch.Tensor
		if torch.gt(q[i], (q[i - m[i]*tau]) - m_eps ) and torch.lt(absdiff, m_epstensor) then
			outorder = l[i]
			--print('out order,m= ', outorder, 'l[i]', l[i])
		else
			break
		end
	until torch.gt(q[i], (q[i - m[i]*tau]) - m_eps ) and torch.lt(absdiff, m_epstensor)
		--now determine input order
		--check
		--print('i', i, 'm[i]', m[i], 'n[i]', n[i], 'l[i]', l[i])
	repeat
		n[i]   = i;        m[i]  = n[i] - m[i - 1]  ; l[i] = n[i] - m[i]  ;
		--print('m[i]', m[i])
		x[n[i]]   = u_off[i - m[i] * tau]   		--reselect x3
		q[n[i]]   = torch.abs(y[i] - y[i - m[i]*tau]) / torch.norm(x[i] - x[i - tau])
		local delta = q[i] - q[i - tau]
		--establish the inequality that makes input order determination possible
		if torch.lt(delta, torch.Tensor({0}) ) and torch.lt(delta, torch.Tensor({-0.5})) then                      -- this is when q(i) is sig. smaller than q(i-1)
			i = i + 1
			n[i]	   = i;        m[i]  = n[i] - m[i - 1]  ; l[i] = n[i] - m[i]  ;
			x[i] = u_off[i - m[i] * tau]							  		-- select next x
			q[i] = torch.abs(y[i] - y[i - m[i] * tau]) / torch.norm(x[i] - x[i - m[i] * tau])  -- this is q(2+2)
			--print('q[2 + 2] ', q[i])
			qlt = {}  qgt = {}
			qlt[i]  = q[i - 1] - opt.l_eps		--  lower bound on l stopping criterion
			qgt[i]  = q[i - 1] + opt.l_eps		--	upper bound on l stopping criterion

			--Create inequality q(m+l) - epsilon < q(m + l) < q(m + l - 1) + epsilon
			if torch.gt(q[i], qlt[i]) and torch.lt(q[i], qgt[i]) then --i.e. m_eps - 0.05 < q < m_eps + 0.05
				inorder = l[n[i] - m[i]]
				--print('inoput order: ', inorder)
			else
				break
			end
		end
	until torch.gt(q[i], qlt[i]) and torch.lt(q[i], qgt[i]) 
		
    return inorder, outorder, q
end

inorder, outorder, q =  computeq(u_off, y_off)
print('inorder: ', inorder, 'outorder: ', outorder)
print('system order:', inorder + outorder)

--Print out some Lipschitz quotients for your observation
if opt.quots == 1 then
	for k, v in pairs( q ) do
		print(k, v)
		if k == 5 then      --print only the first five elements
			break
		end
	end
end

qn  = computeqn(u_off, y_off)
print('\nqn:' , qn)
print('Optimal number of input variables is: ', torch.ceil(qn))

--print('\ninput order: ', l, 'output_order: ', m, '\nsystem order: ', n, 'Lipschitz Quotients: ', q)
--print(Res) make install
local inp= torch.randn(2);

--[[In the dynamic layer, feedforward the first layer, then do a self-feedback of 2nd layer neurons
 and weighted connections with the feedback from other neurons  ]]

mlp        = nn.Sequential();  --to enable us plug the hidden layer with dynamic layer in a feedforward manner
input = 1 	 output = 1 	HUs = 3;		-- Hidden units in dynamic layer parameters
mlp:add(nn.Linear(input, HUs))      		-- 1 input vector, 3 hidden units
mlp:add(nn.ReLU())                       	-- maxOut function
print('mlp weights', mlp.weight)
print('mlp biases', mlp.bias)
mlp:add(nn.Linear(HUs, output))				-- to map out of dynamic layer to states

--Training using the MSE criterion
i = 0
repeat
	local input = u_off
	local output = y_off
	criterion = nn.MSECriterion()           -- Loss function
	trainer   = nn.StochasticGradient(mlp, criterion)
	trainer.learningRate = 0.0055
	trainer.maxIteration = opt.maxIter
	--Forward Pass
	local err = criterion:forward(mlp:forward(input), output)
	i = i + 1
	print('iteration', i, 'error: ', err)
	  -- train over this example in 3 steps
	  -- (1) zero the accumulation of the gradients
	  mlp:zeroGradParameters()
	  -- (2) accumulate gradients
	  mlp:backward(input, criterion:backward(mlp.output, output))
	  -- (3) update parameters with a 0.0055 learning rate
	  mlp:updateParameters(trainer.learningRate)
--end
until err <= 141    --stopping criterion for MSE based optimization

--Test Network (MSE)
x = u_on
print('=========================================================')
print('       Example results head post-training using MSE      ')
print(              mlp:forward(x)[{ {1, 5}, {} }]                   )
print('=========================================================')                


--Train using the Negative Log Likelihood Criterion
function gradUpdate(mlp, x, y, learningRate)	
   local NLLcriterion 	= nn.ClassNLLCriterion()
   local pred 	  		= mlp:forward(x)
   local NLLerr 		= criterion:forward(pred, y)
   i = i + 1
   print('NLL_iter', i, 'error: ', NLLerr)
   mlp:zeroGradParameters()
   local t          = criterion:backward(pred, y)
   mlp:backward(x, t)
   mlp:updateParameters(opt.learningRate)
   return NLLerr
end


local inNLL = u_off 	local outNLL = y_off

repeat
	delta = gradUpdate(mlp, inNLL, outNLL, opt.learningRate)
	print('iteration', i, 'NLL error: ', delta)
until delta < 141    --stopping criterion for backward pass