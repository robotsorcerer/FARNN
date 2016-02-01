--[[Source Code that implements the algorithm described in the paper:
   A Fully Automated Recurrent Neural Network for System Identification and Control
   Jeen-Shing Wang, and Yen-Ping Chen. IEEE Transactions on Circuits and Systems June 2006

   Author: Olalekan Ogunmolu, SeRViCE Lab, UT Dallas, December 2015
   MIT License
   ]]

require 'torch'
require 'nn'

do

    local Linear, parent = torch.class('nn.CustomLinear', 'nn.Linear')
    
    -- override the constructor to have the additional range of initialization
    function Linear:__init(inputSize, outputSize, mean, std)
        parent.__init(self,inputSize,outputSize)
                
        self:reset(mean,std)
    end
    
    -- override the :reset method to use custom weight initialization.        
    function Linear:reset(mean,stdv)
        
        if mean and stdv then
            self.weight:normal(mean,stdv)
            self.bias:normal(mean,stdv)
        else
            self.weight:normal(0,1)
            self.bias:normal(0,1)
        end
    end

end

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
cmd:option('-trainStop', 0.5, 'stopping criterion for input order determination')
cmd:option('-sigma', 0.01, 'initialize weights with this std. dev from a normally distributed Gaussian distribution')

--Gpu settings
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU; >=0 use gpu')
cmd:option('-backend', 'cudnn', 'nn|cudnn')

-- Neural Network settings
cmd:option('-learningRate', 0.0055, 'learning rate for the neural network')
cmd:option('-maxIter', 1000, 'maximum iteration for training the neural network')

--parse input params
params = cmd:parse(arg)


-- misc
local opt = cmd:parse(arg)
torch.manualSeed(opt.seed)

-- create log file if user specifies true for rundir
if(opt.rundir==true) then
	params.rundir = cmd:string('experiment', params, {dir=false})
	paths.mkdir(params.rundir)
	cmd:log(params.rundir .. '/log', params)
end

cmd:addTime('FARNN Identification', '%F %T')
cmd:text('Attaboy, I am running!')
cmd:text()
cmd:text()


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
  idx 			= cutorch.getDevice()
  print('System has', cutorch.getDeviceCount(), 'gpu(s).', 'Code is running on GPU:', idx)
  data    		= opt.pose       --ship raw data to gpu
else 
	data 		= opt.pose
end

--  cudata 		= data:cuda()
----------------------------------------------------------------------------------------
--Data Preprocessing
----------------------------------------------------------------------------------------
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

--Neural Network Identification
input = 1 	 output = 1 	HUs = 1;	
mlp        = nn.Sequential();  

print('mlp1 biases Linear', mlp.bias)

mlp:add(nn.Linear(input, HUs, 0, 0.1))
mlp:add(nn.ReLU())                       	

--mlp.modules[1].weights = torch.rand(input, HUs):mul(opt.sigma)
mlp:add(nn.Linear(1, 1, 0, 0.001))
mlp:add(nn.ReLU())

mlp:add(nn.Linear(1, 1, 0, 0.01))
mlp:add(nn.ReLU())

mlp:add(nn.Linear(HUs, output, 0, 0.0501))	

mlp:add(nn.Sigmoid())

mlp2 = mlp:clone('weight', bias);
print('\nmlp1 biases\n', mlp:get(1).bias, '\tmlp1 weights: ', mlp:get(1).weights)

--Training using the MSE criterion
i = 0
function msetrain(mlp, x, y, learningRate)
	repeat
		local input = x
		local output = y
		criterion = nn.MSECriterion()           -- Loss function
		trainer   = nn.StochasticGradient(mlp, criterion)
		learningRate = 0.5 --
		learningRateDecay = 0.0055
		trainer.maxIteration = opt.maxIter
		--Forward Pass
		 err = criterion:forward(mlp:forward(input), output)
		i = i + 1
		print('MSE_iter', i, 'MSE error: ', err)
		  mlp:zeroGradParameters()
		  mlp:backward(input, criterion:backward(mlp2.output, output))		  mlp2:updateParameters(learningRate, learningRateDecay)
	until err <= opt.trainStop    --stopping criterion for MSE based optimization
return i, err
end
--[[]]
ii, mse_error = msetrain(mlp2, u_off, y_off, learningRate)
print('MSE iteration', ii, 'MSE error: ', mse_error, '\n')

 print('mlp gradient weights', mlp.gradWeight)
 print('mlp gradient biases', mlp.gradBias)

--[[
--Test Network (MSE)
x = u_on
print('=========================================================')
print('       Example results head post-training using MSE      ')
print(              mlp:forward(x)[{ {1, 5}, {} }]               )
print('                        Error: ', err                     )
print('=========================================================')                
--]]

--create a deep copy of mlp for NLL training
mlp2 = mlp:clone('weight', bias);
print('\nmlp1 biases\n', mlp:get(1).bias, '\tmlp1 weights: ', mlp:get(1).weights)
--[[print('\nmlp2 biases\n', mlp2:get(1).bias, '\tmlp2 weights: ', mlp2:get(1).weights)
--]]

iN = 0
function nllTrain(mlp, x, y, learningRate)	
   local criterion 		= nn.ClassNLLCriterion()
   local pred 	  		= mlp:forward(x)
   NLLerr 				= criterion:forward(pred, y)
   iN 					= iN + 1
  print('NLL_iter', iN, 'NLL error: ', NLLerr)
   mlp:zeroGradParameters()
   local t          	= criterion:backward(pred, y)
   mlp:backward(x, t)
   mlp:updateParameters(opt.learningRate)
   return iN, NLLerr
end


local inNLL = u_off 	local outNLL = y_off

repeat
	iNLL, delta = nllTrain(mlp, inNLL, outNLL, opt.learningRate)
	--print('NLL_iter', iNLL, 'NLL error: ', delta)
until delta < opt.trainStop    --stopping criterion for backward pass

--[[
for j, module in ipairs(mlp:listModules()) do
	print('MLP1 Modules are: \n', module)
end
--]]