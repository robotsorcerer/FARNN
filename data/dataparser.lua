--[[
  Author: Olalekan Ogunmolu, December 2015 - May 2016
  Freely distributed under the MIT License
]]
require 'torch'

local function data_path_printer(x)  
  print(sys.COLORS.green .. string.format("you have specified the data path %s", x))
end

local function get_filename(x)
	local filename = x:match("(%a+)")
	local filenamefull = 'data/' .. filename .. '.mat'
  return filename, filenamefull
end

function split_data(opt)
	
	local filename, filenamefull = get_filename(opt.data)  -- we strip the filename extension from the data
	  -- if epoch==1 then data_path_printer(filenamefull) ends

	  local splitData = {}
	--ballbeam and robotarm are siso systems from the DaiSy dataset
	if (string.find(filename, 'robotArm')) or (string.find(filename, 'ballbeam')) then  

	  data = matio.load(filenamefull)
	  data = data[filename];
	  input = data[{{}, {1}}]     

	  out = data[{{}, {2}}]

	  k = input:size(1)
	  off = torch.ceil(torch.abs(0.1*k))

	  splitData.train = data[{{1, off}, {1, 2}}]
	  splitData.test = data[{{off+1}, {1, 2}}]

	  splitData.train_input = input[{{1, off}, {1}}]
	  splitData.train_out = {out[{{1, off}, {}}]}

	  --create testing data
	  splitData.test_input = input[{{off + 1, k}, {1}}]
	  splitData.test_out   = {  out[{{off+1, k}, {1}}] }  

	  width       = splitData.train_input:size(2)
	  height      = splitData.train_input:size(1)
	  ninputs     = 1
	  noutputs    = 1
	  nhiddens_rnn = 6 
	--SIMO System from my soft_robot system
	elseif(string.find(filename, 'softRobot')) then 
	  data = matio.load(filenamefull);  
	  data = data.pose
	  input       = data[{{}, {1}}]    
	  out         = { 
	                  data[{{}, {2}}],       --x
	                  data[{{}, {3}}],       --y
	                  0.1* data[{{}, {4}}],  --z
	                  data[{{}, {5}}],       --roll
	                  data[{{}, {6}}],       --pitch
	                  data[{{}, {7}}]       --yaw
	                }

	  k           = input:size(1)    
	  off         = torch.ceil( torch.abs(0.6*k))

	  splitData.train_input = input[{{1, off}, {1}}]   -- order must be preserved. cuda tensor does not support csub yet
	  splitData.train_out   = { 
	                 out[1][{{1, off}, {1}}], out[2][{{1, off}, {1}}],            -- most of the work is done here              (out[{{1, off}, {1}}])/10, outlln[{{1, off}, {1}}], 
	                 out[3][{{1, off}, {1}}], out[4][{{1, off}, {1}}],
	                 out[5][{{1, off}, {1}}], out[6][{{1, off}, {1}}]
	                } 
	  splitData.train = data[{{1, off}, {1, 7}}]
	  --create testing data
	  splitData.test =  data[{{off + 1, k}, {1, 7}}]
	  splitData.test_input = input[{ {off+1, k}, {1}}]
	  splitData.test_out   = {
	                 out[1][{{off+1, k}, {1}}], out[2][{{off+1, k}, {1}}], 
	                 out[3][{{off+1, k}, {1}}], out[4][{{off+1, k}, {1}}], 
	                 out[5][{{off+1, k}, {1}}], out[6][{{off+1, k}, {1}}] 
	                }  

	  splitData.test = {splitData.test_input, splitData.test_out}

	  width       = splitData.train_input:size(2)
	  height      = splitData.train_input:size(1)

	  ninputs     = 1; noutputs    = 1; nhiddens = 1; nhiddens_rnn = 1
	  
	-- MIMO dataset from the Daisy  glassfurnace dataset (3 inputs, 6 outputs)
	elseif (string.find(filename, 'glassfurnace')) then
	  data = matio.load(filenamefull)  ;   data = data[filename];
	  -- three inputs i.e. heating input, cooling input, & heating input
	  input =   data[{{}, {2, 4}}]   

	  -- six outputs from temperatures sensors in a cross sectiobn of the furnace          
	  out =   data[{{}, {5,10}}] 

	  k = input:size(1)
	  off = torch.ceil(torch.abs(0.6*k))
	  --create actual training datasets
	  	splitData.train_input = input[{{1, off}, {}}];
	  	splitData.train_out   =   out[{{1, off}, {}}];
	  	splitData.test_input  = input[{{off+1, k}, {}}];
	  	splitData.test_out	  = out[{{off+1, k}, {}}];

	  width       = splitData.train_input:size(2)
	  height      = splitData.train_input:size(2)
	  ninputs     = 3; nhiddens = 6;  noutputs = 6; nhiddens_rnn = 6 
	end
	return splitData
end

function batchNorm(x, N)
  --apply batch normalization to training data 
  -- http://arxiv.org/pdf/1502.03167v3.pdf
  local eps = 1e-5
  local momentum = 1e-1
  local affine = true
  local BN = nn.BatchNormalization(N, eps, momentum, affine)

  if type(x) == 'userdata' then       --inputs
   if opt.batchNorm then x  = BN:forward(x) end
    x  = transfer_data(x) --forward doubleTensor as CudaTensor
  elseif type(x) == 'table' then
    for i = 1, #x do
      if opt.batchNorm then  x[i] = BN:forward(x[i]) end
      x[i] = transfer_data(x[i])
    end
  end  
  collectgarbage()    
  return x
end

function get_datapair(args)	
	local inputs, targets = {}, {}
	local test_inputs, test_targets = {}, {}

	local splitData = {}
	splitData = split_data(args)	
	local testHeight = splitData.test_out[1]:size(1)
	if (args.data=='softRobot') then  --SIMO dataset

		offsets 	= torch.LongTensor(args.batchSize):random(1,height)  
		test_offsets = torch.LongTensor(args.batchSize):random(1,testHeight) 
		 -- 1. create a sequence of rho time-steps
		inputs 		= splitData.train_input:index(1, offsets)
		test_inputs = splitData.test_input:index(1, test_offsets)

		offsets 	 = torch.LongTensor():resize(offsets:size()[1]):copy(offsets)		
		test_offsets = torch.LongTensor():resize(test_offsets:size()[1]):copy(test_offsets)

		--batch of targets
		targets 	 = {splitData.train_out[1]:index(1, offsets), splitData.train_out[2]:index(1, offsets), 
		                  splitData.train_out[3]:index(1, offsets), splitData.train_out[4]:index(1, offsets), 
		                  splitData.train_out[5]:index(1, offsets), splitData.train_out[6]:index(1, offsets)}		
		test_targets = {splitData.test_out[1]:index(1, test_offsets), splitData.test_out[2]:index(1, test_offsets), 
		                 splitData.test_out[3]:index(1, test_offsets), splitData.test_out[4]:index(1, test_offsets), 
		                 splitData.test_out[5]:index(1, test_offsets), splitData.test_out[6]:index(1, test_offsets)}

		--increase offsets indices by 1      
		offsets:add(1) -- increase indices by 1
		test_offsets:add(1)
		offsets[offsets:gt(height)] = 1  
		test_offsets[test_offsets:gt(testHeight)] = 1

		--pre-whiten the inputs and outputs in the mini-batch
		local N = 1
		inputs = batchNorm(inputs, N)
		targets = batchNorm(targets, N)

		test_inputs = batchNorm(test_inputs, N)
		test_targets = batchNorm(test_targets, N)

	elseif (args.data == 'glassfurnace') then   --MIMO Dataset
		offsets = torch.LongTensor(args.batchSize):random(1,height)  
		test_offsets = torch.LongTensor(args.batchSize):random(1,testHeight) 
		
		-- print('train_inputs', splitData.train_input)
		--recurse inputs and targets into one long sequence
		inputs = 	splitData.train_input:index(1, offsets)		
		test_inputs =  splitData.test_input:index(1, test_offsets) 

		--batch of targets
		targets = 	splitData.train_out:index(1, offsets)        
		test_targets =   splitData.test_out:index(1, test_offsets)  
		                  
		--pre-whiten the inputs and outputs in the mini-batch
		inputs = batchNorm(inputs, 3)
		targets = batchNorm(targets, 6)

		test_inputs = batchNorm(test_inputs, 3)		
		test_targets = batchNorm(test_targets, 6)

		--increase offsets indices by 1      
		offsets:add(1) -- increase indices by 1
		test_offsets:add(1)
		offsets[offsets:gt(height)] = 1  
		test_offsets[test_offsets:gt(testHeight)] = 1
		
	--ballbeam and robotarm are siso systems from the DaiSy dataset
	elseif (string.find(args.data, 'robotArm')) or (string.find(args.data, 'ballbeam')) then
	  -- data_path_printer(data)
	  local testHeight = splitData.test_input:size(1)
	  offsets = torch.LongTensor(args.batchSize):random(1,height)  
	  test_offsets = torch.LongTensor(args.batchSize):random(1,testHeight) 

	  --recurse inputs and targets into one long sequence
	  inputs = {	splitData.train_input:index(1, offsets)		}
	  test_inputs = { splitData.test_input:index(1, test_offsets) }

	  --batch of targets
	  targets = {	splitData.train_out[1]:index(1, offsets)        }
	  test_targets = {   splitData.test_out[1]:index(1, test_offsets)  }

	  --pre-whiten the inputs and outputs in the mini-batch
	  inputs = batchNorm(inputs, 1)
	  targets = batchNorm(targets, 1)

	  test_inputs = batchNorm(test_inputs, 1)		
	  test_targets = batchNorm(test_targets, 1)

	  --increase offsets indices by 1      
	  offsets:add(1) -- increase indices by 1
	  test_offsets:add(1)
	  offsets[offsets:gt(height)] = 1  
	  test_offsets[test_offsets:gt(testHeight)] = 1

	end
return inputs, targets, test_inputs, test_targets
end