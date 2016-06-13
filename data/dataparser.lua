require 'torch'

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
	if (args.data=='soft_robot.mat') then  --SIMO dataset
		offsets = torch.LongTensor(args.batchSize):random(1,height)  
		 -- 1. create a sequence of rho time-steps
		inputs = train_input:index(1, offsets)
		offsets = torch.LongTensor():resize(offsets:size()[1]):copy(offsets)

		--batch of targets
		targets = {train_out[1]:index(1, offsets), train_out[2]:index(1, offsets), 
		                  train_out[3]:index(1, offsets), train_out[4]:index(1, offsets), 
		                  train_out[5]:index(1, offsets), train_out[6]:index(1, offsets)}
		
		--increase offsets indices by 1      
		offsets:add(1) -- increase indices by 1
		offsets[offsets:gt(height)] = 1  

		--pre-whiten the inputs and outputs in the mini-batch
		local N = 1
		inputs = batchNorm(inputs, N)
		targets = batchNorm(targets, N)
	elseif (args.data == 'glassfurnace.mat') then   --MIMO Dataset
		offsets = torch.LongTensor(args.batchSize):random(1,height)  

		--recurse inputs and targets into one long sequence
		inputs = --nn.JoinTable(1):forward
				torch.cat({train_input[1]:index(1, offsets), train_input[2]:index(1, offsets), 
					train_input[3]:index(1, offsets)})
		--batch of targets
		targets =-- nn.JoinTable(1):forward
						{train_out[1]:index(1, offsets), train_out[2]:index(1, offsets), 
		                  train_out[3]:index(1, offsets), train_out[4]:index(1, offsets), 
		                  train_out[5]:index(1, offsets), train_out[6]:index(1, offsets)}
		--pre-whiten the inputs and outputs in the mini-batch
		local N = 10
		inputs = batchNorm(inputs, N)
		targets = batchNorm(targets, N)
		-- inputs:resize(300*10)
		-- targets:resize(600*10)

		print('inputs', inputs)
		print('outputs', targets)

	--ballbeam and robotarm are siso systems from the DaiSy dataset
	elseif (string.find(data, 'robotArm.mat')) or (string.find(data, 'ballbeam.mat')) t$
	  data_path_printer(data)

	  data = matio.load(data)
	  data = data.filename;
	  input = data[{{}, {1}}]
	  out = data[{{}, {2}}]

	  k = input:size(1)
	  off = torch.ceil(torch.abs(0.6*k))

	  train_input = input[{{1, off}, {1}}]
	  train_out = {out[{{1, off}, {}}]}

	  --create testing data
	  test_input = input[{{off + 1, k}, {1}}]
	  test_out   = {  out[{{off+1, k}, {1}}] }


	end
return inputs, targets
end

local function data_path_printer(x)  
  print(sys.COLORS.green .. string.format("you have specified the data path %s", x))
end

local function get_filename(x)
  -- print('x:match', x:match("^.+$"), x:match("(%a+)"))
  return x:match("(%a+)")
end

function split_data(opt)
	
	local filename = get_filename(opt.data)  -- we strip the filename extention from the data
	--ballbeam and robotarm are siso systems from the DaiSy dataset
	if (string.find(data, 'robotArm.mat')) or (string.find(data, 'ballbeam.mat')) then  
	  data_path_printer(data)

	  data = matio.load(data)
	  data = data.filename;
	  input = data[{{}, {1}}]     
	  out = data[{{}, {2}}]

	  k = input:size(1)
	  off = torch.ceil(torch.abs(0.6*k))

	  train_input = input[{{1, off}, {1}}]
	  train_out = {out[{{1, off}, {}}]}

	  --create testing data
	  test_input = input[{{off + 1, k}, {1}}]
	  test_out   = {  out[{{off+1, k}, {1}}] }  

	--SIMO System from my soft_robot system
	elseif(string.find(data, 'soft_robot.mat')) then  
	  data_path_printer(data);   data = matio.load(data);   data = data.pose
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

	  train_input = input[{{1, off}, {1}}]   -- order must be preserved. cuda tensor does not support csub yet
	  train_out   = { 
	                 out[1][{{1, off}, {1}}], out[2][{{1, off}, {1}}],            -- most of the work is done here              (out[{{1, off}, {1}}])/10, outlln[{{1, off}, {1}}], 
	                 out[3][{{1, off}, {1}}], out[4][{{1, off}, {1}}],
	                 out[5][{{1, off}, {1}}], out[6][{{1, off}, {1}}],
	                } 
	  --create testing data
	  test_input = input[{{off + 1, k}, {1}}]
	  test_out   = {
	                 out[1][{{off+1, k}, {1}}], out[2][{{off+1, k}, {1}}], 
	                 out[3][{{off+1, k}, {1}}], out[4][{{off+1, k}, {1}}], 
	                 out[5][{{off+1, k}, {1}}], out[6][{{off+1, k}, {1}}] 
	                }  

	  width       = train_input:size(2)
	  height      = train_input:size(1)
	  ninputs     = 1
	  noutputs    = 6
	  nhiddens_rnn = 6 
	-- MIMO dataset from the Daisy  glassfurnace dataset (3 inputs, 6 outputs)
	elseif (string.find(data, 'glassfurnace.mat')) then
	  data_path_printer(data);  data = matio.load(data)  ;   data = data[filename];
	  -- three inputs i.e. heating input, cooling input, & heating input
	  input =   data[{{}, {2, 4}}]:resize(3, data:size(1), data:size(2))   
	  -- six outputs from temperatures sensors in a cross sectiobn of the furnace          
	  out =   data[{{}, {5,10}}]:resize(6, data:size(1), data:size(2))
	  print('input', input:size())
	  print('out', out:size())

	  k = input[1]:size(1)
	  off = torch.ceil(torch.abs(0.6*k))

	  -- allocate storage for tensors
	  train_input = torch.DoubleTensor(3, off, data:size(2))
	  train_out   = torch.DoubleTensor(6, off, data:size(2))
	  test_input  = torch.DoubleTensor(3, k-off, data:size(2))
	  test_out    = torch.DoubleTensor(6, k-off, data:size(2))

	  --create actual training datasets
	  for i=1, train_input:size(1) do
	    train_input[i] = input[i]:sub(1, off) 
	  end
	  
	  for i=1, train_out:size(1) do
	    train_out[i] = out[i]:sub(1, off)
	  end

	  --create validation/testing data
	  for i = 1, test_input:size(1) do
	    test_input[i] = input[i]:sub(off + 1, k)
	  end

	  for i=1,test_out:size(1) do
	    test_out[i] = out[i]:sub(off + 1, k)
	  end

	  -- print('test_input:size(), test_out:size()')
	  -- print(test_input:size(), test_out:size())
	  width       = train_input:size(2)
	  height      = train_input:size(2)
	  ninputs     = 3
	  noutputs    = 6
	  nhiddens_rnn = 6 

	end
end