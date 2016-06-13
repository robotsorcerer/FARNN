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
	  data_path_printer(filenamefull)
	--ballbeam and robotarm are siso systems from the DaiSy dataset
	if (string.find(filename, 'robotArm')) or (string.find(filename, 'ballbeam')) then  
	  data_path_printer(filenamefull)

	  data = matio.load(filenamefull)
	  data = data[filename];
	  input = data[{{}, {1}}]     
	  print(input)
	  out = data[{{}, {2}}]

	  k = input:size(1)
	  off = torch.ceil(torch.abs(0.6*k))

	  train_input = input[{{1, off}, {1}}]
	  train_out = {out[{{1, off}, {}}]}

	  --create testing data
	  test_input = input[{{off + 1, k}, {1}}]
	  test_out   = {  out[{{off+1, k}, {1}}] }  

	--SIMO System from my soft_robot system
	elseif(string.find(filename, 'softRobot')) then 
	  data_path_printer(filenamefull);   data = matio.load(filenamefull);  
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
	elseif (string.find(filename, 'glassfurnace')) then
	  data_path_printer(filenamefull);  data = matio.load(filenamefull)  ;   data = data[filename];
	  -- three inputs i.e. heating input, cooling input, & heating input
	  input =   data[{{}, {2, 4}}] --:resize(3, data:size(1), data:size(2))   

	  print('input', input[])
	  -- six outputs from temperatures sensors in a cross sectiobn of the furnace          
	  out =   data[{{}, {5,10}}]:resize(6, data:size(1), data:size(2))

	  k = input[1]:size(1)
	  off = torch.ceil(torch.abs(0.6*k))

	  print('data', data:size())
	  -- allocate storage for tensors
	  train_input = torch.DoubleTensor(3, off, 1)
	  train_out   = torch.DoubleTensor(6, off, 1)
	  test_input  = torch.DoubleTensor(3, k-off, 1)
	  test_out    = torch.DoubleTensor(6, k-off, 1)

	  print('train_input', train_input[1]:size())
	  --create actual training datasets
	  for i=1, train_input:size(1) do
	    train_input[i] = input[i]:sub(1, off) 
	  end

	  print('train_input after sub', train_input[1])
	  
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

	  width       = train_input:size(2)
	  height      = train_input:size(2)
	  ninputs     = 3
	  noutputs    = 6
	  nhiddens_rnn = 6 
	end
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
	if (args.data=='softRobot') then  --SIMO dataset
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

		print('targets', targets)

	elseif (args.data == 'glassfurnace') then   --MIMO Dataset
		offsets = torch.LongTensor(args.batchSize):random(1,height)  

		--recurse inputs and targets into one long sequence
		inputs = --nn.JoinTable(1):forward
				-- torch.cat(
					{train_input[1]:index(1, offsets), train_input[2]:index(1, offsets), 
					train_input[3]:index(1, offsets)}
					-- )
		--batch of targets
		targets =-- nn.JoinTable(1):forward
						{train_out[1]:index(1, offsets), train_out[2]:index(1, offsets), 
		                  train_out[3]:index(1, offsets), train_out[4]:index(1, offsets), 
		                  train_out[5]:index(1, offsets), train_out[6]:index(1, offsets)}

		--pre-whiten the inputs and outputs in the mini-batch
		local N = 10
		for i = 1, #inputs do
			inputs[i] = batchNorm(inputs[i], N)
		end
		for i=1,#targets do
			targets[i] = batchNorm(targets[i], N)
		end
		
		print('inputs', inputs)		
		print('targets', targets)

	--ballbeam and robotarm are siso systems from the DaiSy dataset
	elseif (string.find(args.data, 'robotArm')) or (string.find(dargs.ata, 'ballbeam')) then
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