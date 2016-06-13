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
	if (args.data=='soft_robot.mat') then 
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
	elseif (args.data == 'glassfurnace.mat') then 
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
	end
return inputs, targets
end
