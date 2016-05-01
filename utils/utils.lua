require 'torch'

function outResize(input, step)
	local _gradOutput = {}
	_gradOutputs[step] = input[step]:clone()
	print('gradOutputs[step]', gradOutputs[step])

end

function catOut(targets, step, noutputs, opt)
	local targets_, targsTab = {}, {}
	local targTable = {}
	targets_ = torch.cat({targets[step][1], targets[step][2], targets[step][3], 		
						   targets[step][4], targets[step][5], targets[step][6]})
	--print('targets_', targets_)
	--targets_ = torch.reshape(targets_, noutputs, noutputs)

	for i = 1, opt.batchSize do
		targsTab[i] = targets_[{{i}, {1, opt.batchSize}}]
		table.insert(targTable, targsTab[i])
		targTable[i] = torch.reshape(targTable[i], opt.batchSize, 1)
	end

	return targets_, targTable
end

function gradInputReshape(inputs, step, noutputs, opt)
	local inputer  = inputs[step]:expand(noutputs, noutputs)
	local inpuTab = {}
	for i = 1, opt.batchSize do
		inputer[i] = inputer[{{i}, {1, opt.batchSize}}]
		table.insert(inpuTab, inputer[i])
		inpuTab[i] = torch.reshape(inpuTab[i], 1, opt.batchSize)
	end

	return inpuTab
end

function gradOutputsReshape(gradOutputs, step, opt)
	local gradder, gradderTab = {}, {}
	for i = 1, opt.batchSize do		
		table.insert(gradderTab, gradOutputs[i])
		gradder[i] = torch.reshape(gradderTab[i], 1, opt.batchSize)
	end

	print('gradder'); 		print(gradder)
	print('gradderTab');	print(gradderTab)

	return gradder
end


--print a bunch of stuff if user enables print option
local function perhaps_print(q, qn, inorder, outorder, input, out, off, train_out, trainData)
  
  print('training_data', trainData)
  print('\ntesting_data', test_data)    

  --random checks to be sure data is consistent
  print('train_data_input', trainData[1]:size())  
  print('train_data_output', trainData[2])        
  print('\ntrain_xn', trainData[2][1]:size())  
  print('\ntrain_yn', trainData[2][2]:size()) 
  print('\ntrain_zn', trainData[2][3]:size())  
  print('\ntrain_roll', trainData[2][4]:size()) 
  print('\ntrain_pitch', trainData[2][5]:size())  
  print('\ntrain_yaw', trainData[2][6]:size()) 

  print('\ninput head', input[{ {1,5}, {1}} ]) 
  print('k', input:size()[1], 'off', off, '\nout\n', out, '\ttrain_output\n', train_out)
  print('\npitch head\n\n', out.zn[{ {1,5}, {1}} ])

  print('\nqn:' , qn)
  print('Optimal number of input variables is: ', torch.ceil(qn))
  print('inorder: ', inorder, 'outorder: ', outorder)
  print('system order:', inorder + outorder)

  --Print out some Lipschitz quotients (first 5) for user
  for ii, v in pairs( q ) do
    print('Lipschitz quotients head', ii, v)
    if ii == 5 then break end
  end
  --print neural net parameters
  print('neunet biases Linear', neunet.bias)
  print('\nneunet biases\n', neunet:get(1).bias, '\tneunet weights: ', neunet:get(1).weights)

  
  print('inputs: ', inputs)
  print('targets: ', targets)
end