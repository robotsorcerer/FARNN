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