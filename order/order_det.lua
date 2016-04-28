--------------------------------------------------------------------------------------------------------------
--[[Compute the Lipschitz quotients and Estimate Model order

This function implements He and Asada's order selection algorithm as enumerated in their 1993 paper:
"A New Method for Identifying Orders of Input-Output Models for Nonlinear Dynamic Systems" {MIT}]]
--------------------------------------------------------------------------------------------------------------

require 'torch'
if use_cuda then
	require 'cutorch'
end

order_det = {}
print('Determinining input-output order determination using He and Asada\'s method')
--------------------------------------------------------------------------------------------------------------
--Section 3.2, He and Asada: Find Optimal number of input parameters 
--------------------------------------------------------------------------------------------------------------
local qn = {}
function computeqn(u_off, y_off)
	local p = torch.ceil(0.02 * off)       --parameter p that determines number of iterations
	local qinner = {}
	for i = 1, p do
		local qk = {}
		--hack to use pitch of head as output only
		--y_off = y_off[3]
		-- print('y_off', y_off)
		qk[i]		 = torch.abs(y_off[i] - y_off[i + 1])  / torch.norm(u_off[i] - u_off[i + 1]) 
		--print('qk[i', qk[i], 'sqrt(i)', torch.sqrt(i), 'p', p)
		qinner[i] 	 = qk[i] * torch.sqrt(i) 
		qbrace       = torch.cumprod(qinner[i])		
	end		
		--print('qbrace', qbrace)
		qn         	 = torch.pow(qbrace, 1/p)
		--print('qbrace', qbrace:size())
		--print('off', off)
	return qn
end
-- qn is not needed anymore
qn = nil
collectgarbage()
--------------------------------------------------------------------------------------------------------------
--[[Compute the Lipschitz quotients and Estimate Model order

This function implements He and Asada's order selection algorithm as enumerated in their MIT 1993 paper:
"A New Method for Identifying Orders of Input-Output Models for Nonlinear Dynamic Systems"]]
--------------------------------------------------------------------------------------------------------------
local inorder, outorder, q

function computeq (u_off, y_off, opt)
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
		if use_cuda then
			m_epstensor = m_epstensor:cuda()
		end
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
			q[n[i]]   = torch.abs(y_off[i] - y_off[i - m[i]*tau]) / torch.norm(x[i] - x[i - tau])
			local delta = q[i] - q[i - tau]
			if use_cuda then 
				zero = torch.CudaTensor({0})
				nut5 = torch.CudaTensor({-0.5})
			else
				zero = torch.Tensor({0}); torch.Tensor({-0.5})
			end
			--establish the inequality that makes input order determination possible
			if torch.lt(delta, zero ) and torch.lt(delta, nut5) then                      -- this is when q(i) is sig. smaller than q(i-1)
				i = i + 1
				n[i]	   = i;        m[i]  = n[i] - m[i - 1]  ; l[i] = n[i] - m[i]  ;
				x[i] = u_off[i - m[i] * tau]							  		-- select next x
				q[i] = torch.abs(y_off[i] - y_off[i - m[i] * tau]) / torch.norm(x[i] - x[i - m[i] * tau])  -- this is q(2+2)
				--print('q[2 + 2] ', q[i])
				qlt = {}  qgt = {}
				qlt[i]  = q[i - 1] - opt.l_eps		--  lower bound on l stopping criterion
				qgt[i]  = q[i - 1] + opt.l_eps		--	upper bound on l stopping criterion

				--Create inequality q(m+l) - epsilon < q(m + l) < q(m + l - 1) + epsilon
				if torch.gt(q[i], qlt[i]) and torch.lt(q[i], qgt[i]) then --i.e. m_eps - 0.05 < q < m_eps + 0.05
					inorder = l[n[i] - m[i]]
				else
					break
				end
			end
		until torch.gt(q[i], qlt[i]) and torch.lt(q[i], qgt[i]) 
		
    return inorder, outorder, q
end
--not needed anymore
inorder, outorder, q = nil
collectgarbage()