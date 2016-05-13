-- Table to pre-process raw data against outliers, high frequencies etc
-- The following are from cs231n.github.io --batch normalization from karpathy
stats={}

-- Get the mean value of a table
function stats.mean( t )
  local meanTab = {}
  for k,v in ipairs(t) do
      meanTab[k] = torch.mean(v)
  end   
  return meanTab
end

function stats.submeans( t )
  local mean = {}
  mean = stats.mean(t)
  for i = 1, #t do
    t[i]:csub(mean[i])
  end

  return t
end

-- Get the standard deviation of a table
function stats.standardDeviation( t )
  local vm = {}
  local t_stddev = {}
  local result = {}

  --mean shift t
  vm = stats.submeans(t)

  for i = 1, #t do
    t_stddev[i] = torch.std(vm[i]) 
  end

  if opt.print then print('stddev out', t_stddev) end
  return t_stddev
end

-- Get the max and min for a table
function stats.maxmin( t )
  local max = -math.huge
  local min = math.huge

  for k,v in pairs( t ) do
    if type(v) == 'number' then
      max = math.max( max, v )
      min = math.min( min, v )
    end
  end

  return max, min
end

function stats.normalize (t)
--  first find the mean and mean-centered data in one go
  local submeans, divisor, results, stddev = {} , {}, {}, {}

  submeans = stats.submeans(t)   
  stddev = stats.standardDeviation(t)   -- returns the tables of standard deviations
  --normalize the data with the standard deviation to zero-normalize the data
  for k = 1, #submeans do    
    divisor[k] = stddev[k]
    for j = 1, submeans[1]:size(1) do
      submeans[k][j] = submeans[k][j]/divisor[k]
    end
  end
  if opt.print then print('submeans', submeans) end

if opt.plot then
  local xaxis = torch.linspace(1, submeans[1]:size(1), submeans[1]:size(1))
  plotter = require 'gnuplot'

  plotter.title('Mean shifted output train set')
  plotter.plot({'train out, x', xaxis, submeans[1], '**'},
           {'train out, y', xaxis, submeans[2], '-*'}, {'train out, z', xaxis, submeans[3], '+'},
           {'train out, roll', xaxis, submeans[4], '~-'}, {'train out, pitch', xaxis, submeans[5], '-~'},
           {'train out, yaw', xaxis, submeans[6], '*-'})
end

  return submeans
end

function stats.inputnorm(t)
  t:csub(torch.mean(t))             --subtract the input means
  local t_stddev = torch.std(t)           --find the input standard deviation
  local result = t/t_stddev         -- ratio of sum of squares to std

  if opt.print then print('inputs', result) end

  if(opt.plot) then
    local xaxis = torch.linspace(1, kk, kk)

    plt.title('Mean shifted output train input')
    plt.plot({'train input', xaxis, train_input, '~'})
  end

  return result
end