import numpy as np
import math

# prints formatted price
def formatPrice(n):
	return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))

# returns the vector containing stock data from a fixed file
def getStockDataVec(key):
	vec = []
	lines = open("data/" + key + ".csv", "r").read().splitlines()

	for line in lines[1:]:
		vec.append(float(line.split(",")[4]))

	return vec

# returns the sigmoid
def sigmoid(x):
	return 1 / (1 + math.exp(-x))

# returns an an n-day state representation ending at time t
def getState(data, t, n):
	d = t - n + 1
	block = data[d:t + 1] if d >= 0 else np.append(-d * [data[0]], data[0:t + 1]) # pad with t0
	#print(block)
	res = []
	for i in range(n - 1):
		block_dif = block[i + 1] - block[i]
		#block_dif = (block[i+1] - np.nanmean(block))/np.var(block) #* 100 # 10 #(10**(np.max([1,len(str(int(reward)))-2])))
		block_dif = block_dif / 30 
		
		#block_dif = block_dif / 50 
		#if i <= 10: print('Blockdif:' + str(block_dif) )
		if block_dif < -100:
			res.append(0.0)
		elif block_dif > 100:
			res.append(1.0)
		else:
			res.append(sigmoid(block_dif  ))
	
	#print('Min:' + str(np.min(res)))
	#print('Mean: ' + str(np.mean(res)))
	#print('Max: ' + str(np.max(res)))
	return np.array([res])
