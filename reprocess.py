import numpy as np
import os

globalDir = '../dataSingleLowPass/'
opts = os.listdir(globalDir)
dirs = [i for i in opts if not os.path.isfile(os.path.join(globalDir,i))]

for directory in dirs:
	print(directory)
	names = os.listdir(globalDir + directory)
	fNames = [globalDir + directory + '/' + f for f in names]

	fs = []
	ss = []

	for i,fname in enumerate(fNames):
		data = np.load(fname)
		np.savetxt(fname, data)