from multinestWrapper import run, analyze, plot1D, plot2D
import fractal
import autoregress
import pickle
import numpy as np
import os
from mpi4py import MPI

def infer(frequencies, spectrum, name):

	# Book-keeping
	dir_path = '/data/vault/asj42/'
	oDir = dir_path + 'O/' + name + '/'
	oDir = os.path.abspath(oDir)

	# Setup fractal model

	# Setup likelihood
	loglike = fractal.makeLogLike(frequencies, spectrum)

	parameters = ['$A$', '$\\beta$', '$\\gamma$', '$\\kappa$']
	ranges = fractal.ranges
	ndim = len(ranges)

	for i,p in enumerate(parameters):
		print(i, p)

	# Run
	oPref = 'f'
	run(oDir, oPref, ranges, parameters, loglike)
	if MPI.COMM_WORLD.Get_rank() == 0:
		# Analyze
		fractalOut = analyze(oDir, oPref, oDir, oPref)

		# Plot and dump
		import pickle
		pickle.dump(fractalOut, open(oDir+'/' + 'analyzerF.pickle','wb'))

		plot1D(fractalOut[0], parameters, oDir, oPref)
		plot2D(fractalOut[0], parameters, oDir, oPref)

	# Setup autoregress model

	# Setup likelihood
	loglike = autoregress.makeLogLike(frequencies, spectrum)

	parameters = ['$A$', '$\\tau$', '$\\phi$', '$\\theta$', '$\\kappa$']
	ranges = autoregress.ranges
	ndim = len(ranges)

	for i,p in enumerate(parameters):
		print(i, p)

	# Run
	oPref = 'a'
	run(oDir, oPref, ranges, parameters, loglike)

	if MPI.COMM_WORLD.Get_rank() == 0:
		# Analyze
		autoregressOut = analyze(oDir, oPref, oDir, oPref)

		# Plot and dump
		import pickle
		pickle.dump(autoregressOut, open(oDir+'/' + 'analyzerA.pickle','wb'))

		plot1D(autoregressOut[0], parameters, oDir, oPref)
		plot2D(autoregressOut[0], parameters, oDir, oPref)

		print('Done. Fractal logZ:', fractalOut[2], 'Autoregress logZ:', autoregressOut[2], 'Difference:', fractalOut[2] - autoregressOut[2])

def loader(fname):
	print(fname)
	data = np.loadtxt(fname)

	freqs = data[:,0]
	data = data[:,1]

	data = data[freqs < 1e1]
	freqs = freqs[freqs < 1e1]

	data = data[freqs > 1e-2]
	freqs = freqs[freqs > 1e-2]

	return freqs, data

globalDir = '../dataSingleLowPass/'
opts = os.listdir(globalDir)
dirs = [i for i in opts if not os.path.isfile(os.path.join(globalDir,i))]
dirs = dirs[::-1]

for directory in dirs:
	print(directory)
	names = [n for n in os.listdir(globalDir + directory) if n[0] != '.']
	fNames = [globalDir + directory + '/' + f for f in names if f[0] != '.']
	
	names = names[::-1]
	fNames = fNames[::-1]

	fs = []
	ss = []

	for i,fname in enumerate(fNames):
		frequencies, spectrum = loader(fname)
		fs.append(frequencies)
		ss.append(spectrum)
		name = (directory + '_' + names[i][:names[i].index('.')]).replace('_44k','')

		dir_path = '/data/vault/asj42/'
		oDir = dir_path + 'O/' + name + '/'
		oDir = os.path.abspath(oDir)
		print(fname, name)
		infer(frequencies, spectrum, name)
