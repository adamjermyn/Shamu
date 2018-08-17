from multinestWrapper import run, analyze, plot1D, plot2D
import fractal
import autoregress
import pickle
import numpy as np
import os
from mpi4py import MPI
from scipy.misc import logsumexp

# For computing gaussian priors
def gaussianLogLike(x, mu, sigma):

	# There's no other term because we explicitly handle the log of the prefactor.
	y = (x - mu) / sigma
	return -0.5 * np.log(2*np.pi*sigma) - y**2/2

# For excluding duplicates
def indicator(y):
	isSorted = (y == np.sort(y)).all()

	if isSorted:
		return 0
	else:
		return -1e50

def infer(x, dx, num, name):
	# Fits a multimodal gaussian to the observed points.
	# The number of modes is given by num.

	# Book-keeping
	dir_path = '/Users/adamjermyn/ModesOut/'
	oDir = dir_path + '/' + name + '_' + str(num) + '/'
	oDir = os.path.abspath(oDir)	

	# Setup likelihood
	p0 = ['$\\mu_{' + str(i) + '}$' for i in range(num)]
	p1 = ['$\\sigma_{' + str(i) + '}$' for i in range(num)]
	p2 = ['$A_{' + str(i) + '}$' for i in range(num)]
	params = list(val for pair in zip(p0,p1,p2) for val in pair)

	r0 = [(0,4) for i in range(num)]
	r1 = [(0,1) for i in range(num)]
	r2 = [(-5,5) for i in range(num)]
	ranges = list(val for pair in zip(r0,r1,r2) for val in pair)

	print(ranges)

	ndim = 2 * num

	x = np.array(x)
	dx = np.array(dx)

	def like(y):

		mus = np.array(y[::3])
		sigs = np.array(y[1::3])
		amps = np.array(y[2::3])
		amps -= logsumexp(amps)

		z = indicator(mus)

		zis = list(amps[i] + gaussianLogLike(x,mus[i],(dx**2 + sigs[i]**2)**0.5) for i in range(num))
		zis = np.array(zis)

		z += sum(logsumexp(zis,axis=0))

		return z


	# Run
	oPref = 'mode'

	run(oDir, oPref, ranges, params, like, log_zero=-1e40)

	if MPI.COMM_WORLD.Get_rank() == 0:
		# Analyze
		out = analyze(oDir, oPref, oDir, oPref)

		# Plot and dump
		import pickle
		pickle.dump(out, open(oDir+'/' + 'analyzer.pickle','wb'))

		plot1D(out[0], params, oDir, oPref)
		plot2D(out[0], params, oDir, oPref)

		return out[2]

# Load data

from collections import defaultdict, OrderedDict
import json

path = '/Users/adamjermyn/WhalesOut/O/'
dirs = os.listdir(path)
dirs = list(d for d in dirs if not os.path.isfile(os.path.join(path,d)))

paramsF = []

obsF = defaultdict(list)

for d in dirs:
	dataA = json.load(open(path + d + '/astats.json'))
	dataF = json.load(open(path + d + '/fstats.json'))

	if 'Bach' not in d:
		sp = d.rsplit('-',1)
	else:
		sp = [d[:4], d[4:]]

	species = sp[0].strip()
	recording = sp[1].strip()
	recording, recording_ID = sp[1].split('_')

	if species == 'Ryukyu Scops':
		species = 'Ryukyu Scops Owl'
		recording = 'Otus elegans'

	q = []

	q.append(species)
	q.append(recording)
	q.append(recording_ID)
	q.append(dataF['marginals'][0]['median'])
	q.append(dataF['marginals'][0]['sigma'])
	q.append(dataF['marginals'][1]['median'])
	q.append(dataF['marginals'][1]['sigma'])
	q.append(dataF['marginals'][2]['median'])
	q.append(dataF['marginals'][2]['sigma'])
	q.append(dataF['marginals'][3]['median'])
	q.append(dataF['marginals'][3]['sigma'])

	obsF[species].append(q[3:])

	paramsF.append(q)
	
paramsF = sorted(paramsF, key=lambda x: x[0])

species = list(s[0] for s in paramsF)
species = list(OrderedDict.fromkeys(species))


summary = []
for s in species:
	x = []
	dx = []
	for p in paramsF:
		if p[0] == s:
			x.append(p[5])
			dx.append(p[6])

	zs = list((i,infer(x, dx, i, s)) for i in range(1,6))
	zs = [s] + zs

	print('OUT:',zs)
	summary.append(zs)

print('OUT:',summary)