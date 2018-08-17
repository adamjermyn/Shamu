import numpy as np
import json
import pickle
import os
from tabulate import tabulate
from collections import defaultdict
import pymultinest
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from autoregress import ARMAspectrum
from fractal import FractalSpectrum

path = '/Users/adamjermyn/WhalesOut/O/'
dirs = os.listdir(path)
dirs = list(d for d in dirs if not os.path.isfile(os.path.join(path,d)))

spath = '../dataSingleLowPass/'

compareZ = []
paramsA = []
paramsF = []

obsA = defaultdict(list)
obsF = defaultdict(list)

for d in dirs:

	if 'Bach' not in d:
		sp = d.rsplit('-',1)
	else:
		sp = [d[:4], d[4:]]

	species = sp[0].strip()
	recording = sp[1].strip()
	latin = recording.split('_')[0]
	idd = d.split('_')[-1]

	try:
		spectrum = np.loadtxt(spath + species + ' - ' + latin + '/' + idd + '_44k.wav.16.wav')
	except:
		spectrum = np.loadtxt(spath + species + ' - ' + latin + '/' + idd + '.wav.16.wav')

	freqs, spec = spectrum[:,0], spectrum[:,1]
	sel = (freqs < 10) & (freqs > 0.01)
	freqs = freqs[sel]
	spec = spec[sel]

	dataA = json.load(open(path + d + '/astats.json'))

	amp = dataA['marginals'][0]['median']
	tau = dataA['marginals'][1]['median']
	phi = dataA['marginals'][2]['median']
	theta = dataA['marginals'][3]['median']

	specA = ARMAspectrum(freqs, amp, tau, phi, theta)

	dataF = json.load(open(path + d + '/fstats.json'))

	amp = dataF['marginals'][0]['median']
	beta = dataF['marginals'][1]['median']
	gamma = dataF['marginals'][2]['median']

	specF = FractalSpectrum(freqs, amp, beta, gamma)

	plt.figure(figsize=(4,4))
	plt.plot(freqs, spec, label='Data')
	plt.plot(freqs, specA, label='ARMA')
	plt.plot(freqs, specF, label='Fractal')
	plt.title(species)
	plt.xlabel('$f (\\mathrm{Hz})$')
	plt.ylabel('$\\log P (A.U.)$')
	plt.xscale('log')
	plt.yscale('log')
	plt.legend()
	plt.tight_layout()
	plt.savefig('Output/' + species + '_' + recording + '.pdf')