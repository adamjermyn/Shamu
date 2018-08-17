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

	if 'Humpback Whale' in species:

		try:
			spectrum = np.loadtxt(spath + species + ' - ' + latin + '/' + idd + '_44k.wav.16.wav')
		except:
			spectrum = np.loadtxt(spath + species + ' - ' + latin + '/' + idd + '.wav.16.wav')

		freqs, spec = spectrum[:,0], spectrum[:,1]

		plt.figure(figsize=(4,4))
		plt.plot(freqs, spec)
		plt.title(species)
		plt.xlabel('$f (\\mathrm{Hz})$')
		plt.ylabel('$\\log P (A.U.)$')
		plt.xscale('log')
		plt.yscale('log')
		plt.tight_layout()
		plt.savefig('Output/' + species + '_' + recording + '_highF.pdf')