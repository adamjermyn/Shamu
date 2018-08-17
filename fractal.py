'''
This file contains the likelihood and prior functions for comparing
spectra against that due to a fractal random process.

The process takes as input three parameters:
	- amp
		The amplitude. We pick as a prior the log-uniform distribution from 1e-8 to 1e12.
	- alpha
		The exponent. Must be negative. We give the range [-4,0].
	- beta
		An offset. Must be positive. We give the range [0-10].
'''

import numpy as np
from numpy import log, pi, cos

ranges = [(-8,12), (0,4), (0,10), (-8,12)]

def FractalSpectrum(freqs, amp, alpha, beta):
	return 10**amp * (beta**2 + freqs**(-alpha))

def makeLogLike(freqs, spectrum):
	'''
	Given a spectrum, this method returns the method which evaluates
	the log likelihood of the model as a function of the model parameters.
	'''

	def Loglike(args):
		fractal = FractalSpectrum(freqs, *(args[:-1]))
		return -0.5 * np.sum((fractal - spectrum)**2 / (2*fractal*10**args[-1])) - 0.5 * log(2*pi) - np.sum(log(fractal*10**args[-1]))

	return Loglike
