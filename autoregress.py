'''
This file contains the likelihood and prior functions for comparing
spectra against that due to a first-order autoregressive random process.

The process takes as input three parameters:
	- amp
		The amplitude. We pick as a prior the log-uniform distribution from 1e-8 to 1e12.
	- phi
		physically must be in the range 0-1, as negative values produce
		small-scale oscillations which are sensitive to the step scale
		while values greater than unity have no steady state solution.
	- theta
		Not constrained a priori. Generally expected to be of order unity, so
		we sample the range [-2,2].
'''

import numpy as np
from numpy import log, pi, cos

ranges = [(-8,12), (0, 100), (0,1), (-2,2), (-8,12)]

def ARMAspectrum(freqs, amp, tau, phi, theta):
	return 10**amp * (1 + 2*theta*cos(2*pi*freqs*tau) + theta**2) / (1 - 2*phi*cos(2*pi*freqs) + phi**2)

def makeLogLike(freqs, spectrum):
	'''
	Given a spectrum, this method returns the method which evaluates
	the log likelihood of the model as a function of the model parameters.
	'''

	def Loglike(args):
		arma = ARMAspectrum(freqs, *(args[:-1]))

		return - 0.5 * np.sum((arma - spectrum)**2 / (2*arma*10**args[-1])) - 0.5 * log(2*pi) - np.sum(log(arma*10**args[-1]))

	return Loglike
