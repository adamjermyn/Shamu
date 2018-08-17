import numpy as np
import json
import os
from tabulate import tabulate
from collections import defaultdict

path = '/Users/adamjermyn/WhalesOut/O/'
dirs = os.listdir(path)
dirs = list(d for d in dirs if not os.path.isfile(os.path.join(path,d)))

compareZ = []
paramsA = []
paramsF = []

obsA = defaultdict(list)
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
	q.append(dataA['nested sampling global log-evidence'])
	q.append(dataF['nested sampling global log-evidence'])

	if q[-2] > q[-1]:
		print(species, recording, recording_ID, q[-2] - q[-1])

	compareZ.append(q)

	q = []

	q.append(species)
	q.append(recording)
	q.append(recording_ID)
	q.append(dataA['marginals'][0]['median'])
	q.append(dataA['marginals'][0]['sigma'])
	q.append(dataA['marginals'][1]['median'])
	q.append(dataA['marginals'][1]['sigma'])
	q.append(dataA['marginals'][2]['median'])
	q.append(dataA['marginals'][2]['sigma'])
	q.append(dataA['marginals'][3]['median'])
	q.append(dataA['marginals'][3]['sigma'])
	q.append(dataA['marginals'][4]['median'])
	q.append(dataA['marginals'][4]['sigma'])

	obsA[species].append(q[3:])

	paramsA.append(q)

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
	
compareZ = sorted(compareZ, key=lambda x: x[0])
paramsA = sorted(paramsA, key=lambda x: x[0])
paramsF = sorted(paramsF, key=lambda x: x[0])

fi = open('../Proc RSB Submission/logZ.tex','w+')
fi.write(tabulate(compareZ, tablefmt='latex', floatfmt='.3f').split('\n',2)[2].rsplit('\n',1)[0])

fi = open('../Proc RSB Submission/paramsA.tex','w+')
fi.write(tabulate(paramsA, tablefmt='latex', floatfmt='.3f').split('\n',2)[2].rsplit('\n',1)[0])
fi.close()


fi = open('../Proc RSB Submission/paramsF.tex','w+')
fi.write(tabulate(paramsA, tablefmt='latex', floatfmt='.3f').split('\n',2)[2].rsplit('\n',1)[0])
fi.close()

species = list(s[0] for s in compareZ)
from collections import OrderedDict
species = list(OrderedDict.fromkeys(species))
netZ = list((s,sum(z[3] for z in compareZ if z[0] == s)-sum(z[4] for z in compareZ if z[0] == s)) for s in species)
#netZ = list((z[0], "{:.2e}".format(float(z[1])), "{:.2e}".format(float(z[2]))) for z in netZ)

print(tabulate(netZ, tablefmt='latex'))

# Compute population estimates

meansA = {}
stdsA = {}

for s in obsA.keys():
	for i in range(3):
		obs = np.array(obsA[s])
		siginv = 1 / np.sum(obs[:,2*i+1]**(-2))
		mean = np.sum(obs[:,2*i]/obs[:,2*i+1]**2)
		mean *= siginv
		std = np.sqrt(siginv)
		meansA[(s,i)] = mean
		stdsA[(s,i)] = std

meansF = {}
stdsF = {}
sampleVF = {}

for s in obsA.keys():
	for i in range(3):
		obs = np.array(obsF[s])
		siginv = 1 / np.sum(obs[:,2*i+1]**(-2))
		mean = np.sum(obs[:,2*i]/obs[:,2*i+1]**2)
		mean *= siginv
		svf = np.sum((obs[:,2*i] - mean)**2 / obs[:,2*i+1]**2) * siginv
		std = np.sqrt(siginv)
		meansF[(s,i)] = mean
		stdsF[(s,i)] = std
		sampleVF[(s,i)] = svf

out = []
for s in meansF.keys():
	if s[1] == 1:
		out.append([s[0], round(meansF[s],2), round(stdsF[s],3), round(sampleVF[s],3)])

out = sorted(out, key=lambda x: x[0])

print(tabulate(out, tablefmt='latex'))

out = []
for s in meansF.keys():
	if s[1] == 1:
		obs = np.array(obsF[s[0]])
		mest = np.sum((obs[:,2] - meansF[s])**2/(obs[:,3]**2 + stdsF[s]**2)) / (len(obs) - 1)
		out.append([s[0], mest])

out = sorted(out, key=lambda x: x[0])

print(tabulate(out, tablefmt='latex'))

# Compute ks test
from scipy.stats import kstest

for s in species:
	obs = np.array(obsF[s])
	siginv = 1. / sampleVF[(s,1)]
	mean = meansF[(s,1)]
	rvs = (obs[:,2] - mean) * siginv
	cdf = 'norm'
	d, p = kstest(rvs, cdf)
	print(s,'&$', round(np.log10(p),2),'$\\\\')



# Plot F parameters
import matplotlib.pyplot as plt

for s in species:
	fig = plt.figure(figsize=(5,4))
	for p in paramsF:
		if p[0] == s:
			plt.errorbar(p[5],p[7],xerr=p[6],yerr=p[8])
	plt.title(s)
	plt.xlabel('$\\beta$')
	plt.ylabel('$\\gamma$')
	plt.savefig('Output/' + s + '.pdf')
	plt.clf()
