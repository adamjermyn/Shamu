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

import matplotlib.pyplot as plt
from scipy.stats import norm

fig = plt.figure(figsize=(4,4))
ax = fig.subplots(1,1)
#ax = ax[0]
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
#ax.yaxis.set_ticks_visible(False)
ax.xaxis.set_ticks_position('bottom')
#ax.yaxis.set_visible(False)
ax.set_ylabel('Probability Density')

nonmusic = ['Adelie Penguin', 'Field Cricket', 'Frog', 'Green-Rumped Parrotlet', 'Ryukyu Scops-Owl']
x = np.linspace(0,1.5,num=3000,endpoint=True)
labeled = [0,0]
for o in out:
	if 'Bach' not in o[0]:
		if o[0] in nonmusic:
			color = 'r'
			label = 'Non-Musical'
			labeled[0] += 1
		else:
			color = 'b'
			label = 'Musical'
			labeled[1] += 1

		if color == 'r' and labeled[0] < 2:
			ax.fill_between(x, 0*x, norm.pdf((x - o[1]) / o[3]), label=label, alpha=0.4, color=color)
		elif color == 'b' and labeled[1] < 2:
			ax.fill_between(x, 0*x, norm.pdf((x - o[1]) / o[3]), label=label, alpha=0.1, color=color)
		elif color == 'r':
			ax.fill_between(x, 0*x, norm.pdf((x - o[1]) / o[3]), alpha=0.4, color=color)
		else:
			ax.fill_between(x, 0*x, norm.pdf((x - o[1]) / o[3]), alpha=0.1, color=color)

ax.legend()
ax.set_xlabel('$\\beta$')
plt.tight_layout()
plt.savefig('Output/hist.pdf')

