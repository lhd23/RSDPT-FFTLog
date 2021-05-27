from __future__ import print_function
import numpy as np
try:
	import pandas as pd
except:
	raise ImportError('pandas not found')

import os
dir = os.path.dirname(os.path.abspath(__file__))

def get_fftlog_param_dict(kernel='2K22'):
	"""
	Valid kernels are:

		2K22 --> 2[K^s]^2 ('22')
		4KG22 -> 4 K^s G^s_2 ('22')
		KG13 --> K^s G^s_2 ('13')
		2G22 --> 2[G^s_2]^2 ('22')
		G13 ---> G^s_3 ('13')
		KB22 --> K^s_B ('22')
		2F22 --> 2[F^s_2]^2 ('22')
		F13 ---> F^s_3 ('13')
		KF22 --> K^s_F ('22')
	"""
	if kernel not in ['2K22', '4KG22', 'KG13', '2G22', 'G13', 'KB22', '2F22', 'F13', 'KF22']:
		raise ValueError('kernel invalid: {}'.format(kernel))

	df = pd.read_csv(dir+'/params-sym-all.csv')
	n1_all = df['n1'].values
	n2_all = df['n2'].values

	d = {
			'2K22':  [(0,'A0n','A0d'), (2,'A2n','A2d'), (4,'A4n','A4d')],
			'4KG22': [(0,'B0n','B0d'), (2,'B2n','B2d')],
			'KG13':  [(0,'C0n','C0d'), (2,'C2n','C2d')],
			'2G22':  [(0,'D0n','D0d')],
			'G13':   [(0,'E0n','E0d')],
			'KB22':	 [(0,'F0n','F0d'), (2,'F2n','F2d'), (4,'F4n','F4d'), (6,'F6n','F6d')],
			'2F22':  [(0,'G0n','G0d')],
			'F13':   [(0,'H0n','H0d')],
			'KF22':	 [(0,'I0n','I0d'), (2,'I2n','I2d')],
		}

	ret = {}
	for (ell,sn,sd) in d[kernel]:
		f_all = df[sn].values.astype(float) / df[sd].values.astype(float)
		ind = np.nonzero(f_all) # indices with nonzero values
		ret[ell] = (n1_all[ind], n2_all[ind], f_all[ind])

	return ret

