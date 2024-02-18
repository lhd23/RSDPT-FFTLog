from __future__ import print_function
import warnings
import sys
import os
dir = os.path.dirname(os.path.abspath(__file__))

import numpy as np
from scipy.interpolate import interp1d
from scipy.special import gamma
from scipy.special import legendre
from scipy.integrate import quad
import h5py


OM0_P15 = 0.315
GAMMA = 0.55 # growth index
FGROWTH = OM0_P15**GAMMA

from pars import get_fftlog_param_dict

# Note: theta is defined as \theta := (\nabla\cdot\v) / (-calH f)
# 		i.e. we divide the conventional theta by -calH f

# Note: 'G13' kernel expansion coefficients do not include the factor of 3;
# they are multiplied in after the fact. (While it would make more sense to
# store the coefficients for 3G13, we do not want to risk introducing errors.)

def my_logspace(kmin, kmax, N): # convenience function for base e log-spacing
	k_arr = np.logspace(np.log(kmin), np.log(kmax), N, endpoint=False, base=np.exp(1))
	# Delta_lnk = np.log(kmax/kmin) / N
	# k_arr2 = kmin * np.exp(Delta_lnk * np.arange(0, N))
	# assert np.allclose(k_arr, k_arr2)
	return k_arr # Note kmax - Delta_lnk

class _base:

	def __init__(self, k, P, kernel, N, kmax, kmin, nu, multipoles,
				save_matrices, with_padding, with_windowing):
		"""
		Parameters
		----------
		k:			1d numpy array
					Wavenumbers regularly spaced in ln(k),
		P:			1d numpy array
					Linear Matter power spectrum P(k) eval'd at the input k's
					This array will be FFTLog'd (subject to padding)
		kernel:		str
		kmax:		float
					Upper bound not included in the array k, kmin <= k < kmax
					Note: kmax = np.log(k)[-1] + Delta
		kmin:		float
					Lower bound in the array k
					Note: kmin = np.log(k)[0]
		nu:			float
					bias parameter
		save_matrices: boolean
					Note if True and file exists this the matrisx
					will be read in and not saved

		Note
		----
		Log spacing is for base e not base 10

		Creating the 'imat' matrix (22) or vector (13) is the intensive part
		We have a unique imat for a given (kmin,kmax,nu,N), ell, AND set
		of coefficients {f_n1n2^ell : n1,n2=...} that characterise the kernel.

		All matrices associated with a given kernel (e.g. '2K22') will be saved
		to its own hdf5 file. So this will include three matrices for ell=0,2,4.
		Thus given the kernel string (e.g. '2K22') we only need to check that
		if there is a file 'imat_2K22_NXXX.hdf5' that (kmin,kmax,nu,N) match

		"""
		assert k.size == P.size
		assert N % 2 == 0 # must be even

		self.kernel = kernel
		self.fftlog_expansion_dict = get_fftlog_param_dict(self.kernel)
		self.integral_type = self.kernel[-2:] # indicated by last two characters in string, '22' or '13'
		assert self.integral_type in ['22','13']

		self._sigmau2 = None
		self.multipoles = multipoles

		# Signal array properties
		self.N = N
		self.kmax = kmax if kmax is not None else np.max(k)
		self.kmin = kmin if kmin is not None else np.min(k)
		self._Delta = np.log(self.kmax/self.kmin) / self.N # log spacing fixed

		# The input array to be FFTLog'd:
		self.plin_in = P
		self.lnk = np.log(k)

		# Processing of signal array
		if with_padding:
			tmp = self._pad(lnk=np.log(k), f=P)
			self.lnk = tmp[0]
			self.plin_in = tmp[1]
			# note lnk[-1] < ln(kmax) = lnk[-1] + Delta
			# Need to update the following
			self.kmax = np.exp(self.lnk[-1] + self.Delta)
			self.kmin = np.exp(self.lnk[0])
			self.N = self.lnk.size
		else:
			pass

		# Consistency checks
		assert np.allclose(np.diff(self.lnk), self.Delta)
		assert np.isclose(np.log(self.kmin), self.lnk[0])
		assert np.isclose(np.log(self.kmax), self.lnk[-1]+self.Delta)

		# FFT parameters
		self.nu = nu # bias
		self.ms = np.arange(-self.N/2, self.N/2+1, 1)
		self.m_upper = np.arange(0, self.N/2+1) # non-negative m
		self.etas = 1j * 2*np.pi * self.ms / np.log(self.kmax/self.kmin)
		self.nus = -0.5 * (self.nu + self.etas) # eq 2.22 Simonovic+ 2017

		# Number of c_m's to window for pos m (25% x 2 = 50% of c_m's will be windowed)
		self.N_window = int(0.25 * self.N/2)
		self.with_windowing = with_windowing

		self.cm = self._get_cm(self.with_windowing)


		# The imat array (a matrix for 22 and vector for 13 kernels) is independent
		# of the input signal array so can be reused for different cosmologies as
		# long as we are using the same (kmin,kmax,nu,N).

		# Create the matrix (22) or vector (13) or load if exists
		imat_pars = {'kmin':self.kmin, 'kmax':self.kmax, 'nu':self.nu, 'N':self.N} # sans kernel expansion coeffs
		filename = dir+'/fft_matrices/imat_{}_N{}.hdf5'.format(self.kernel, self.N)
		try:
			hf = h5py.File(filename, 'r')

			# Check imat matrix/vector was created using
			# the same FFTLog parameters (kmin,kmax,nu,N)
			for kw in imat_pars:
				if not np.isclose(hf.attrs[kw], imat_pars[kw]):
					hf.close()
					raise ValueError

			# print('Loading precomputed {} matrix from file...'.format(self.kernel))
			self.imat = {}
			for ell in self.multipoles:
				self.imat[ell] = np.array(hf.get(str(ell)))
			hf.close()

		except: # Compute from scratch
			print('Table not found: Creating imat array for {}'.format(self.kernel))
			self.imat = {}
			for ell in self.multipoles:
				self.imat[ell] = self.Ifuncarr(*self.fftlog_expansion_dict[ell])
			if save_matrices:
				print('Saving FFTLog matrices to file')
				hf = h5py.File(filename, 'w')
				# Write metadata
				for kw in imat_pars:
					hf.attrs[kw] = imat_pars[kw]
				for ell in self.multipoles:
					dset = hf.create_dataset(str(ell), data=self.imat[ell]) # dataset is complex 2d-array
				hf.close()

	@property # spacing should not change even with padding
	def Delta(self):
		return self._Delta

	def _pad(self, lnk, f):
		Npad = int(self.N)//2
		lnk_mid = lnk
		f_mid = f
		assert np.isclose(np.log(self.kmax), lnk_mid[-1] + self.Delta)
		assert np.isclose(np.log(self.kmin), lnk_mid[0])
		lnk_right = np.log(self.kmax) + np.arange(0,Npad)*self.Delta
		lnk_left = np.log(self.kmin) + np.arange(-Npad,0)*self.Delta
		lnk_new = np.hstack((lnk_left, lnk_mid, lnk_right))
		f_new = np.hstack((np.zeros(Npad), f_mid, np.zeros(Npad)))
		return lnk_new, f_new

	def _get_cm(self, with_windowing=True): # get fft c_m's
		plin_biased = self.plin_in* np.exp(self.lnk)**(-self.nu)
		cm_pos = np.fft.rfft(plin_biased)
		cm_pos[-1] *= 0.5

		if with_windowing:
			cm_pos *= _base._window(self.m_upper, self.N_window)

		cm_neg = np.conjugate(cm_pos[1:])
		cs = np.hstack((cm_neg[::-1], cm_pos)) / self.N
		return cs * self.kmin**(-self.etas)

	@staticmethod
	def _window(m, N_window):
		# Filter the c_m's using window function eq (C1) in 1603.04826
		# note x in eq (C1) equals m here
		m_max = m[-1]
		m_right = m_max - N_window # begin windowing at this m
		m_r = m[m > m_right] 
		r = (m_max-m_r) / float(m_max-m_right-1) 
		ret = np.ones(m.size)
		ret[m > m_right] = r - np.sin(2*np.pi*r)/(2.*np.pi)
		return ret

	def PL(self, k): # P(k) as a DFT (is periodic)
		k = np.asarray(k)
		if k.ndim == 0:
			k = k[np.newaxis]
		tmp = np.zeros((k.size))
		for i in range(k.size):
			tmp[i] = np.real(np.sum(self.cm * k[i]**(self.nu+self.etas)))
		assert np.allclose(np.imag(tmp), 0.0)
		return np.real(tmp[0]) if k.size == 1 else np.asarray(np.real(tmp))


class FFT_22(_base):

	def __init__(self, k=None, PL=None, kernel='2K22', fgrowth=FGROWTH, 
				N=256, nu=-0.6, save_matrices=False, with_padding=False, with_windowing=True):
		"""
		Parameters
		----------
		k:		 1d numpy array
				 Input k array. Does not have to be logarithmically spaced but
				 should be span a wide k range.
		PL:		 1d numpy array
				 Input linear matter power spectrum specified at k_in

		Note
		----
		The growth rate parameter is not used for G22
		"""
		self.fgrowth = fgrowth

		if kernel == '2K22':
			multipoles = [0,2,4]
			self.fac = self.fgrowth**2
		elif kernel == '4KG22':
			multipoles = [0,2]
			self.fac = self.fgrowth
		elif kernel == '2G22':
			multipoles = [0]
			self.fac = 1.
		elif kernel == 'KB22':
			multipoles = [0,2,4,6]
			self.fac = self.fgrowth**2
		elif kernel == 'KF22':
			multipoles = [0,2]
			self.fac = self.fgrowth**2
		elif kernel == '2F22':
			multipoles = [0]
			self.fac = 1.

		if k is None or PL is None:
			k, PL = np.loadtxt(dir+'/Pk_Planck15_large.dat', skiprows=4, unpack=True) # ~1e3 points

		self.ln_Pspl = interp1d(np.log(k), np.log(PL), bounds_error=False, fill_value='extrapolate')

		# Resample k so that k is logarithmically spaced (base e)
		kmin = k.min()
		kmax = k.max()
		k_in = my_logspace(kmin, kmax, N) # NB endpoint kmax is not included
		PL_in = np.exp(self.ln_Pspl(np.log(k_in)))

		_base.__init__(self, k_in, PL_in, kernel, N, kmax, kmin, nu, multipoles,
						save_matrices, with_padding, with_windowing)

		if self.integral_type != '22':
			raise ValueError('integral_type {} is invalid'.format(self.integral_type))

	def Ifunc(self, nu1, nu2): # eq 2.6 (S17) [NOT the kernel Iell]
		nu12 = nu1 + nu2
		G123 = gamma(1.5-nu1) * gamma(1.5-nu2) * gamma(nu12-1.5)
		G456 = gamma(nu1) * gamma(nu2) * gamma(3.-nu12)
		ret = G123/G456 / (8.*np.pi**1.5)
		return ret

	def Ifuncarr(self, n1s, n2s, fs):
		# Build complex matrix of \sum_{n1,n2} f_{n1n2}^l x Ifunc(nu1-n1,nu2-n2)
		Iarr = np.zeros((self.N+1,self.N+1), dtype=complex)
		for m1 in range(self.N+1):
			for m2 in range(self.N+1):
				if m1 - self.N/2 < 1:
					Iarr[m1,m2] = np.dot(fs, self.Ifunc(self.nus[m1]-n1s, self.nus[m2]-n2s))
				else: 
					Iarr[m1,m2] = np.conjugate(Iarr[self.N-m1, self.N-m2])
		return Iarr

	def _conv_int(self, k, ell=0): # \int I_ell(q,k-1) P(q) P(|k-q|)
		k = np.asarray(k)
		if k.ndim == 0:
			k = k[np.newaxis]
		tmp = np.zeros((k.size,), dtype=complex)
		for i in range(k.size):
			cks = self.cm * k[i]**(-2.*self.nus)
			tmp[i] = np.real(np.dot(cks, np.dot(self.imat[ell], cks)))
		assert np.allclose(np.imag(tmp), 0.0)
		ret = k**3 * np.real(tmp)
		return ret[0] if k.size == 1 else np.asarray(ret)

	def _P22_unnorm(self, k, ell): # unnormalized
		return self._conv_int(k,ell)

	def P22(self, k, ell=0):
		ret = self.fac * self._P22_unnorm(k,ell)
		if self.kernel == '2K22' and ell == 0: # legacy pattern
			ret += self.PL(k)
		return ret

	def DelP0(self, k):
		P0 = self.P22(k, ell=0)
		return P0 - self.PL(k)

	def __call__(self, k, ell):
		assert ell in self.multipoles
		return self.P22(k,ell)


class FFT_13(_base):

	def __init__(self, k=None, PL=None, kernel='G13', fgrowth=FGROWTH, 
				N=256, nu=-0.6, save_matrices=False, with_padding=False, with_windowing=True):
		"""
		Parameters
		----------
		k:		 1d numpy array
				 Input k array. Does not have to be logarithmically spaced but
				 should span a wide k range e.g. (1e-5,1e3).
		PL:		 1d numpy array
				 Input linear matter power spectrum specified at k_in

		Note
		----
		The growth rate parameter is not used for G13
		"""
		self.fgrowth = fgrowth

		if kernel == 'G13':
			multipoles = [0]
			self.UV_dict = {0: 0.9}
			self.fac = 3.
		elif kernel == 'KG13':
			multipoles = [0,2]
			A = 8.*self.fgrowth # NOTE! not 4/3 or 4
			self.UV_dict = {0: 19./84*A, 2: 23./210*A} # dirty fix
			self.fac = A
		elif kernel == 'F13':
			multipoles = [0]
			self.UV_dict = {0: 61./210}
			self.fac = 3.

		if k is None or PL is None:
			k, PL = np.loadtxt(dir+'/Pk_Planck15_large.dat', skiprows=4, unpack=True) # ~1e3 points

		self.ln_Pspl = interp1d(np.log(k), np.log(PL), bounds_error=False, fill_value='extrapolate')

		# Resample k so that k is logarithmically spaced (base e)
		kmin = k.min()
		kmax = k.max()
		k_in = my_logspace(kmin, kmax, N) # NB endpoint kmax is not included
		PL_in = np.exp(self.ln_Pspl(np.log(k_in)))

		_base.__init__(self, k_in, PL_in, kernel, N, kmax, kmin, nu, multipoles,
						save_matrices, with_padding, with_windowing)

		warnings.warn('I(nu1,nu2) with nu2=0,-1,-2,... will be regularized to zero', stacklevel=2)

		if self.integral_type != '13':
			raise ValueError('integral_type {} is invalid'.format(self.integral_type))

	@property
	def sigmau2(self): # the scaled 1D velocity dispersion (H^2f^2 divided out)
		if self._sigmau2 is None:
			y,err = quad(lambda lnq: np.exp(lnq)*np.exp(self.ln_Pspl(lnq)), -8., 3., limit=150)
			self._sigmau2 = y / (6.*np.pi**2)
			if err/y > 0.01:
				print('Warning: the estimated error on sigmau2 is >1%')
		return self._sigmau2

	def Ifunc(self, nu1, nu2):
		nu12 = nu1 + nu2
		G123 = gamma(1.5-nu1) * gamma(1.5-nu2) * gamma(nu12-1.5)
		G456 = gamma(nu1) * gamma(nu2) * gamma(3.-nu12)
		ret = G123/G456 / (8.*np.pi**1.5)
		if np.any(np.isnan(ret)): # regularization
			ret[np.argwhere(np.isnan(ret))] = 0.0
		return ret

	def Ifuncarr(self, n1s, n2s, fs):
		# Build complex vector of \sum_{n1,n2} f_{n1n2}^l x Ifunc(nu1-n1,-n2)
		Iarr = np.zeros((self.N+1,), dtype=complex)
		for m1 in range(self.N+1):
			if m1 - self.N/2 < 1 :
				Iarr[m1] = np.dot(fs, self.Ifunc(self.nus[m1]-n1s, -n2s))
			else: 
				Iarr[m1] = np.conjugate(Iarr[self.N-m1])
		return Iarr

	def _kcorr_int(self, k, ell):
		"""
		Evalute type 13 integrals of form

			int G_3(k,q,-q) P(q)

		Returns real value scalar or array
		"""
		k = np.asarray(k)
		if k.ndim == 0:
			k = k[np.newaxis]
		tmp = np.zeros((k.size,), dtype=complex)
		for i in range(k.size):
			cks = self.cm * k[i]**(-2.*self.nus)
			tmp[i] = np.real(np.dot(cks, self.imat[ell]))
		assert np.allclose(np.imag(tmp), 0.0)
		ret = k**3 * np.real(tmp)
		return ret[0] if k.size == 1 else np.asarray(ret)

	def UV(self, k, ell):
		PL = np.exp(self.ln_Pspl(np.log(k)))
		return -self.UV_dict[ell] * k*k * PL * self.sigmau2

	def P13(self, k, ell):
		PL = np.exp(self.ln_Pspl(np.log(k)))
		ret = self.fac*PL*self._kcorr_int(k,ell) + self.UV(k,ell)
		return ret

	def K3_ell0(self, k):
		PL = np.exp(self.ln_Pspl(np.log(k)))
		return -(1./3) * self.fgrowth**2 * k*k * PL * self.sigmau2

	def K3_ell2(self, k):
		PL = np.exp(self.ln_Pspl(np.log(k)))
		return -(2./3) * self.fgrowth**2 * k*k * PL * self.sigmau2

	def __call__(self, k, ell):
		assert ell in self.multipoles
		return self.P13(k,ell)



def main():
	pass


if __name__ == '__main__':
	main()







