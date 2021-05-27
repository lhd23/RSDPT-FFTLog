from __future__ import print_function
import warnings
import sys
import os
dir = os.path.dirname(os.path.abspath(__file__))

import numpy as np
from scipy.interpolate import interp1d
from scipy.special import legendre,erf
from scipy.integrate import quad

import FFTLog_integrals as FFTLog


OM0_P15 = 0.315
GAMMA = 0.55 # growth index
FGROWTH = OM0_P15**GAMMA


def cut_off_ringing(k, P, ir_cutoff=5e-5, uv_cutoff=1e2):
	"""
	Cut off ringing tail at high and low k

	Parameters
	----------
	k:	1d numpy array
		Note does not need to be ln-space
	P:	1d numpy array
		Function evaluated at corresponding k
	"""
	if ir_cutoff < k.min(): ir_cutoff = k.min()
	if uv_cutoff > k.max(): uv_cutoff = k.max()
	ind = (k < uv_cutoff) * (k > ir_cutoff)
	return k[ind], P[ind]

def upsample(k, P, ir_cutoff=2e-4, uv_cutoff=1e2, num=1024): # cutoff, interpolate then resample (FFTLog)
	k, P = cut_off_ringing(k, P, ir_cutoff, uv_cutoff)
	kmin = np.min(k)
	kmax = np.max(k)
	knew = FFTLog.my_logspace(kmin, kmax, num)
	if np.all(P > 0.0):
		lnP_spl = interp1d(np.log(k), np.log(P), bounds_error=False, fill_value='extrapolate')
		Pnew = np.exp(lnP_spl(np.log(knew)))
	else:
		P_spl = interp1d(np.log(k), P, bounds_error=False, fill_value='extrapolate')
		Pnew = P_spl(np.log(knew))
	return knew, Pnew

def Ptt_1loop(k, PL, get_compts=False, N=512, nu=-0.6, save_matrices=True):
	"""
	Compute the theta-theta real-space power spectrum
	using one-loop SPT. Note theta is scaled by -1/(calH f)

	Parameters
	----------
	k:	1d numpy array
		Note does not need to be ln-space
	P:	1d numpy array
		Linear matter power spectrum given at corresponding k

	Returns
	-------
	k_fftlog, [...], P_1loop:	tuple
	Integral output arrays in FFTLog form (ln spacing)

	Note
	----
	theta is divided by -calH f so to get the physical theta
	one will need to multiply the power spectrum by (calH f)^2

	"""
	kw = {'N':N, 'nu':nu, 'save_matrices':save_matrices}
	fft_2G22 = FFTLog.FFT_22(k, PL, kernel='2G22', **kw)
	fft_3G13 = FFTLog.FFT_13(k, PL, kernel='G13', **kw)

	k_fftlog = np.exp(fft_2G22.lnk)
	P11 = fft_2G22.PL(k_fftlog)
	P13 = fft_3G13.P13(k_fftlog, ell=0)
	P22 = fft_2G22.P22(k_fftlog, ell=0)
	P_1loop = P11 + P22 + 2.*P13
	if get_compts:
		return k_fftlog, P11, P13, P22, P_1loop
	else:
		return k_fftlog, P_1loop

def Pmm_1loop(k, PL, get_compts=False, N=512, nu=-0.6, save_matrices=True):
	"""
	As with Ptt_1loop but for matter-matter
	"""
	kw = {'N':N, 'nu':nu, 'save_matrices':save_matrices}
	fft_2F22 = FFTLog.FFT_22(k, PL, kernel='2F22', **kw)
	fft_3F13 = FFTLog.FFT_13(k, PL, kernel='F13', **kw)

	k_fftlog = np.exp(fft_2F22.lnk)
	P11 = fft_2F22.PL(k_fftlog)
	P13 = fft_3F13.P13(k_fftlog, ell=0)
	P22 = fft_2F22.P22(k_fftlog, ell=0)
	P_1loop = P11 + P22 + 2.*P13
	if get_compts:
		return k_fftlog, P11, P13, P22, P_1loop
	else:
		return k_fftlog, P_1loop

def P0tt_1loop(k, PL, fgrowth=FGROWTH, get_compts=False, N=512, nu=-0.6, save_matrices=True): # with RSD
	"""
	Compute the monopole moment (ell=0) of redshift-space power
	spectrum (theta-theta) using one-loop SPT.

	Parameters
	----------
	k:	1d numpy array
		Note does not need to be ln-space
	P:	1d numpy array
		Linear matter power spectrum given at corresponding k

	Returns
	-------
	k_fftlog, [...], P_1loop:	tuple
	Integral output arrays in FFTLog form (ln spacing)


	Note
	----
	theta here is divided -calH f so to get the physical theta
	one will need to multiply the power spectrum by (calH f)^2

	"""
	kw = {'N':N, 'nu':-0.6, 'save_matrices':True}
	fft_2K22 = FFTLog.FFT_22(k, PL, kernel='2K22', fgrowth=fgrowth, **kw)
	fft_KG13 = FFTLog.FFT_13(k, PL, kernel='KG13', fgrowth=fgrowth, **kw)
	fft_4KG22 = FFTLog.FFT_22(k, PL, kernel='4KG22', fgrowth=fgrowth, **kw)

	k_fftlog, P_1loop = Ptt_1loop(k, PL, **kw)
	assert np.allclose(k_fftlog, np.exp(fft_2K22.lnk))

	P_2K22_ell0 = fft_2K22.DelP0(k_fftlog) # Note we subract out P_11 !!!
	P_4KG22_ell0 = fft_4KG22.P22(k_fftlog, ell=0)
	P_KG13_ell0 = fft_KG13.P13(k_fftlog, ell=0)
	P_3K13_ell0 = fft_KG13.K3_ell0(k_fftlog) # the last term
	P0 = P_1loop + P_2K22_ell0 + P_3K13_ell0 + P_4KG22_ell0 + P_KG13_ell0
	if get_compts:
		P11 = fft_2K22.PL(k_fftlog)
		return k_fftlog, P11, P_1loop, P_2K22_ell0, P_3K13_ell0, P_4KG22_ell0, P_KG13_ell0, P0
	else:
		return k_fftlog, P0

def Ps_tt_1loop(k, PL, mu, fgrowth=FGROWTH, get_compts=False, N=512, nu=-0.6, save_matrices=True): # with RSD
	"""
	Compute the redshift-space power spectrum P(k,mu)

	Parameters
	----------
	k:	1d numpy array
		Note does not need to be ln-space
	P:	1d numpy array
		Linear matter power spectrum given at corresponding k
	mu: cosine of angular separation between k and n (LOS)

	Returns
	-------
	k_fftlog, [...], P_1loop:	tuple
	Integral output arrays in FFTLog form (ln spacing)


	Note
	----
	theta here is divided -calH f so to get the physical theta
	one will need to multiply the power spectrum by (calH f)^2

	"""
	kw = {'N':N, 'nu':-0.6, 'save_matrices':True}
	fft_2K22 = FFTLog.FFT_22(k, PL, kernel='2K22', fgrowth=fgrowth, **kw)
	fft_4KG22 = FFTLog.FFT_22(k, PL, kernel='4KG22', fgrowth=fgrowth, **kw)
	fft_KG13 = FFTLog.FFT_13(k, PL, kernel='KG13', fgrowth=fgrowth, **kw)

	k_fftlog, P_1loop = Ptt_1loop(k, PL, **kw)
	assert np.allclose(k_fftlog, np.exp(fft_2K22.lnk))
	P11 = fft_2K22.PL(k_fftlog)

	P_2K22_ell0 = fft_2K22.DelP0(k_fftlog) # Note we subract out P_11 !!!
	P_2K22_ell2 = fft_2K22.P22(k_fftlog, ell=2)
	P_2K22_ell4 = fft_2K22.P22(k_fftlog, ell=4)

	P_4KG22_ell0 = fft_4KG22.P22(k_fftlog, ell=0)
	P_4KG22_ell2 = fft_4KG22.P22(k_fftlog, ell=2)

	P_KG13_ell0 = fft_KG13.P13(k_fftlog, ell=0)
	P_KG13_ell2 = fft_KG13.P13(k_fftlog, ell=2)

	# the last term
	P_3K13_ell0 = fft_KG13.K3_ell0(k_fftlog)
	P_3K13_ell2 = fft_KG13.K3_ell2(k_fftlog)

	P0 = P_2K22_ell0 + P_4KG22_ell0 + P_KG13_ell0 + (P_1loop) + P_3K13_ell0
	P2 = P_2K22_ell2 + P_4KG22_ell2 + P_KG13_ell2 + P_3K13_ell2
	P4 = P_2K22_ell4

	if get_compts:
		return k_fftlog, P11, P_1loop, P0, P2, P4
	else:
		Ps = legendre(0)(mu)*P0 + legendre(2)(mu)*P2 + legendre(4)(mu)*P4
		return k_fftlog, Ps


class Ps_tt:

	def __init__(self, k, PL, fgrowth=FGROWTH, sig_fog=3.0, lmax_fog=8, N=512, nu=-0.6, save_matrices=True):

		self.k_input = k
		self.PL_input = PL
		self.fgrowth = fgrowth
		self.H0f = 100. * self.fgrowth
		self.sig_fog = sig_fog
		self.fog = FoG(self.sig_fog)
		self.lmax_fog = lmax_fog
		self.ell_fog = np.arange(0, self.lmax_fog+2, 2)
		self.ell_Pvv = [0,2,4,6] # the multipoles in power spectrum (sans FoG)

		self.N = N
		self.nu = nu

		kw = {'N':self.N, 'nu':self.nu, 'save_matrices':save_matrices}

		# real-space contributions
		fft_2G22 = FFTLog.FFT_22(k, PL, kernel='2G22', **kw)
		fft_3G13 = FFTLog.FFT_13(k, PL, kernel='G13', **kw)

		self.sigmau2 = fft_3G13.sigmau2
		self.lnk = fft_2G22.lnk # fftlog spacing
		self.k = np.exp(self.lnk) # fftlog spacing
		self.kmin = np.min(self.k)
		self.kmax = np.max(self.k)
		self.P11 = fft_2G22.PL(self.k)
		self.P13 = fft_3G13.P13(self.k, ell=0)
		self.P22 = fft_2G22.P22(self.k, ell=0)

		# redshift-space contributions
		fft_2K22 = FFTLog.FFT_22(k, PL, kernel='2K22', fgrowth=self.fgrowth, **kw)
		fft_4KG22 = FFTLog.FFT_22(k, PL, kernel='4KG22', fgrowth=self.fgrowth, **kw)
		fft_KG13 = FFTLog.FFT_13(k, PL, kernel='KG13', fgrowth=self.fgrowth, **kw)

		self.P_2K22_ell0 = fft_2K22.DelP0(self.k) # Note we subract out P_11 !!!
		self.P_2K22_ell2 = fft_2K22.P22(self.k, ell=2)
		self.P_2K22_ell4 = fft_2K22.P22(self.k, ell=4)

		self.P_4KG22_ell0 = fft_4KG22.P22(self.k, ell=0)
		self.P_4KG22_ell2 = fft_4KG22.P22(self.k, ell=2)

		self.P_KG13_ell0 = fft_KG13.P13(self.k, ell=0)
		self.P_KG13_ell2 = fft_KG13.P13(self.k, ell=2)

		# the last term
		self.P_3K13_ell0 = fft_KG13.K3_ell0(self.k)
		self.P_3K13_ell2 = fft_KG13.K3_ell2(self.k)

	@property
	def P_1loop(self):
		return self.P11 + self.P22 + 2.*self.P13

	@property
	def P0(self):
		return self.P_2K22_ell0 + self.P_4KG22_ell0 \
				+ self.P_KG13_ell0 + self.P_3K13_ell0 + self.P_1loop

	@property
	def P2(self):
		return self.P_2K22_ell2 + self.P_4KG22_ell2 \
				+ self.P_KG13_ell2 + self.P_3K13_ell2

	@property
	def P4(self):
		return self.P_2K22_ell4

	def Ps(self, mu): # P^s_{\theta\theta}(k,mu)
		ret = legendre(0)(mu)*self.P0 \
			+ legendre(2)(mu)*self.P2 \
			+ legendre(4)(mu)*self.P4
		return ret

# NB. calXX methods have the mu^2 part included in them but are not divided by k^2

	@property
	def calP0(self):
		return self.P0/3. + 2.*self.P2/15.

	@property
	def calP2(self):
		return 2.*self.P0/3. + 11.*self.P2/21. + 4.*self.P4/21.

	@property
	def calP4(self):
		return 12.*self.P2/35. + 39.*self.P4/77.

	@property
	def calP6(self):
		return 10.*self.P4/33.

	# no RSD model
	@property
	def P0uu_norsd(self):
		return 1./3 * self.P_1loop / self.k**2

	@property
	def P2uu_norsd(self):
		return 2./3 * self.P_1loop / self.k**2

	@property
	def P0vv_norsd(self):
		return self.P0uu_norsd * self.H0f**2

	@property
	def P2vv_norsd(self):
		return self.P2uu_norsd * self.H0f**2

	def Pvv_norsd(self, mu):
		return legendre(0)(mu)*self.P0vv_norsd \
				+ legendre(2)(mu)*self.P2vv_norsd

# Dispersion model/FoG methods.
# NB. sig = sigma_p/sqrt(2) and has units h/Mpc i.e. it has been divided by H0=100h km/s/Mpc

	def Psuu_ell(self, ell, with_fog=True):
		calP_ell = {0: self.calP0, 2: self.calP2, 4: self.calP4, 6: self.calP6}
		if with_fog:
			ell_fog = np.arange(0, self.lmax_fog+2, 2)
			D_ell = {i: self.fog.Dv_ell(self.k, ell=i) for i in ell_fog}
			ret = self.fog.couple(ell, D_ell, calP_ell)
			ret *= 1./self.k**2
			return ret
		else:
			return calP_ell[ell] / self.k**2

	def Psvv_ell(self, ell, with_fog=False):
		val = self.Psuu_ell(ell, with_fog=with_fog)
		return val * self.H0f**2

	def Psuu(self, mu, with_fog=False):
		ret = 0.
		lmax = (self.ell_Pvv[-1] + self.ell_fog[-1]) if with_fog else self.ell_Pvv[-1]
		for l in np.arange(0, lmax+2, 2):
			ret += legendre(l)(mu) * self.Psuu_ell(l, with_fog=with_fog)	
		return ret

	def Psvv(self, mu, with_fog=False):
		val = self.Psuu(mu, with_fog=with_fog)
		return val * self.H0f**2


class Ps_vv_cumexp:

	def __init__(self, k, PL, fgrowth=FGROWTH, sig_fog=3.0, lmax_fog=8, N=512, nu=-0.6, save_matrices=True):

		self.k_input = k
		self.PL_input = PL
		self.fgrowth = fgrowth
		self.H0f = 100. * self.fgrowth
		self.sig_fog = sig_fog
		self.fog = FoG(self.sig_fog)
		self.lmax_fog = lmax_fog
		self.ell_fog = np.arange(0, self.lmax_fog+2, 2)
		self.ell_Pvv = [0,2,4,6] # the multipoles in power spectrum (sans FoG)

		self.N = N
		self.nu = nu

		kw = {'N':self.N, 'nu':self.nu, 'save_matrices':save_matrices}

		# Real-space contributions
		fft_2G22 = FFTLog.FFT_22(k, PL, kernel='2G22', **kw)
		fft_3G13 = FFTLog.FFT_13(k, PL, kernel='G13', **kw)

		self.sigmau2 = fft_3G13.sigmau2
		self.lnk = fft_2G22.lnk # fftlog spacing
		self.k = np.exp(self.lnk) # fftlog spacing
		self.kmin = np.min(self.k)
		self.kmax = np.max(self.k)
		self.P11 = fft_2G22.PL(self.k)
		self.P13 = fft_3G13.P13(self.k, ell=0)
		self.P22 = fft_2G22.P22(self.k, ell=0)

		# Redshift-space contributions

		# K_F contribution (pure 22 loop)
		fft_KF22 = FFTLog.FFT_22(k, PL, kernel='KF22', fgrowth=self.fgrowth, **kw)
		self.P0uu_KF = fft_KF22.P22(self.k, ell=0) / self.k**2
		self.P2uu_KF = fft_KF22.P22(self.k, ell=2) / self.k**2

		# K_B contribution (pure 22 loop)
		fft_KB22 = FFTLog.FFT_22(k, PL, kernel='KB22', fgrowth=self.fgrowth, **kw)
		self.P0uu_KB = fft_KB22.P22(self.k, ell=0) / self.k**2
		self.P2uu_KB = fft_KB22.P22(self.k, ell=2) / self.k**2
		self.P4uu_KB = fft_KB22.P22(self.k, ell=4) / self.k**2
		self.P6uu_KB = fft_KB22.P22(self.k, ell=6) / self.k**2
		# note we divided by k^2 because the kernel expansion is of k^2 * K_B

		# K_A contribution (bispectrum = 22 + 13)
		# (we can reuse the velocity divergence result then convert to velocity)
		fft_4KG22 = FFTLog.FFT_22(k, PL, kernel='4KG22', fgrowth=self.fgrowth, **kw)
		P0tt_4KG22 = fft_4KG22.P22(self.k, ell=0)
		P2tt_4KG22 = fft_4KG22.P22(self.k, ell=2)
		fft_KG13 = FFTLog.FFT_13(k, PL, kernel='KG13', fgrowth=self.fgrowth, **kw)
		P0tt_KG13 = fft_KG13.P13(self.k, ell=0)
		P2tt_KG13 = fft_KG13.P13(self.k, ell=2)
		# convert to velocity
		P0tt = P0tt_4KG22 + P0tt_KG13
		P2tt = P2tt_4KG22 + P2tt_KG13
		P4tt = 0.0
		self.P0uu_KA = (1./3*P0tt + 2./15*P2tt) / self.k**2
		self.P2uu_KA = (2./3*P0tt + 11./21*P2tt + 4./21*P4tt) / self.k**2
		self.P4uu_KA = (12./35*P2tt + 39./77*P4tt) / self.k**2
		self.P6uu_KA = 10./33*P4tt / self.k**2

	@property
	def P_1loop(self):
		return self.P11 + self.P22 + 2.*self.P13

	# the no-RSD model
	@property
	def P0uu_norsd(self):
		return 1./3 * self.P_1loop / self.k**2

	@property
	def P2uu_norsd(self):
		return 2./3 * self.P_1loop / self.k**2

	@property
	def P0vv_norsd(self):
		return self.P0uu_norsd * self.H0f**2

	@property
	def P2vv_norsd(self):
		return self.P2uu_norsd * self.H0f**2

	def Puu_norsd(self, mu):
		return legendre(0)(mu)*self.P0uu_norsd \
				+ legendre(2)(mu)*self.P2uu_norsd

	def Pvv_norsd(self, mu):
		return self.Puu_norsd(mu) * self.H0f**2

	# the RSD model
	@property
	def P0uu(self):
		return self.P0uu_norsd + self.P0uu_KA + self.P0uu_KB #+ self.P0uu_KF

	@property
	def P2uu(self):
		return self.P2uu_norsd + self.P2uu_KA + self.P2uu_KB #+ self.P2uu_KF

	@property
	def P4uu(self):
		return self.P4uu_KA + self.P4uu_KB

	@property
	def P6uu(self):
		return self.P6uu_KA + self.P6uu_KB

	@property
	def P0vv(self):
		return self.P0uu * self.H0f**2

	@property
	def P2vv(self):
		return self.P2uu * self.H0f**2

	@property
	def P4vv(self):
		return self.P4uu * self.H0f**2

	@property
	def P6vv(self):
		return self.P6uu * self.H0f**2

	def Psuu_ell(self, ell, with_fog=True):
		P_ell = {0: self.P0uu, 2: self.P2uu, 4: self.P4uu, 6: self.P6uu}
		if with_fog:
			ell_fog = np.arange(0, self.lmax_fog+2, 2)
			D_ell = {i: self.fog.Dv_ell(self.k, ell=i) for i in ell_fog}
			ret = self.fog.couple(ell, D_ell, P_ell)
			return ret
		else:
			return P_ell[ell]

	def Psvv_ell(self, ell, with_fog=False):
		val = self.Psuu_ell(ell, with_fog=with_fog)
		return val * self.H0f**2

	def Psuu(self, mu, with_fog=False):
		ret = 0.
		lmax = (self.ell_Pvv[-1] + self.ell_fog[-1]) if with_fog else self.ell_Pvv[-1]
		for l in np.arange(0, lmax+2, 2):
			ret += legendre(l)(mu) * self.Psuu_ell(l, with_fog=with_fog)	
		return ret

	def Psvv(self, mu, with_fog=False):
		val = self.Psuu(mu, with_fog=with_fog)
		return val * self.H0f**2


class FoG:

	def __init__(self, sig=3.0):
		self.sig = sig
		self.kx = 1./sig if sig > 0. else 1e-3 # turning point
		self.lnD_spl = {}
		self.w3j = np.load('wigner3j_ell.npy')

	def Dv(self, k, mu):
		x = k*mu*self.sig
		return np.exp(-x*x)

	def _Dv_ell(self, k, ell, **kwargs):
		fcn = lambda mu: self.Dv(k,mu) * legendre(ell)(mu)
		val, err = quad(fcn, a=-1.0, b=1.0, **kwargs)
		val *= (2.*ell+1.)/2.
		return val

	def Dv_ell(self, k, ell, num=50):
		lnk = np.log(k)
		try:
			lnD = self.lnD_spl[ell](lnk)
		except: # if spline hasn't been created yet
			k_arr = np.logspace(np.log10(self.kx)-1.5, 1, num)
			D_arr = np.array([self._Dv_ell(_,ell) for _ in k_arr])
			self.lnD_spl[ell] = interp1d(np.log(k_arr), np.log(np.abs(D_arr)), \
										bounds_error=False, fill_value='extrapolate')
			lnD = self.lnD_spl[ell](lnk)
		phase_fac = np.real(1j**ell)
		ret = phase_fac * np.exp(lnD)
		return ret

	def couple(self, ell, Din, Pin):
		"""
		Compute the multipoles of the damped power spectrum

				D_FoG(k,mu) * P(k,mu)

		given the multipoles of D_FoG and P. Coupling is
		determined by the Wigner 3j-symbols.

		Parameters
		----------
		Din:	the input dictionary of D_FoG multipoles
		Pin:	the input dictionary of power spectrum multipoles (no FoG)

		"""
		ells_D = np.sort([kw for kw in Din]) # 0,2,4,...
		ells_P = np.sort([kw for kw in Pin]) # 0,2,4,...
		assert (ells_D.max() < 20) and (ells_P.max() < 20)
		if Din is not None:
			ret = 0.
			for m in ells_D:
				for n in ells_P:
					w3j = self.w3j[ell,m,n]
					ret += w3j**2 * Din[m] * Pin[n]
			ret *= 2.*ell + 1.
			return ret
		else:
			return Pin[ell]

# some analytic multipoles of Dv = exp(-x^2) for testing

	def Dv0(self, k):
		x = k*self.sig
		ret = np.sqrt(np.pi)*erf(x)/(2.*x)
		return ret

	def Dv2(self, k):
		x = k*self.sig
		ret = 5./(8.*x*x*x) * (-6.*np.exp(-x*x)*x \
				+ np.sqrt(np.pi)*(3.-2.*x*x)*erf(x))
		return ret

	def Dv4(self, k):
		x = k*self.sig
		ret = 9./(64.*x**5) * (-10.*np.exp(-x*x)*x*(21.+2.*x*x) \
				+ 3.*np.sqrt(np.pi)*(35.-20.*x*x+4.*x**4)*erf(x))
		return ret



def main():
	F = damping(sig=3.)
	F.test()


if __name__ == '__main__':
	main()








