from __future__ import print_function
import sys
import os
dir = os.path.dirname(os.path.abspath(__file__))

from FFTLog_integrals import *
import power_FFTLog as power

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker
locmin = matplotlib.ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * .1, numticks=100)

def find_ind(k, P):
	ipos = P >= 0.0
	ineg = P < 0.0
	kpos, Ppos = k[ipos], P[ipos]
	kneg, Pneg = k[ineg], P[ineg]
	return (kpos, Ppos, kneg, Pneg)

def plot_all():
	N = 1400
	nu = -0.6
	with_padding = False
	save_matrices = False
	kw = {'N':N, 'nu':nu, 'with_padding':with_padding, 'save_matrices':save_matrices}
	fft_2G22 = FFT_22(kernel='2G22', **kw)
	fft_G13 = FFT_13(kernel='G13', **kw)
	fft_2K22 = FFT_22(kernel='2K22', **kw)
	fft_4KG22 = FFT_22(kernel='4KG22', **kw)
	fft_KG13 = FFT_13(kernel='KG13', **kw)

	k = np.exp(fft_2G22.lnk)
	PL = fft_2G22.PL(k)

	# one-loop
	P13 = fft_G13.P13(k, ell=0)
	P22 = fft_2G22.P22(k, ell=0)
	P_1loop_corr = P22 + 2*P13

	P_2K22_ell0 = fft_2K22.DelP0(k) # Note we subract out P_11 !!!
	P_2K22_ell2 = fft_2K22.P22(k, ell=2)
	P_2K22_ell4 = fft_2K22.P22(k, ell=4)

	P_4KG22_ell0 = fft_4KG22.P22(k, ell=0)
	P_4KG22_ell2 = fft_4KG22.P22(k, ell=2)

	P_KG13_ell0 = fft_KG13.P13(k, ell=0)
	P_KG13_ell2 = fft_KG13.P13(k, ell=2)

	P_3K13_ell0 = fft_KG13.K3_ell0(k)
	P_3K13_ell2 = fft_KG13.K3_ell2(k)

	P_1loop = PL + P_1loop_corr # no rsd corrections
	P0 = P_2K22_ell0 + P_4KG22_ell0 + P_KG13_ell0 + (P_1loop) + P_3K13_ell0
	P2 = P_2K22_ell2 + P_4KG22_ell2 + P_KG13_ell2 + P_3K13_ell2
	P4 = P_2K22_ell4

	plt.figure(figsize=(6,6))
	plt.loglog(k, P0, 'k', lw=1.1) # label=r'$\ell=0$', 
	# plt.loglog(k, np.abs(P2), 'b', label=r'$\ell=2$', lw=1.2)
	kp, P2p, kn, P2n = find_ind(k, P2)
	plt.loglog(kp, P2p, 'b', lw=1.4) # label=r'$\ell=2$', 
	plt.loglog(kn, np.abs(P2n), 'b--', dashes=(5,3), lw=1.4)

	plt.loglog(k, P4, 'r', lw=1.4) # label=r'$\ell=4$', 

	plt.loglog(k, P_1loop, 'k-.', label=r'$P^{1\!-\!loop}_{\theta\theta}$', lw=1.1)
	plt.loglog(k, PL, c='gray', ls=':', lw=1.4)

	plt.text(x=0.0035, y=7500, s=r'$P^0_{\theta\theta}$')
	plt.text(x=0.19, y=2430, s=r'$P_L$')
	plt.text(x=3e-2, y=400, s=r'$P^2_{\theta\theta}$', c='b')
	plt.text(x=5e-2, y=36, s=r'$P^4_{\theta\theta}$', c='r')
	# plt.grid(ls=':')
	plt.legend(frameon=False, loc='upper right', fontsize=16)
	plt.tick_params(right=True, top=True, which='both')
	# plt.xlim(1e-3,3e0)
	plt.xlim(3e-3,0.3)
	plt.ylim(1e1,4e4)
	# plt.xticks([1e-3,1e-2,1e-1,1e0])
	plt.xlabel(r'Wavenumber $k$ [h Mpc$^{-1}$]')
	plt.ylabel(r'$P^\ell_{\theta\theta}(k)$ [h$^{-3}$ Mpc$^3$]')
	plt.show()

def plot_ell0_compts():
	N = 1400
	nu = -0.6
	with_padding = False
	save_matrices = False
	kw = {'N':N, 'nu':nu, 'with_padding':with_padding, 'save_matrices':save_matrices}
	fft_2G22 = FFT_22(kernel='2G22', **kw)
	fft_G13 = FFT_13(kernel='G13', **kw)
	fft_2K22 = FFT_22(kernel='2K22', **kw)
	fft_4KG22 = FFT_22(kernel='4KG22', **kw)
	fft_KG13 = FFT_13(kernel='KG13', **kw)

	k = np.exp(fft_2G22.lnk)
	PL = fft_2G22.PL(k)

	# one-loop
	P13 = fft_G13.P13(k, ell=0)
	P22 = fft_2G22.P22(k, ell=0)
	P_1loop_corr = P22 + 2*P13
	P_2K22_ell0 = fft_2K22.DelP0(k) # Note we subract out P_11 !!!
	P_4KG22_ell0 = fft_4KG22.P22(k, ell=0)
	P_KG13_ell0 = fft_KG13.P13(k, ell=0)

	# the last term
	P_3K13_ell0 = fft_KG13.K3_ell0(k)

	P_1loop = PL + P_1loop_corr # no rsd corrections
	P0 = P_2K22_ell0 + P_4KG22_ell0 + P_KG13_ell0 + (P_1loop) + P_3K13_ell0

	plt.figure(figsize=(6,6))
	plt.loglog(k, P0, 'k', lw=1.2)
	plt.loglog(k, P_2K22_ell0, 'b', lw=1.2)
	plt.loglog(k, P_4KG22_ell0, 'magenta', lw=1.2)
	plt.loglog(k, np.abs(P_KG13_ell0), 'r', ls='--', dashes=(5,3), lw=1.2)
	plt.loglog(k, np.abs(P_3K13_ell0), 'lime', ls='--', dashes=(5,3), lw=1.2)

	plt.loglog(k, np.abs(P22+2*P13), 'turquoise', ls='--', dashes=(5,3), lw=1.2)
	plt.loglog(k, PL, c='gray', ls=':', lw=1.2)

	plt.text(x=0.0035, y=7500, s=r'$P^0_{\theta\theta}$')
	plt.text(x=0.19, y=2430, s=r'$P_L$')
	plt.text(x=0.015, y=1100, s=r'$P_{22}+2P_{13}$', c='turquoise')
	plt.text(x=0.1, y=74, s=r'$K^{(2)}_S K^{(2)}_S$', c='b')
	plt.text(x=0.096, y=283, s=r'$K^{(2)}_S G^{(2)}_S$ (22)', c='magenta', fontsize=13) # 0.0269
	plt.text(x=0.0155, y=115, s=r'$K^{(2)}_S G^{(2)}_S$ (13)', c='r', fontsize=13) # label=r'$KG13$',
	plt.text(x=0.01, y=16, s=r'$K^{(3)}_S$', c='lime') # label=r'$3K13$'

	# plt.grid(ls=':')
	# plt.legend(frameon=False, loc='center left', fontsize=14)
	# plt.xlim(1e-3,3e0)
	plt.xlim(3e-3,0.3)
	plt.ylim(1e1,4e4)
	plt.tick_params(right=True, top=True, which='both')
	# plt.xticks([1e-3,1e-2,1e-1,1e0])
	plt.xlabel(r'Wavenumber $k$ [h Mpc$^{-1}$]')
	plt.ylabel(r'$P^0_{\theta\theta}(k)\,$ [h$^{-3}$ Mpc$^3$]')
	plt.show()

def plot_oneloop_theta():
	k, PL, P13, P22, P_1loop = power.Ptt_1loop(k=None, PL=None, get_compts=True, N=1024)

	fig, ax = plt.subplots(figsize=(6,6))
	ax.loglog(k, P_1loop, 'k', label=r'$P_L+P_{22}+2P_{13}$', lw=1.4)
	kp, Pp, kn, Pn = find_ind(k, P22+2*P13)
	ax.loglog(kp, Pp, 'b', label=r'$P_{22}+2P_{13}$', lw=1.2)
	ax.loglog(kn, np.abs(Pn), 'b--', lw=1.2)
	# ax.loglog(k, np.abs(P22+2*P13), 'b', label=r'$|P_{22}+2P_{13}|$', lw=1.2)
	ax.loglog(k, P22, 'r', label=r'$P_{22}$', lw=1.2)
	ax.loglog(k, np.abs(2*P13), 'lime', ls='--', label=r'$2P_{13}$', lw=1.2)
	ax.loglog(k, PL, 'gray', ls=':', label=r'$P_L$', lw=1.4)

	ax.legend(frameon=False, loc='upper right', fontsize=13)
	ax.set_xlim(2e-4,1e2)
	ax.set_ylim(1e0,1e5)
	ax.tick_params(right=True, top=True, which='both')

	ax.xaxis.set_minor_locator(locmin)
	ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
	ax.set_xticks([1e-3,1e-2,1e-1,1e0,1e1,1e2])
	ax.set_xlabel(r'Wavenumber $k$ [h Mpc$^{-1}$]')
	ax.set_ylabel(r'$P_{\theta\theta}(k)$ [h$^{-3}$ Mpc$^3$]')
	plt.show()

def plot_oneloop_matter():
	k, PL, P13, P22, P_1loop = power.Pmm_1loop(k=None, PL=None, get_compts=True, N=1024)

	fig, ax = plt.subplots(figsize=(6,6))
	ax.loglog(k, P_1loop, 'k', label=r'$P_L+P_{22}+2P_{13}$', lw=1.4)
	kp, Pp, kn, Pn = find_ind(k, P22+2*P13)
	ax.loglog(kp, Pp, 'b', label=r'$P_{22}+2P_{13}$', lw=1.2)
	ax.loglog(kn, np.abs(Pn), 'b--', lw=1.2)
	ax.loglog(k, P22, 'r', label=r'$P_{22}$', lw=1.2)
	ax.loglog(k, np.abs(2*P13), 'lime', ls='--', label=r'$2P_{13}$', lw=1.2)
	ax.loglog(k, PL, 'gray', ls=':', label=r'$P_L$', lw=1.4)

	# ax.grid(ls=':')
	ax.legend(frameon=False, loc='upper right', fontsize=13)
	ax.set_xlim(2e-4,1e2)
	ax.set_ylim(1e0,1e5)
	ax.tick_params(right=True, top=True, which='both')

	ax.xaxis.set_minor_locator(locmin)
	ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
	ax.set_xticks([1e-3,1e-2,1e-1,1e0,1e1,1e2])
	ax.set_xlabel(r'Wavenumber $k$ [h Mpc$^{-1}$]')
	ax.set_ylabel(r'$P_{mm}(k)$ [h$^{-3}$ Mpc$^3$]')
	plt.show()

def plot_Ps_vv_with_ratio(N=512): # P(k,mu) for diff mu
	H0f = 51.57 # Om^0.55=0.3^0.55=0.5157
	kin, plin = np.loadtxt('Pk_Planck15_large.dat', unpack=True, skiprows=4)
	F = power.Ps_tt(kin, plin, N=N)

	k = F.k
	mu1, mu2, mu3, mu4 = 1.0, 0.6, 0.3, 0.1
	Pvv1_norsd = F.Pvv_norsd(mu1)
	Pvv2_norsd = F.Pvv_norsd(mu2)
	Pvv3_norsd = F.Pvv_norsd(mu3)
	Pvv4_norsd = F.Pvv_norsd(mu4)
	Psvv1 = F.Psvv(mu1, with_fog=False)
	Psvv2 = F.Psvv(mu2, with_fog=False)
	Psvv3 = F.Psvv(mu3, with_fog=False)
	Psvv4 = F.Psvv(mu4, with_fog=False)
	fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(6,8), gridspec_kw={'height_ratios': [2.5, 1], 'hspace': 0.0})
	ax[0].loglog(k, Psvv1, 'k', lw=1.2, label=r'$\mu=1.0$')
	ax[0].loglog(k, Psvv2, 'b', lw=1.2, label=r'$\mu=0.6$')
	ax[0].loglog(k, Psvv3, 'r', lw=1.2, label=r'$\mu=0.3$')
	ax[0].loglog(k, Psvv4, 'lime', lw=1.2, label=r'$\mu=0.1$')
	ax[0].loglog(k, Pvv1_norsd, 'k', ls=':', lw=1.7)
	ax[0].loglog(k, Pvv2_norsd, 'b', ls=':', lw=1.7)
	ax[0].loglog(k, Pvv3_norsd, 'r', ls=':', lw=1.7)
	ax[0].loglog(k, Pvv4_norsd, 'lime', ls=':', lw=1.5)

	ax[1].semilogx(k, Psvv1/Pvv1_norsd, 'k', lw=1.2)
	ax[1].semilogx(k, Psvv2/Pvv2_norsd, 'b', lw=1.2)
	ax[1].semilogx(k, Psvv3/Pvv3_norsd, 'r', lw=1.2)
	ax[1].semilogx(k, Psvv4/Pvv4_norsd, 'lime', lw=1.2)

	ax[0].legend(frameon=False, loc='upper right', fontsize=16)
	ax[1].text(x=4e-3, y=0.4, s=r'$P^s_{vv}(k,\mu)\,/\,P_{vv,no\:RSD}(k,\mu)$', color='k', fontsize=18)
	ax[1].set_yticks([0.4,0.6,0.8,1.0])
	ax[0].set_xlim(3e-3,0.24)
	ax[0].set_ylim(8e0*H0f**2, 2e9*H0f**2)
	ax[1].set_ylim(0.3,1.05)
	ax[0].tick_params(right=True, top=True, which='both')
	ax[1].tick_params(right=True, top=True, which='both')
	ax[1].set_xlabel(r'Wavenumber $k$ [h Mpc$^{-1}$]')
	ax[0].set_ylabel(r'$P^s_{vv}(k,\mu)$  [$(km/s)^2\, (h^{-1}\, Mpc)^3$]')
	ax[1].set_ylabel(r'Ratio')
	ax[0].yaxis.set_minor_locator(locmin)
	ax[0].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
	plt.show()

def plot_Ps_vv_with_ratio2(N=512): # Pvv^ell
	H0f = 51.57
	kin, plin = np.loadtxt('Pk_Planck15_large.dat', unpack=True, skiprows=4)
	F = power.Ps_tt(kin, plin, N=N)

	P0vv = F.Psvv_ell(ell=0, with_fog=False)
	P2vv = F.Psvv_ell(ell=2, with_fog=False)
	P4vv = F.Psvv_ell(ell=4, with_fog=False)
	P6vv = F.Psvv_ell(ell=6, with_fog=False)
	fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(6,8), gridspec_kw={'height_ratios': [2.5, 1], 'hspace': 0.0})
	ax[0].loglog(F.k, P0vv, 'k', lw=1.2, label=r'$\ell=0$')
	ax[0].loglog(F.k, P2vv, 'b', lw=1.2, label=r'$\ell=2$')

	pos_signal = np.ma.masked_where(P4vv<=0.0, P4vv)
	neg_signal = np.ma.masked_where(P4vv>0.0, P4vv)
	ax[0].loglog(F.k, pos_signal, 'r', lw=1.2, label=r'$\ell=4$')
	ax[0].loglog(F.k, np.abs(neg_signal), 'r--', dashes=(5,3), lw=1.2)

	ax[0].loglog(F.k, P6vv, 'lime', lw=1.2, label=r'$\ell=6$')

	ax[0].loglog(F.k, F.P0vv_norsd, 'k:', lw=1.7)
	ax[0].loglog(F.k, F.P2vv_norsd, 'b:', lw=1.7)

	ax[1].semilogx(F.k, P0vv/F.P0vv_norsd, 'k', lw=1.2, label=r'$P^0_{vv}\,/\,P^0_{vv,no\:RSD}$')
	ax[1].semilogx(F.k, P2vv/F.P2vv_norsd, 'b', lw=1.2, label=r'$P^2_{vv}\,/\,P^2_{vv,no\:RSD}$')

	ax[0].legend(frameon=False, loc='upper right', fontsize=18, ncol=1)
	ax[1].legend(frameon=False, loc='lower left', fontsize=18, ncol=1)
	ax[1].set_yticks([0.4,0.6,0.8,1.0])
	ax[0].set_xlim(3e-3,0.24)
	ax[0].set_ylim(8e0*H0f**2,2e9*H0f**2)
	ax[1].set_ylim(0.3,1.05)
	ax[0].tick_params(right=True, top=True, which='both')
	ax[1].tick_params(right=True, top=True, which='both')
	ax[1].set_xlabel(r'Wavenumber $k$ [h Mpc$^{-1}$]')
	ax[0].set_ylabel(r'$P^\ell_{vv}(k)$  [$(km/s)^2\, (h^{-1}\, Mpc)^3$]')
	ax[1].set_ylabel(r'Ratio')
	ax[0].yaxis.set_minor_locator(locmin)
	ax[0].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
	plt.show()

def plot_Ps_vv_disp_with_ratio(N=512): # P(k,mu) for diff mu
	H0f = 51.57 # Om^0.55=0.3^0.55=0.5157
	sig_fog = 3.5
	kin, plin = np.loadtxt('Pk_Planck15_large.dat', unpack=True, skiprows=4)
	F = power.Ps_tt(kin, plin, sig_fog=sig_fog, N=N)

	k = F.k
	mu1, mu2, mu3, mu4 = 1.0, 0.6, 0.3, 0.1
	Pvv1_norsd = F.Pvv_norsd(mu1)
	Pvv2_norsd = F.Pvv_norsd(mu2)
	Pvv3_norsd = F.Pvv_norsd(mu3)
	Pvv4_norsd = F.Pvv_norsd(mu4)
	Psvv1_disp = F.Psvv(mu1, with_fog=True)
	Psvv2_disp = F.Psvv(mu2, with_fog=True)
	Psvv3_disp = F.Psvv(mu3, with_fog=True)
	Psvv4_disp = F.Psvv(mu4, with_fog=True)


	fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(6,8), gridspec_kw={'height_ratios': [2.5, 1], 'hspace': 0.0})
	ax[0].loglog(k, Psvv1_disp, 'k', lw=1.2, label=r'$\mu=1.0$')
	ax[0].loglog(k, Psvv2_disp, 'b', lw=1.2, label=r'$\mu=0.6$')
	ax[0].loglog(k, Psvv3_disp, 'r', lw=1.2, label=r'$\mu=0.3$')
	ax[0].loglog(k, Psvv4_disp, 'lime', lw=1.2, label=r'$\mu=0.1$')
	ax[0].loglog(k, Pvv1_norsd, 'k', ls=':', lw=1.7)
	ax[0].loglog(k, Pvv2_norsd, 'b', ls=':', lw=1.7)
	ax[0].loglog(k, Pvv3_norsd, 'r', ls=':', lw=1.7)
	ax[0].loglog(k, Pvv4_norsd, 'lime', ls=':', lw=1.5)

	ax[1].semilogx(k, Psvv1_disp/Pvv1_norsd, 'k', lw=1.2)
	ax[1].semilogx(k, Psvv2_disp/Pvv2_norsd, 'b', lw=1.2)
	ax[1].semilogx(k, Psvv3_disp/Pvv3_norsd, 'r', lw=1.2)
	ax[1].semilogx(k, Psvv4_disp/Pvv4_norsd, 'lime', lw=1.2)

	# uncomment to add more clutter to the plot
	# Ps1 = F.Ps(mu1) * (H0f*mu1/k)**2 # no damping
	# Ps2 = F.Ps(mu2) * (H0f*mu2/k)**2
	# Ps3 = F.Ps(mu3) * (H0f*mu3/k)**2
	# Ps4 = F.Ps(mu4) * (H0f*mu4/k)**2
	# ax[1].semilogx(k, Ps1/Pvv1_norsd, 'k:', lw=1.2)
	# ax[1].semilogx(k, Ps2/Pvv2_norsd, 'b:', lw=1.2)
	# ax[1].semilogx(k, Ps3/Pvv3_norsd, 'r:', lw=1.2)
	# ax[1].semilogx(k, Ps4/Pvv4_norsd, 'lime', ls=':', lw=1.2)

	ax[0].legend(frameon=False, loc='upper right', fontsize=16)
	ax[1].text(x=4e-3, y=0.4, s=r'$P^s_{vv}(k,\mu)\,/\,P_{vv,no\:RSD}(k,\mu)$', color='k', fontsize=18)
	ax[1].set_yticks([0.4,0.6,0.8,1.0])
	ax[0].set_xlim(3e-3,0.24)
	ax[0].set_ylim(8e0*H0f**2,2e9*H0f**2)
	ax[1].set_ylim(0.3,1.05)
	ax[0].tick_params(right=True, top=True, which='both')
	ax[1].tick_params(right=True, top=True, which='both')
	ax[1].set_xlabel(r'Wavenumber $k$ [h Mpc$^{-1}$]')
	ax[0].set_ylabel(r'$P^s_{vv}(k,\mu)$  [$(km/s)^2\, (h^{-1}\, Mpc)^3$]')
	ax[1].set_ylabel(r'Ratio')
	ax[0].yaxis.set_minor_locator(locmin)
	ax[0].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
	plt.show()

def plot_Ps_vv_disp_with_ratio2(N=512): # Puu^ell for ell=0,2
	H0f = 51.57 # Om^0.55=0.3^0.55=0.5157
	sig_fog = 3.5 # 6.a
	kin, plin = np.loadtxt('Pk_Planck15_large.dat', unpack=True, skiprows=4)
	F = power.Ps_tt(kin, plin, sig_fog=sig_fog, N=N)

	P0vv_disp = F.Psvv_ell(ell=0, with_fog=True)
	P2vv_disp = F.Psvv_ell(ell=2, with_fog=True)
	P4vv_disp = F.Psvv_ell(ell=4, with_fog=True)
	P6vv_disp = F.Psvv_ell(ell=6, with_fog=True)
	P8vv_disp = F.Psvv_ell(ell=8, with_fog=True)
	P10vv_disp = F.Psvv_ell(ell=10, with_fog=True)
	P0vv = F.Psvv_ell(ell=0, with_fog=False)
	P2vv = F.Psvv_ell(ell=2, with_fog=False)

	fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(6,8), gridspec_kw={'height_ratios': [2.5, 1], 'hspace': 0.0})
	ax[0].loglog(F.k, P0vv_disp, 'k', lw=1.2, label=r'$\ell=0$')
	ax[0].loglog(F.k, P2vv_disp, 'b', lw=1.2, label=r'$\ell=2$')

	pos_signal = np.ma.masked_where(P4vv_disp<=0.0, P4vv_disp)
	neg_signal = np.ma.masked_where(P4vv_disp>0.0, P4vv_disp)
	ax[0].loglog(F.k, pos_signal, 'r', lw=1.2, label=r'$\ell=4$')
	ax[0].loglog(F.k, np.abs(neg_signal), 'r--', dashes=(5,3), lw=1.2)

	ax[0].loglog(F.k, P6vv_disp, 'lime', lw=1.2, label=r'$\ell=6$')

	pos_signal = np.ma.masked_where(P8vv_disp<=0.0, P8vv_disp)
	neg_signal = np.ma.masked_where(P8vv_disp>0.0, P8vv_disp)
	ax[0].loglog(F.k, pos_signal, 'cyan', lw=1.2, label=r'$\ell=8$')
	ax[0].loglog(F.k, np.abs(neg_signal), 'cyan', ls='--', dashes=(5,3), lw=1.2)

	pos_signal = np.ma.masked_where(P10vv_disp<=0.0, P10vv_disp)
	neg_signal = np.ma.masked_where(P10vv_disp>0.0, P10vv_disp)
	ax[0].loglog(F.k, pos_signal, 'magenta', lw=1.2, label=r'$\ell=10$')
	ax[0].loglog(F.k, np.abs(neg_signal), 'magenta', ls='--', dashes=(5,3), lw=1.2)

	ax[0].loglog(F.k, F.P0vv_norsd, 'k:', lw=1.7)
	ax[0].loglog(F.k, F.P2vv_norsd, 'b:', lw=1.7)

	ax[1].semilogx(F.k, P0vv_disp/F.P0vv_norsd, 'k', lw=1.2, label=r'$P^0_{vv}\,/\,P^0_{vv,no\:RSD}$')
	ax[1].semilogx(F.k, P2vv_disp/F.P2vv_norsd, 'b', lw=1.2, label=r'$P^2_{vv}\,/\,P^2_{vv,no\:RSD}$')
	ax[1].semilogx(F.k, P0vv/F.P0vv_norsd, 'k:', lw=1.2)
	ax[1].semilogx(F.k, P2vv/F.P2vv_norsd, 'b:', lw=1.2)

	# ax[0].text(x=4e-3, y=5e4, s='with damping', color='k', fontsize=18)
	ax[0].legend(frameon=False, loc='upper right', fontsize=16, ncol=2, \
				columnspacing=0.8, handlelength=1.2, handletextpad=0.5)
	ax[1].legend(frameon=False, loc='lower left', fontsize=18, ncol=1)
	ax[1].set_yticks([0.4,0.6,0.8,1.0])
	ax[0].set_xlim(3e-3,0.24)
	ax[0].set_ylim(8e0*H0f**2,2e9*H0f**2)
	ax[1].set_ylim(0.3,1.05)
	ax[0].tick_params(right=True, top=True, which='both')
	ax[1].tick_params(right=True, top=True, which='both')
	ax[1].set_xlabel(r'Wavenumber $k$ [h Mpc$^{-1}$]')
	ax[0].set_ylabel(r'$P^\ell_{vv}(k)$  [$(km/s)^2\, (h^{-1}\, Mpc)^3$]')
	ax[1].set_ylabel(r'Ratio')
	ax[0].yaxis.set_minor_locator(locmin)
	ax[0].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
	plt.show()

# cumulant expansion model

def plot_Pvv_cumexp2(N=1024):
	H0f = 100. * 0.3**0.55
	kin, plin = np.loadtxt('Pk_Planck15_large.dat', unpack=True, skiprows=4)
	F = power.Ps_vv_cumexp(kin, plin, N=N)
	k = F.k

	# A (bispectrum term = 22 + 13)
	P0vv_KA = F.P0uu_KA * F.H0f**2
	P2vv_KA = F.P2uu_KA * F.H0f**2
	P4vv_KA = F.P4uu_KA * F.H0f**2
	P6vv_KA = F.P6uu_KA * F.H0f**2

	# B (pure 22 loop)
	P0vv_KB = F.P0uu_KB * F.H0f**2
	P2vv_KB = F.P2uu_KB * F.H0f**2
	P4vv_KB = F.P4uu_KB * F.H0f**2
	P6vv_KB = F.P6uu_KB * F.H0f**2

	fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20,6), sharex=True, sharey=True, gridspec_kw={'wspace': 0.05})

	ax[0].loglog(k, F.P0vv_norsd, 'k', ls=':', lw=1.7)
	ax[1].loglog(k, F.P2vv_norsd, 'k', ls=':', lw=1.7)

	# A
	color = 'lime'
	pos_signal = np.ma.masked_where(P0vv_KA<=0.0, P0vv_KA)
	neg_signal = np.ma.masked_where(P0vv_KA>0.0, P0vv_KA)
	ax[0].loglog(F.k, pos_signal, c=color, lw=1.4, label=r'$\ell=0$')
	ax[0].loglog(F.k, np.abs(neg_signal), c=color, ls='--', dashes=(5,3), lw=1.7)

	pos_signal = np.ma.masked_where(P2vv_KA<=0.0, P2vv_KA)
	neg_signal = np.ma.masked_where(P2vv_KA>0.0, P2vv_KA)
	ax[1].loglog(F.k, pos_signal, c=color, lw=1.4, label=r'$\ell=2$')
	ax[1].loglog(F.k, np.abs(neg_signal), c=color, ls='--', dashes=(5,3), lw=1.7)

	pos_signal = np.ma.masked_where(P4vv_KA<=0.0, P4vv_KA)
	neg_signal = np.ma.masked_where(P4vv_KA>0.0, P4vv_KA)
	ax[2].loglog(F.k, pos_signal, c=color, lw=1.4, label=r'$\ell=4$')
	ax[2].loglog(F.k, np.abs(neg_signal), c=color, ls='--', dashes=(5,3), lw=1.7)

	# B
	color = 'r'
	pos_signal = np.ma.masked_where(P0vv_KB<=0.0, P0vv_KB)
	neg_signal = np.ma.masked_where(P0vv_KB>0.0, P0vv_KB)
	ax[0].loglog(F.k, pos_signal, c=color, lw=1.4, label=r'$\ell=0$')
	ax[0].loglog(F.k, np.abs(neg_signal), c=color, ls='--', dashes=(5,3), lw=1.7)

	pos_signal = np.ma.masked_where(P2vv_KB<=0.0, P2vv_KB)
	neg_signal = np.ma.masked_where(P2vv_KB>0.0, P2vv_KB)
	ax[1].loglog(F.k, pos_signal, c=color, lw=1.4, label=r'$\ell=2$')
	ax[1].loglog(F.k, np.abs(neg_signal), c=color, ls='--', dashes=(5,3), lw=1.7)

	pos_signal = np.ma.masked_where(P4vv_KB<=0.0, P4vv_KB)
	neg_signal = np.ma.masked_where(P4vv_KB>0.0, P4vv_KB)
	ax[2].loglog(F.k, pos_signal, c=color, lw=1.4, label=r'$\ell=4$')
	ax[2].loglog(F.k, np.abs(neg_signal), c=color, ls='--', dashes=(5,3), lw=1.7)

	pos_signal = np.ma.masked_where(P6vv_KB<=0.0, P6vv_KB)
	neg_signal = np.ma.masked_where(P6vv_KB>0.0, P6vv_KB)
	ax[2].loglog(F.k, pos_signal, 'magenta', lw=1.4, label=r'$\ell=6$')
	ax[2].loglog(F.k, np.abs(neg_signal), 'magenta', ls='--', dashes=(5,3), lw=1.7)

	# A+B
	color = 'blue'
	ApB0 = P0vv_KA + P0vv_KB
	pos_signal = np.ma.masked_where(ApB0<=0.0, ApB0)
	neg_signal = np.ma.masked_where(ApB0>0.0, ApB0)
	ax[0].loglog(F.k, pos_signal, c=color, lw=1.4, label=r'$\ell=0$')
	ax[0].loglog(F.k, np.abs(neg_signal), c=color, ls='--', dashes=(5,3), lw=1.7)

	ApB2 = P2vv_KA + P2vv_KB
	pos_signal = np.ma.masked_where(ApB2<=0.0, ApB2)
	neg_signal = np.ma.masked_where(ApB2>0.0, ApB2)
	ax[1].loglog(F.k, pos_signal, c=color, lw=1.4, label=r'$\ell=2$')
	ax[1].loglog(F.k, np.abs(neg_signal), c=color, ls='--', dashes=(5,3), lw=1.7)

	ApB4 = P4vv_KA + P4vv_KB
	pos_signal = np.ma.masked_where(ApB4<=0.0, ApB4)
	neg_signal = np.ma.masked_where(ApB4>0.0, ApB4)
	ax[2].loglog(F.k, pos_signal, c=color, lw=1.4, label=r'$\ell=4$')
	ax[2].loglog(F.k, np.abs(neg_signal), c=color, ls='--', dashes=(5,3), lw=1.7)

	# All together
	ax[0].loglog(F.k, F.P0vv_norsd + ApB0, 'k', lw=1.4)
	ax[1].loglog(F.k, F.P2vv_norsd + ApB2, 'k', lw=1.4)


	ax[0].text(0.06, 0.58, s=r'$A_\ell+B_\ell$', transform=ax[0].transAxes, fontsize=24, color='b')
	ax[0].text(0.74, 0.24, s=r'$B_\ell$', transform=ax[0].transAxes, fontsize=24, color='r')
	ax[0].text(0.5, 0.44, s=r'$A_\ell$', transform=ax[0].transAxes, fontsize=24, color='lime')
	ax[2].text(0.55, 0.16, s=r'$B_6$', transform=ax[2].transAxes, fontsize=24, color='magenta')

	kwargs = {'verticalalignment': 'top', 'horizontalalignment':'right'}
	ax[0].text(0.95, 0.95, s=r'Monopole ($\ell=0$)', transform=ax[0].transAxes, fontsize=26, **kwargs)
	ax[1].text(0.95, 0.95, s=r'Quadrupole ($\ell=2$)', transform=ax[1].transAxes, fontsize=26, **kwargs)
	ax[2].text(0.95, 0.95, s=r'Hexadecapole ($\ell=4$)', transform=ax[2].transAxes, fontsize=26, **kwargs)

	ax[0].set_xlim(3e-3,0.24)
	ax[0].set_ylim(8e0*H0f**2,2e9*H0f**2)
	ax[0].tick_params(right=True, top=True, which='both')
	ax[1].tick_params(right=True, top=True, which='both')
	ax[2].tick_params(right=True, top=True, which='both')
	ax[1].set_xlabel(r'Wavenumber $k$ [h Mpc$^{-1}$]', fontsize=24)
	ax[0].set_ylabel(r'Multipole moment  [$(km/s)^2\, (h^{-1}\, Mpc)^3$]', fontsize=22)
	ax[0].yaxis.set_minor_locator(locmin)
	ax[0].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
	plt.show()

def plot_Pvv_cumexp_ratio_disp(N=1024):
	H0f = 51.57
	sig_fog = 3.5
	kin, plin = np.loadtxt('Pk_Planck15_large.dat', unpack=True, skiprows=4)
	F = power.Ps_vv_cumexp(kin, plin, sig_fog=sig_fog, N=N)
	k = F.k

	fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(6,8), gridspec_kw={'height_ratios': [2.5, 1], 'hspace': 0.0})
	ax[0].loglog(k, F.P0vv_norsd, 'k:', lw=1.7)
	ax[0].loglog(k, F.P2vv_norsd, 'b:', lw=1.7)

	P0vv_disp = F.Psvv_ell(ell=0, with_fog=True)
	P2vv_disp = F.Psvv_ell(ell=2, with_fog=True)
	P4vv_disp = F.Psvv_ell(ell=4, with_fog=True)
	P6vv_disp = F.Psvv_ell(ell=6, with_fog=True)
	P8vv_disp = F.Psvv_ell(ell=8, with_fog=True)
	P10vv_disp = F.Psvv_ell(ell=10, with_fog=True)

	ax[0].loglog(k, P0vv_disp, 'k', lw=1.2, label=r'$\ell=0$')
	ax[0].loglog(k, P2vv_disp, 'b', lw=1.2, label=r'$\ell=2$')

	pos_signal = np.ma.masked_where(P4vv_disp<=0.0, P4vv_disp)
	neg_signal = np.ma.masked_where(P4vv_disp>0.0, P4vv_disp)
	ax[0].loglog(k, pos_signal, 'r', lw=1.2, label=r'$\ell=4$')
	ax[0].loglog(k, np.abs(neg_signal), 'r--', dashes=(5,3), lw=1.2)

	pos_signal = np.ma.masked_where(P6vv_disp<=0.0, P6vv_disp)
	neg_signal = np.ma.masked_where(P6vv_disp>0.0, P6vv_disp)
	ax[0].loglog(k, pos_signal, 'lime', lw=1.2, label=r'$\ell=6$')
	ax[0].loglog(k, np.abs(neg_signal), 'lime', ls='--', dashes=(5,3), lw=1.2)

	pos_signal = np.ma.masked_where(P8vv_disp<=0.0, P8vv_disp)
	neg_signal = np.ma.masked_where(P8vv_disp>0.0, P8vv_disp)
	ax[0].loglog(k, pos_signal, 'cyan', lw=1.2, label=r'$\ell=8$')
	ax[0].loglog(k, np.abs(neg_signal), 'cyan', ls='--', dashes=(5,3), lw=1.2)

	pos_signal = np.ma.masked_where(P10vv_disp<=0.0, P10vv_disp)
	neg_signal = np.ma.masked_where(P10vv_disp>0.0, P10vv_disp)
	ax[0].loglog(k, pos_signal, 'magenta', lw=1.2, )#label=r'$\ell=10$')
	ax[0].loglog(k, np.abs(neg_signal), 'magenta', ls='--', dashes=(5,3), lw=1.2)

	ax[1].semilogx(k, P0vv_disp/F.P0vv_norsd, 'k', lw=1.2, label=r'$P^0_{vv}\,/\,P^0_{vv,no\:RSD}$')
	ax[1].semilogx(k, P2vv_disp/F.P2vv_norsd, 'b', lw=1.2, label=r'$P^2_{vv}\,/\,P^2_{vv,no\:RSD}$')

	ax[0].legend(frameon=False, loc='upper right', fontsize=16, ncol=1)
	ax[1].legend(frameon=False, loc='lower left', fontsize=18, ncol=1)
	ax[1].set_yticks([0.4,0.6,0.8,1.0])
	ax[0].set_xlim(3e-3,0.24)
	ax[0].set_ylim(8e0*H0f**2,2e9*H0f**2)
	ax[1].set_ylim(0.3,1.05)
	ax[0].tick_params(right=True, top=True, which='both')
	ax[1].tick_params(right=True, top=True, which='both')
	ax[1].set_xlabel(r'Wavenumber $k$ [h Mpc$^{-1}$]')
	ax[0].set_ylabel(r'$P^\ell_{vv}(k)$  [$(km/s)^2\, (h^{-1}\, Mpc)^3$]')
	ax[1].set_ylabel(r'Ratio')
	ax[0].yaxis.set_minor_locator(locmin)
	ax[0].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
	plt.show()


plot_oneloop_theta()


