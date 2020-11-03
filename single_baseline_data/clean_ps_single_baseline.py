from functions import *

## set the cosmolog ##
from astropy.cosmology import Planck18_arXiv_v2 as Planck18
default_cosmology.set(Planck18)

### Reading the SKA simulation data stored in a file ## 
### this is a simulation of SKA-1 low with HI cube + point sources (EN1 catalogue) at z = 9 as sky model ##

data_array = np.load('data/data_single_baseline.pkl',allow_pickle=True)  # loading the data 




## Frequency info #
Nspws = 1 # single spectral window 

freq = (np.fromfile('data/SKA_HI_z_9_freq', dtype=float)*units.Hz)

Nfreqs = freq.shape[0]
freq_array = freq.reshape(1,Nfreqs)
redshift = calc_z(freq_array).mean(axis=1)

# Reading UVW array ##

uvw = np.load('data/uvw_single_baseline.pkl',allow_pickle=True)
uvw_dist = np.linalg.norm(uvw)
baseline_length = uvw_dist*units.m

mean_freq = np.mean(freq_array.value, axis=1) * freq_array.unit
uvw_wavelength = baseline_length / (const.c / mean_freq.to('1/s')).to('m')
k_perpendicular = u2kperp(uvw_wavelength , redshift,cosmo=None)   ## this gives me the k_perpendicular values for the baselines


## Calculation of  K_parallel ##

Ndelays = np.int(Nfreqs)
delays = np.fft.fftfreq(Ndelays, d=np.diff(freq_array[0])[0].value)
delays = np.fft.fftshift(delays) / freq_array.unit
delay_array = delays.to('ns').reshape(1,Ndelays)
k_parallel = eta2kparr(delay_array,redshift.reshape(Nspws, 1),cosmo=None)

delay_array_micro_s = delay_array/1e3  # In micro-sec


########################################################################
## Generating a array with all False. This will be inverted inside the delay_transform function. 
## Hence, that will give all the entry true, i.e. no flagging. All channels are present.  

no_flag_array = np.zeros(shape=(1, Nfreqs),dtype=np.bool) 


delay_vis_without_BH_without_flag = np.asarray(delay_transform_without_window(data_array, flag_array = no_flag_array, freq_array = freq_array, inverse= False, shift = True))

### plotting the visibility spectrum ### 

fig, ax = subplots(1, figsize=(9, 7))

ax.plot(freq_array[0,:].value/1e6 , np.abs(data_array), color='blue',label='visibility spectrum')


ax.set_xlabel(r'Freq [MHz]')
ax.set_ylabel(r'Vis $(\nu)$ [Jy]')
ax.set_ylim([0,4])
ax.legend()
pl.tight_layout()
pl.savefig('data/vis_spectrum.png')
pl.show()



############################# Plotting delay spectrum    ######################################################


eta = np.asarray([-8.0,-4.0,-2.0,0.0,2.0,4.0,8.0])*1.e-6*units.s ## choose between minimum and maximum of delay
k_par = (eta2kparr(eta, redshift).value).round(decimals=2)

bl_lengths = baseline_length.value
delay_horizon = (bl_lengths/FCNST.c)*1e6  ## delay horizon in micro sec

min_y = np.abs(delay_vis_without_BH_without_flag).min()
max_y = np.abs(delay_vis_without_BH_without_flag).max()

x1 = np.ones(100)*delay_horizon
y1 = np.linspace(min_y,max_y,100)

fig, ax = subplots(1, figsize=(9, 7))

ax.plot(delay_array_micro_s[0,:].value , np.abs(delay_vis_without_BH_without_flag[0,:]), color='red',label='delay spctrum')
ax.plot(x1,y1,linestyle='--',color='black',label='delay horizon')
ax.plot(-x1,y1,linestyle='--',color='black')

ax1 = ax.twiny()
ax1.set_xticks(k_par)
ax1.set_xlabel(r'$k_{\parallel} [h Mpc^{-1}]$')

ax.set_xlabel(r'Delay $[\mu s]$')
ax.set_ylabel(r'Vis $(\tau)$ [Jy]')

ax.annotate(r'|U| = {0} [m]'.format(bl_lengths.astype(int)), xy=(-6.0, 1e6), xytext=(-6.0,1e6), color='k',fontsize=14)

ax.set_yscale('log')
ax.legend()
pl.tight_layout()
pl.savefig('data/delay_spectrum.png')
pl.show()




#############################################################################

## Introducing flagging ### 

flag = np.copy(no_flag_array)

ind = random.sample(range(Nfreqs),16)
flag[0,ind] = np.ones(16,dtype = np.bool)  ## This create a random flagged array for all baselines at each time stamps with 16 randomly freq samples are flagged. 
         


flag_array = np.logical_not(flag).astype(float)  ## This gives flag array in flaot (1/0) values. 
flagged_vis = data_array*flag_array

bandpass = np.ones_like(no_flag_array).astype(float)  ## this will give me flat bandpass 
bandpass_with_flag = bandpass*flag_array


##### ### plotting the flagged visibility spectrum ### 

fig, ax = subplots(1, figsize=(9, 7))

ax.plot(freq_array[0,:].value/1e6 , np.abs(flagged_vis[0,:]), color='blue',label='flagged')
ax.plot(freq_array[0,:].value/1e6 , np.abs(data_array), color='green',label='no flag')

ax.set_xlabel(r'Freq [MHz]')
ax.set_ylabel(r'Flagged Vis $(\nu)$ [Jy]')

ax.annotate(r'|U| = {0} [m]'.format(baseline_length.value.astype(int)), xy=(144, 2e0), xytext=(144,2e0), color='k',fontsize=14)

#ax.set_yscale('log')
ax.set_ylim([0,4])
ax.legend()
pl.tight_layout()
pl.savefig('data/visibility_spectrum_flag.png')
pl.show()

### FT of the flagged vis with and without Blackman Harris window #### 

delay_vis_flag_with_BH = np.asarray(delay_transform(data_array, flag_array = flag, freq_array = freq_array, inverse= False, shift = True))

delay_vis_flag= np.asarray(delay_transform_without_window(data_array, flag_array = flag, freq_array = freq_array, inverse= False, shift = True))



############################# Plotting delay spectrum after flag   ######################################################


fig, ax = subplots(1, figsize=(9, 7))

ax.plot(delay_array_micro_s[0,:].value , np.abs(delay_vis_without_BH_without_flag[0,:]), color='red',label='delay spectrum without flag')
ax.plot(delay_array_micro_s[0,:].value , np.abs(delay_vis_flag[0,:]), color='green',label='delay spectrum with flag')
ax.plot(delay_array_micro_s[0,:].value , np.abs(delay_vis_flag_with_BH[0,:]), color='blue',label='delay spectrum with flag and BH window')
ax.plot(x1,y1,linestyle='--',color='black',label='delay horizon')
ax.plot(-x1,y1,linestyle='--',color='black')

ax1 = ax.twiny()
ax1.set_xticks(k_par)
ax1.set_xlabel(r'$k_{\parallel} [h Mpc^{-1}]$')

ax.set_xlabel(r'Delay $[\mu s]$')
ax.set_ylabel(r'Vis $(\tau)$ [Jy]')

ax.annotate(r'|U| = {0} [m]'.format(bl_lengths.astype(int)), xy=(-5, 1e6), xytext=(-5,1e6), color='k',fontsize=14)

ax.set_yscale('log')
ax.legend()
ax.set_ylim([1e2,1e8])

pl.tight_layout()
pl.savefig('data/delay_spectrum_flagged_vis.png')
pl.show()

## Try with CLEAN ## 

############## Delay transform the data array and the bandpass array #### 

### You need to clean the data and bandpass kernel without fftshift ###

visibility_delay_without_BH = delay_transform_without_window(data_array, flag_array = flag, freq_array = freq_array, delay_array=None, inverse= False, shift = False) ##transform without BH

lag_kernel_without_BH = delay_transform_without_window(bandpass, flag_array= flag, freq_array = freq_array, delay_array = None, shift = False,inverse=False) ## deconvolution kernel without BH

################################################################################

##  CLEAN params ## 

gain = 0.1
maxiter = 100000
threshold_type = 'relative'
threshold = 5e-3
verbose = False


### Doing the CLEAN ### 
cleanstate = complex1dClean(visibility_delay_without_BH[0,:].value, lag_kernel_without_BH[0,:].value, gain=gain, maxiter=maxiter, threshold=threshold, threshold_type=threshold_type, verbose=verbose)

clean_components = cleanstate['cc']
clean_res = cleanstate['res']


clean_vis_delay_space = np.fft.fftshift(clean_components, axes=-1)
clean_vis_res_delay_space = np.fft.fftshift(clean_res, axes=-1)
clean_vis_net_delay_space = np.asarray(clean_vis_delay_space + clean_vis_res_delay_space)

### To go to freq space of the cleaned vis ### 

lags = delays.to('s').value

deta =  lags[1] - lags[0]
vis_cc = np.fft.ifft(clean_components, axis = -1)*deta*Nfreqs
vis_res = np.fft.ifft(clean_res, axis =-1)*deta*Nfreqs

vis_cc_shift = np.fft.ifftshift(vis_cc,axes=-1)
vis_res_shift = np.fft.ifftshift(vis_res,axes=-1)

vis_cleaned_freq =  vis_cc_shift + vis_res_shift

vis_delay_cleaned_freq = np.asarray(delay_transform_without_window(vis_cleaned_freq, flag_array = no_flag_array[0,:], freq_array = freq_array, delay_array=None, inverse= False, shift = True)) ##transform without BH


############################# Plotting delay spectrum after flag and CLEAN ######################################################


fig, ax = subplots(1, figsize=(9, 7))

ax.plot(delay_array_micro_s[0,:].value , np.abs(delay_vis_without_BH_without_flag[0,:]), color='red',label='no flag')
ax.plot(delay_array_micro_s[0,:].value , np.abs(delay_vis_flag[0,:]), color='blue',label='flagged')
ax.plot(delay_array_micro_s[0,:].value , np.abs(delay_vis_flag_with_BH[0,:]), color='green',label='flagged and using BH')
ax.plot(delay_array_micro_s[0,:].value , np.abs(clean_vis_net_delay_space), color='cyan',label='clean')
#ax.plot(delay_array_micro_s[0,:].value , np.abs(vis_delay_cleaned_freq), color='black',label='clean freq')
ax.plot(x1,y1,linestyle='--',color='black',label='delay horizon')
#ax.plot(-x1,y1,linestyle='--',color='black')

#ax1 = ax.twiny()
#ax1.set_xticks(k_par[k_par>=0.0])
#ax1.set_xlabel(r'$k_{\parallel} [Mpc^{-1}]$')

ax.set_xlabel(r'Delay $[\mu s]$')
ax.set_ylabel(r'Vis $(\tau)$ [Jy]')

#ax.set_xlim([0,8.0])  ## Plotting the positive half only 
ax.set_ylim([8e2,1e7])

#ax.annotate(r'|U| = {0} [m]'.format(bl_lengths.astype(int)), xy=(-5, 1e6), xytext=(-5,1e6), color='k',fontsize=14)

ax.set_yscale('log')
ax.legend(ncol=2,loc='lower right')
pl.tight_layout()
pl.savefig('data/clean_spectrum_of_a_baseline_with_flag.png')
pl.show()



##################  Power spectrum, followinf Parsons 2012 delay spectrum approach #######

### Normalization ##
############################################################################## 

bandwidth = np.diff(freq_array[0])[0]*Nfreqs
cosmo = default_cosmology.get()
los_dist = cosmo.comoving_distance(redshift).to('Mpc')  # in MPc 
dlos_dist = (FCNST.c/1e3) * bandwidth * (1+redshift)**2 / rest_freq_HI / cosmo.H0.value / cosmo.efunc(redshift)  # in Mpc

f0 = freq_array[0,int(Nfreqs/2.0)]   ## This is the central freq 
wl = FCNST.c /f0.value  #This is wavelength corresponding to central frequency ## 

Jy2K = wl**2 * Jy / (2*FCNST.k)

omega_bw = 1 # For simplicity, say beam is one.It should be a beam integral as given in Parsons 2014. If one has the beam infor (fits/pyuv object) use beam3Dvol function (mentioned in functions.py) to estimate the beam volume. 
 
jacobian_1 = 1/omega_bw 
jacobian2 = los_dist**2 * dlos_dist / bandwidth.value

normalization_factor = jacobian_1*jacobian2*Jy2K**2  # Normalization constant of the Power spectrum Liu 2014 (appendix)

#### Calculation of normalized powerspectrum of the delay spectrum ### 


avg_power_without_BH_without_flag = (np.abs(delay_vis_without_BH_without_flag)**2.0)*normalization_factor
avg_power_with_BH_without_flag = (np.abs(delay_vis_flag_with_BH)**2.0)*normalization_factor
avg_power_with_flag_dict = (np.abs(delay_vis_flag)**2.0)*normalization_factor
avg_power_with_flag_with_clean = (np.abs(clean_vis_net_delay_space)**2.0)*normalization_factor



 

## plotting Normalized power spectrum of data ## 

x1 = np.ones(100)*delay_horizon
y1 = np.linspace(1e2,1e12,100)

fig, ax = subplots(1, figsize=(9, 7))

ax.plot(delay_array_micro_s[0,:].value , np.abs(avg_power_without_BH_without_flag[0,:]), color='red',label='No flag')
ax.plot(delay_array_micro_s[0,:].value , np.abs(avg_power_with_BH_without_flag[0,:]), color='navy',label='Flag with BH')
ax.plot(delay_array_micro_s[0,:].value , np.abs(avg_power_with_flag_dict[0,:]), color='green',label='Flag')
ax.plot(delay_array_micro_s[0,:].value , np.abs(avg_power_with_flag_with_clean), color='cyan',label='CLEAN')


ax.plot(x1,y1,linestyle='--',color='black',label='delay horizon')
ax.plot(-x1,y1,linestyle='--',color='black')

ax1 = ax.twiny()
ax1.set_xticks(k_par)
ax1.set_xlabel(r'$k_{\parallel} [h Mpc^{-1}]$')


ax.set_xlabel(r'delay [$\mu$ s]')
ax.set_ylabel(r'P(k$_\parallel$) [$\mathrm{K^{2} (Mpc/h)^{3}}$]')
ax.annotate(r'|U| = {0} [m]'.format(bl_lengths.astype(int)), xy=(-6, 1e11), xytext=(-6, 1e11), color='k',fontsize=14)

ax.set_yscale('log')
ax.set_ylim([1e4,1e12])
ax.legend(ncol=2,loc='lower right')
pl.tight_layout()
pl.savefig('data/Power_spectrum.png')
pl.show()







