from functions import *

## set the cosmolog ##
from astropy.cosmology import Planck18_arXiv_v2 as Planck18
default_cosmology.set(Planck18)


## Reading/ loading the uvdata ## 

uv = UVData()
uv.read_uvfits('6th_310MHz.fits')

## Defining data parameters #
Ntimes = uv.Ntimes
Nbls = uv.Nbls
Nblts = uv.Nblts
Nfreqs = uv.Nfreqs
vis_units = 'Jy'
Npols = uv.Npols
Nspws = uv.Nspws


polarization_array = uv.polarization_array

if vis_units == 'Jy':
    data_unit = units.Jy
elif vis_units == 'K str':
     data_unit = units.K * units.sr
else:
    data_unit = units.dimensionless_unscaled

freq_array = uv.freq_array * units.Hz

baseline_array = uv.get_baseline_nums()
redshift = calc_z(freq_array).mean(axis=1)

###########################################################

uvw = uv.uvw_array      ## this gives the UVW array 
baseline = [np.linalg.norm(uvw[i,:]) for i in range(Nblts)]  ## this gives the baselines from UVW array 
baseline_m = baseline*units.m   ## baseline length in m 
mean_freq = np.mean(freq_array.value, axis=1) * freq_array.unit
baseline_wave = baseline_m / (const.c / mean_freq.to('1/s')).to('m')   ## baseline unitless quantity after dividing it by wavelength in m. 
k_perpendicular = u2kperp(baseline_wave, redshift,cosmo=None)  ## this gives me the k_perpendicular values for all baselines


##########################################################

### Reading the data ##

temp_data = np.zeros(shape=(Npols,Nblts, Nfreqs),dtype=np.complex128)
temp_data[:,:,:] = get_data_array(uv)
data_array_all = copy.deepcopy(temp_data)

## Reading the flag array ##

temp_data = np.ones(shape=(Npols,Nblts, Nfreqs),dtype=np.bool)
temp_data[:, :,:] = get_flag_array(uv)
flag_array_all = copy.deepcopy(temp_data)

########## We are selecting a set of data which is <2000 m ########

ia = np.where((baseline_m.value > 0.0) & (baseline_m.value <= 2000.0))
xa = list(set(ia[0]))

bl_length = baseline_m[xa]  ## the baselines below 2km 
uvw_cent = uvw[xa]   ## UVW array for baslines below 2 km 

spacing = 38.0

lengths_bl = int(np.ceil((bl_length.value.max() - bl_length.value.min())/spacing)+1)
bl_length_grid = np.linspace(bl_length.value.min(), bl_length.value.max(),lengths_bl)



################################################################################


data_array = data_array_all[:,xa,:]
data_array_I = np.sum((np.asarray(data_array, dtype = np.complex128)), axis=0)/2.0

flag = flag_array_all[:,xa,:]  ## True means flag 
flag = flag[0][:]

flag_array = np.logical_not(flag).astype(float)  ## This gives flag array in flaot (1/0) values.  1 = no flag, 0 = flag 
flagged_vis = data_array*flag_array
bandpass = np.ones_like(no_flag_array).astype(float)  ## this will give me flat bandpass 
bandpass_with_flag = bandpass*flag_array


## Calculation of  K_parallel ##



Ndelays = np.int(Nfreqs)
delays = np.fft.fftfreq(Ndelays, d= abs(np.diff(freq_array[0])[0].value))
delays = np.fft.fftshift(delays) / freq_array.unit
delay_array = delays.to('ns').reshape(1,Ndelays)
k_parallel = eta2kparr(delay_array,redshift.reshape(Nspws, 1),cosmo=None)



############## Delay transform the data array and the bandpass array #### 

visibility_delay_without_BH = delay_transform_without_window(data_array_I, flag_array = flag_array, freq_array = freq_array, delay_array=None, inverse= False, shift = False)


lag_kernel_without_BH = delay_transform_without_window(bandpass, flag_array= flag_array, freq_array = freq_array, delay_array = None, shift = False,inverse=False)




################################################################################

### Define the empty array ## 

clean_restored_dict = {}
clean_delay_net_dict = {} 

### The parameter to CLEAN. Use it according to your data and machines ### 

gain = 0.1
maxiter = 25000
threshold_type = 'relative'
threshold = 5e-4
verbose = False
nproc = 84

#############################################################################
clean_components = np.zeros_like(visibility_delay_without_BH.value)   ## a array to store 'CLEAN' components 
clean_res = np.zeros_like(visibility_delay_without_BH.value)          ## a array to store 'Residual' components    
clean_comp_restored = np.zeros_like(visibility_delay_without_BH.value)

Nblts = data_array.shape[0]

list_of_vis_lag = []
list_of_dkern = []
list_of_cboxes = []
    
for baseline in range(Nblts):
    list_of_vis_lag += [visibility_delay_without_BH[baseline,:].value]
    list_of_dkern += [lag_kernel_without_BH[baseline,:].value]
    
    clean_area = None
    list_of_cboxes += [clean_area]



list_of_gains = [gain]*Nblts
list_of_maxiter = [maxiter]*Nblts
list_of_thresholds = [threshold]*Nblts
list_of_threshold_types = [threshold_type]*Nblts
#list_of_cboxes = [None]*Nbls*Ntimes
list_of_verbosity = [verbose]*Nblts
list_of_pid = range(Nblts)
# list_of_pid = [None] * self.ia.baselines.shape[0]*self.n_acc
list_of_progressbars = [False] *Nblts
list_of_progressbar_ylocs = np.arange(Nblts) % (nproc)#, WM.term.height)
list_of_progressbar_ylocs = list_of_progressbar_ylocs.tolist()


    
pool = MP.Pool(processes=nproc) 

list_of_noisy_cleanstates = pool.map(complex1dClean_arg_splitter, zip(list_of_vis_lag, list_of_dkern, list_of_cboxes, list_of_gains, list_of_maxiter, list_of_thresholds, list_of_threshold_types, list_of_verbosity, list_of_progressbars, list_of_pid, list_of_progressbar_ylocs))
    
    

               
for bli in range(Nblts):
    ind = bli 
 
    noisy_cleanstate = list_of_noisy_cleanstates[ind]
    clean_components[bli,:] = noisy_cleanstate['cc']
    clean_res[bli,:] = noisy_cleanstate['res']


clean_vis_delay_space = np.fft.fftshift(clean_components, axes=-1)
clean_vis_res_delay_space = np.fft.fftshift(clean_res, axes=-1)
clean_vis_net_delay_space = clean_vis_delay_space + clean_vis_res_delay_space
clean_delay_net_dict =  np.asarray(clean_vis_net_delay_space)


### To go to freq space of the cleaned vis ### 

lags = delays.to('s').value
deta =  lags[1] - lags[0]

vis_cc = np.fft.ifft(clean_components, axis = -1)*deta*Nfreqs
vis_res = np.fft.ifft(clean_res, axis =-1)*deta*Nfreqs

vis_cc_shift = np.fft.ifftshift(vis_cc,axes=-1)
vis_res_shift = np.fft.ifftshift(vis_res,axes=-1)

vis_cleaned_freq =  vis_cc_shift + vis_res_shift

clean_freq_net_dict = np.asarray(vis_cleaned_freq)


#########################################################

## saving the data in pkl files ###

import pickle

output_1 = open('clean_delay_net_dict_flag.pkl', 'wb')
pickle.dump(clean_delay_net_dict, output_1)
output_1.close()



output_2 = open('clean_freq_net_dict_flag.pkl', 'wb')
pickle.dump(clean_freq_net_dict, output_2)
output_1.close()

pool.terminate()
pool.close()







