from functions import *

## set the cosmolog ##
from astropy.cosmology import Planck18_arXiv_v2 as Planck18
default_cosmology.set(Planck18)


### Reading the SKA simulation data stored in a file ## 
### this is a simulation of SKA-1 low with HI cube + point sources (EN1 catalogue) at z = 9 as sky model. I have simulated this with OSKAR ##

data_array_ska = np.fromfile('data_path/ska_9_sem_light_en1_data', dtype=np.complex) # provide the data path. One can load the data from fits file too using pyuvdata.

## There are 128 channels and 5274720 baselines ##

data_ska = np.transpose(data_array_ska.reshape(128, 5274720)) 

freq = (np.fromfile('data_path/SKA_HI_z_9_freq', dtype=float)*units.Hz)


## Defining data parameters #

Nbls = data_ska.shape[0]
Nfreqs = freq.shape[0]
vis_units = 'Jy'
Npols = 1  ## Just RR polarization 
Nspws = 1 ## one spectral window 

freq_array = freq.reshape(1,Nfreqs)
redshift = calc_z(freq_array).mean(axis=1)

# Reading UVW array ##

uvw = np.fromfile('data_path/SKA_HI_z_9_UVW_EN1', dtype=float)
uvw = np.transpose(uvw.reshape(3,Nbls))

uvw_dist = [np.linalg.norm(uvw[i,:]) for i in range(Nbls)]
uvw_length = uvw_dist*units.m



############## Choosing baseline upto 500m. Do it as you wish  ##############################################

ia = np.where((uvw_length.value >0.0) & (uvw_length.value < 500.0))
xa = list(set(ia[0]))

data_array = data_ska[xa,:]  # We have taken only baselines below 500 meters 
uvw_cent = uvw[xa,:]


Nblts = data_array.shape[0]  ## This is the number of baselines between 500m


########################################################################
## Generating a array with all False. This will be inverted inside the delay_transform function. 
## Hence, that will give all the entry true, i.e. no flagging. All channels are present.  


no_flag_array = np.zeros(shape=(Nblts, Nfreqs),dtype=np.bool)


delay_vis_without_BH_without_flag = np.asarray(delay_transform_without_window(data_array, flag_array = no_flag_array, freq_array = freq_array, inverse= False, shift = True))



#############################################################################

## Introducing flagging ### 

flag = np.copy(no_flag_array)

for k in range(Nblts):
    ind = random.sample(range(Nfreqs),16)
    flag[k,ind] = np.ones(16,dtype = np.bool)  ## This create a random flagged array for all baselines at each time stamps with 16 randomly freq samples are flagged. 
         


flag_array = np.logical_not(flag).astype(float)  ## This gives flag array in flaot (1/0) values. 
flagged_vis = data_array*flag_array

bandpass = np.ones_like(no_flag_array).astype(float)  ## this will give me flat bandpass 
bandpass_with_flag = bandpass*flag_array




#### CLEAN the whole data with parallel processing ### 
### You need to clean the data and bandpass kernel without fftshift ###

visibility_delay_without_BH = delay_transform_without_window(data_array, flag_array = flag, freq_array = freq_array, delay_array=None, inverse= False, shift = False) ##transform without BH

lag_kernel_without_BH = delay_transform_without_window(bandpass, flag_array= flag, freq_array = freq_array, delay_array = None, shift = False,inverse=False) ## deconvolution kernel without BH



### Define the empty dictionary  ## 

clean_freq_dict = {}
clean_delay_net_dict = {} 

##  CLEAN params ## 

## Use these parameters as it fits for your data set 

gain = 0.1
maxiter = 100000
threshold_type = 'relative'
threshold = 5e-3
verbose = False
nproc = 24


#############################################################################

clean_components = np.zeros_like(data_array)   ## a array to store 'CLEAN' components 
clean_res = np.zeros_like(data_array)          ## a array to store 'Residual' components    
clean_comp_restored = np.zeros_like(data_array)  ## array to store the cleaned vis 


#### Store the visibilities and kernels and all params in list to process in parallel

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
list_of_verbosity = [verbose]*Nblts
list_of_pid = range(Nblts)
list_of_progressbars = [False] *Nblts
list_of_progressbar_ylocs = np.arange(Nblts) % (nproc)#, WM.term.height)
list_of_progressbar_ylocs = list_of_progressbar_ylocs.tolist()

############# Starting parallel processing #############
    
pool = MP.Pool(processes=nproc) 

list_of_cleanstates = pool.map(complex1dClean_arg_splitter, zip(list_of_vis_lag, list_of_dkern, list_of_cboxes, list_of_gains, list_of_maxiter, list_of_thresholds, list_of_threshold_types, list_of_verbosity, list_of_progressbars, list_of_pid, list_of_progressbar_ylocs))
    
               
for bli in range(Nblts):
     
    cleanstate = list_of_cleanstates[bli]
    clean_components[bli,:] = cleanstate['cc']
    clean_res[bli,:] = cleanstate['res']




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




