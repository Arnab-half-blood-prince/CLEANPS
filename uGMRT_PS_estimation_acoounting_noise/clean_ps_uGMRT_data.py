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


lst_array = np.unique(uv.lst_array) * units.rad
lsts = (lst_array*12./np.pi*units.h).value
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

### Load and store  the data into arrays ##

temp_data = np.zeros(shape=(Npols,Nblts, Nfreqs),dtype=np.complex128)
temp_data[:,:,:] = get_data_array(uv)
data_array_all = copy.deepcopy(temp_data)

## Load and store  the the flag array ##

temp_data = np.ones(shape=(Npols,Nblts, Nfreqs),dtype=np.bool)
temp_data[:, :,:] = get_flag_array(uv)
flag_array_all = copy.deepcopy(temp_data)

########## We are selecting a set of data which is <2000 m ########

ia = np.where((baseline_m.value > 0.0) & (baseline_m.value <= 2000.0))
xa = list(set(ia[0]))

bl_length = baseline_m[xa]  ## the baselines below 2km 
uvw_cent = uvw[xa]   ## UVW array for baslines below 2 km 

data_array = data_array_all[:,xa,:]
data_array_I = np.sum((np.asarray(data_array, dtype = np.complex128)), axis=0)/2.0

flag = flag_array_all[:,xa,:]
flag = flag[0][:] # In this array True means flag, False means no flag
 

################################################################################
## Calculation of  K_parallel ##

Ndelays = np.int(Nfreqs)
delays = np.fft.fftfreq(Ndelays, d= abs(np.diff(freq_array[0])[0].value))
delays = np.fft.fftshift(delays) / freq_array.unit
delay_array = delays.to('ns').reshape(1,Ndelays)
k_parallel = eta2kparr(delay_array,redshift.reshape(Nspws, 1),cosmo=None)



## Reading the CLEAN data ### 
## The visibility data is already being CLEANed in delay domain and stored in pkl files ## 

import pickle as pkl 

infile_2 = open('clean_delay_net_dict_BH_padded_6th.pkl','rb')
clean_delay_net_dict_BH_padded = pkl.load(infile_2)


######################## Gridding the data ###########

u = uvw_cent[:,0]
v = uvw_cent[:,1]
w = uvw_cent[:,2]

binsize = 38.0 ## this is U = 1/theta_HPBW, where theta_HPBW is in radian. do this calculation for your data at corresponding freq and put the binsize accordingle  



xi = np.arange(u.min(), u.max()+binsize, binsize)  # x coordinates
yi = np.arange(v.min(), v.max()+binsize, binsize)  # y coordinates 
uu_g, vv_g = np.meshgrid(xi,yi)

grid = np.zeros(shape = uu_g.shape)

nrow, ncol = grid.shape

bins = {}   # this will store the number of data points goes into each cell 
indexes = {}
wherebin = np.copy(grid)
wherebin = wherebin.tolist()  # this will store the index of data points goes into each cell

u_c = []  # mean value of u points in a cell 
v_c = []  # mean value of v points in a cell
w_c = []  # mean value of w points in a cell



data_array_clean_cell = {} # This is a dictionary, where the data points within a uv-cell is stored. I will correlate among these points within a cell to estimte the power spectrum later. 


i = 0 #  initialization 

##########################################################

for row in range(nrow):
    for col in range(ncol):
        xc = uu_g[row, col]    # x coordinate.
        yc = vv_g[row, col]    # y coordinate.
 
        # find the position that xc and yc correspond to.
        posx = np.abs(u - xc)
        posy = np.abs(v - yc)
        ibin = np.logical_and(posx < binsize/2., posy < binsize/2.)
                 
        ind  = np.where(ibin == True)[0]
           
             
        # fill the bin.
        bin_u = u[ind]
        bin_v = v[ind]
        bin_w = w[ind]      
        
 
        wherebin[row][col] = ind
        d = np.abs(delay_vis_BH_dict_before[ind,:])
        if len(ind) >=2 and d.max() > 0.0:
               
                u_c.append(np.mean(bin_u))
                v_c.append(np.mean(bin_v))
                w_c.append(np.mean(bin_w)) 
                bins[i] = len(ind)
                indexes[i] = ind
                u_cell[i] = bin_u
                v_cell[i] = bin_v 
                w_cell[i] = bin_w  
       
                
                data_array_clean_cell[i] = clean_delay_net_dict_BH_padded[ind,:]

                i = i+1       
        else:
                u_c.append(np.nan)
                v_c.append(np.nan)   
                w_c.append(np.nan) 


######################################################
# Estimating power spectrum (not normalized yet)  ###

## I will take the off-diagonal terms of the correlation matrix for each uv-cell and put the mean of that.
## The off-doiagonal terms contain the cross-correlation of visibilities within that cell, hence assumed to be free of noise bias  

power_clean_off_diag = {}

baseline_m_grid = []
k_perpendicular_grid = []
u_g = []
v_g = []


for j in range(len(u_cell)):
        

    power_clean = remove_auto_correlations(cross_multiply_array(data_array_clean_cell[j]))

    power_clean_off_diag[j] =  (power_clean).mean(axis=0)


              
    baseline = np.sqrt((u_cell[j].mean())**2+(v_cell[j].mean())**2+(w_cell[j].mean())**2)
    baseline_m = baseline*units.m   ## baseline length in m 
    baseline_m_grid.append(baseline_m)
    baseline_wave = baseline_m / (const.c / mean_freq.to('1/s')).to('m')   
    k_perpendicular_grid.append(u2kperp(baseline_wave, redshift,cosmo=None)) 


    u_g.append((u_cell[j].mean())) 
    v_g.append((v_cell[j].mean()))      
                       

################################################################################

## Storing the power spectrum in a pkl file for later use ## 

output_1 = open('clean_power_spectrum.pkl', 'wb')
pkl.dump(power_clean_off_diag, output_1)
output_1.close()

# Read power spectrum file if already run and stored in a pkl file# # 

#infile_1 = open('clean_power_spectrum.pkl','rb')  ## this is true stokes I power 
#power_clean_off_diag = pkl.load(infile_1)


######################################################### 

## power spectrum containing the noise terms too ### 

## I will take the the diagonal terms of the correlation matrix for each uv-cell.
## The doiagonal terms contain the self-correlation of visibilities within that cell, hence contains the  noise bias. 
## Also, I am storing the number of points in each cell, which will give the weight or the number of data points averaged together.   


power_clean_diag = {}
uv_weight = {}


for j in range(len(u_cell)):
        

    power_clean = diagonal_term(cross_multiply_array(data_array_clean_cell[j]))

    power_clean_diag[j] =  (power_clean).mean(axis=0)

    uv_weight[j] = (power_clean.shape[0]) 

             

################################################################################
## Storing this  power with self-correlation and weights for future work ##

output_1 = open('power_clean_noise.pkl', 'wb')
pkl.dump(power_clean_diag, output_1)
output_1.close()


output_3 = open('weight_noise.pkl', 'wb')
pkl.dump(uv_weight, output_3)
output_3.close()

   
### Read the uGMRT beam information and calculation of the powe spectrum normalization  ### 


#########################################################

f = freq_array.reshape(Nfreqs) ## this is the freq array 
f0 = f[int(Nfreqs/2.0)]   ## This is the central freq 

from astropy.io import fits
hd = fits.getdata('healpix_GMRT_PB.fits')
beam_freqs = 3.10e8*units.Hz
extbeam = hd['PB_val']
extbeam = extbeam.reshape(-1,beam_freqs.size)
ia = np.isnan(extbeam)
extbeam[ia] = 0 
beam_nside = hp.npix2nside(extbeam.shape[0])
beam = extbeam[:,0]

omega_bw = beam3Dvol(beam, f.value, freq_wts=None, hemisphere=True)  ##This give the beam in sr Hz. If you use BH window then provide it in the freq_wts.


############################################################################## 

bandwidth = abs(np.diff(freq_array[0])[0]*Nfreqs)
cosmo = default_cosmology.get()
los_dist = cosmo.comoving_distance(redshift).to('Mpc')  # in MPc/h 
dlos_dist = (FCNST.c/1e3) * bandwidth * (1+redshift)**2 / rest_freq_HI / cosmo.H0.value / cosmo.efunc(redshift)  # in Mpc

wl = FCNST.c /f0.value  #This is wavelength corresponding to central frequency ## 

Jy2K = wl**2 * Jy / (2*FCNST.k)  

jacobian2 = los_dist**2 * dlos_dist/ bandwidth.value
jacobian_1 = 1/omega_bw

normalization_factor = jacobian_1*jacobian2*Jy2K**2  # If we multiply by this it will give me the power spectrum in K^2 MPc^3 unit. see Thayagarajan 2015,  Eqn 4. 


################### Folded power spectrum ##### 

## Folding the delay axis here ### 

baseline_length = np.zeros(shape = len(u_cell))
k_perp_grid = np.zeros(shape = len(u_cell))

for ii in range(len(u_cell)):
    baseline_length[ii] = baseline_m_grid[ii].value
    k_perp_grid[ii] = k_perpendicular_grid[ii].value 



def fold_delay_power(power_array,Ndelays):
    folded_array = np.zeros_like(power_array)
    left = power_array[1:Ndelays//2][::-1] 
    right = power_array[Ndelays//2+1:]
    folded_array[Ndelays//2+1:] = np.mean([left, right], axis=0)
    folded_array[:Ndelays//2] = 0.0 
    return folded_array 


## these dictionaries will contain the power spectrum after multiplied by the normalization factor ## 
power_array_CLEAN_fold = {}
noise_array_CLEAN_fold = {}



for group in range(len(u_cell)):
    

    power_array_CLEAN_fold[group] = fold_delay_power(np.abs(power_clean_off_diag[group]*normalization_factor), int(N_window))
    noise_array_CLEAN_fold[group] = fold_delay_power(np.abs(power_clean_diag[group]*normalization_factor), int(N_window))


#### Here I am subtracting the off-diagonal power from the diagonal power of a correlation matrix for a uv-cell. This will give the noise power. Upon dividing by the no. of modes averaged together, it will give power spectrum uncertainty- uncertainty= noise_power/sqrt(modes) (Tegmark 1997)


 
noise_power = {}

for kk in range(len(u_cell)):

    noise_power[kk] = abs(noise_array_CLEAN_fold[kk][noise_array_CLEAN_fold[kk] > 0.0] - power_array_CLEAN_fold[kk][power_array_CLEAN_fold[kk] > 0.0])/np.sqrt(uv_weight[kk])




## Now store the normalized folded power spectrum in pkl files. 

output_1 = open('power_clean_normalized.pkl', 'wb')
pkl.dump(power_array_CLEAN_fold, output_1)
output_1.close()


output_3 = open('power_noise_normalized.pkl', 'wb')
pkl.dump(noise_power, output_3)
output_3.close()

## Later load these and bins these arrays in cylindrically and spherically to  estimate the 2D and 3D PS ### 





