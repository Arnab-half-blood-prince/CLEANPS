# CLEANPS
A simple power spectrum estimation pipeline from flagged visibility data set. The estimation is based on Parsons et al. 2012,2014. 
We use 1D CLEAN (Hogbom) algorithm to mitigate the missing channel issues. 
The single_baseline_data folder contains the data,  py files and a readme with all information.
The multiple_baseline_data folder contains the py files and readme. Note that, I have tested this on SKA-1 low simulation. The data file is huge, so unable to 
provide here.

However, one can simulate any data set for any telescope and try this out. The multiple_baseline_data cotains the py file which will CLEAN the baselines in delay domain parallely. So, dependeing upon the numbers of cores in the machine, it will be faster. 

The all elevant dependencies are mentined in the beginning of the functions.py file inside these folders. 

It is mainly depends on numpy,scipy,astropy, healpy, pyuvdata, progressbar, itertools, skimage, uvtools. 
