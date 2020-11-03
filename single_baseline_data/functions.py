## Import these for plotting ## 

###### For plotting ######

import numpy as np
from matplotlib.pyplot import *
import matplotlib.pyplot as pl 
import matplotlib 
###################################
pl.rcParams['figure.figsize'] = 8, 7
pl.rcParams['ytick.minor.visible'] =True
pl.rcParams['xtick.minor.visible'] = True
pl.rcParams['xtick.top'] = True
pl.rcParams['ytick.right'] = True
pl.rcParams['font.size'] = '20'
pl.rcParams['legend.fontsize'] = '15'
pl.rcParams['legend.borderaxespad'] = '1.9'
#pl.rcParams['legend.numpoints'] = '1'

pl.rcParams['figure.titlesize'] = 'medium'
pl.rcParams['figure.titlesize'] = 'medium'
pl.rcParams['xtick.major.size'] = '10'
pl.rcParams['xtick.minor.size'] = '6'
pl.rcParams['xtick.major.width'] = '2'
pl.rcParams['xtick.minor.width'] = '1'
pl.rcParams['ytick.major.size'] = '10'
pl.rcParams['ytick.minor.size'] = '6'
pl.rcParams['ytick.major.width'] = '2'
pl.rcParams['ytick.minor.width'] = '1'
pl.rcParams['xtick.direction'] = 'in'
pl.rcParams['ytick.direction'] = 'in'
pl.rcParams['axes.labelpad'] = '10.0'
pl.rcParams['lines.dashed_pattern']=3.0, 1.4
#pl.rcParams['axes.formatter.limits']=-10,10
pl.rcParams['lines.dotted_pattern']= 1.0, 0.7

pl.rcParams['xtick.labelsize'] = '16'
pl.rcParams['ytick.labelsize'] = '16'
pl.rcParams['axes.labelsize'] = '16'
pl.rcParams['axes.labelsize'] = '16'
pl.rcParams['axes.labelweight'] = 'bold'

pl.rcParams['xtick.major.pad']='10'
pl.rcParams['xtick.minor.pad']='10'
#pl.rcParams['hatch.color'] = 'black'
pl.rc('axes', linewidth=2)
###########################################

import multiprocessing as MP
import itertools as IT
import progressbar as PGB

#from astroutils import writer_module as WM
import healpy as hp
#from astroutils import mathops as OPS
import scipy.constants as FCNST 
#from astroutils import constants as CNST
from astropy import constants as const

import numpy.ma as MA
import scipy as SP
from scipy import interpolate
from skimage import img_as_float
import skimage.morphology as morphology
from skimage.filters import median
from skimage.filters.rank import mean
from skimage.restoration import unwrap_phase
import astropy.convolution as CONV


import copy
import six
import warnings
from pyuvdata import *

from scipy.signal import windows
import scipy.integrate as integrate

import random
import os
import sys
from astropy import constants as const
from astropy import units
from astropy.units import Quantity
from astropy.cosmology import Planck15, default_cosmology
import uvtools
from scipy.optimize import curve_fit

# the emission frequency of 21m photons in the Hydrogen's rest frame
f21 = 1420405751.7667 * units.Hz

# Using WMAP 9-year cosmology as the default
# move in the the little-h unit frame by setting H0=100
default_cosmology.set(Planck15)

############################################################################## 

# MKS units

Jy = 1.0e-26 # W m^-2 Hz^-1

# Ratio of effective bandwidths of Rectangular and Blackman-Nuttall windows

rect_bnw_ratio = 8.0248887/2.9086642

# Rest frequency of neutral hydrogen line

rest_freq_HI = 1420405751.77 # in Hz

# Rest frequencies of CO transition lines

rest_freq_CO_1_0 = 115271201800.0 # in Hz
rest_freq_CO_2_0 = 230538000000.0 # in Hz

# Sidereal day

sday = 0.99726958 # (in number of mean of solar days)



##############################################################################################################
def healpix_interp_along_axis(indata, theta_phi=None, inloc_axis=None,
                              outloc_axis=None, axis=-1, kind='linear',
                              bounds_error=True, fill_value=np.nan,
                              assume_sorted=False, nest=False):

    """
    -----------------------------------------------------------------------------
    Interpolate healpix data to specified angular locations (HEALPIX 
    interpolation) and along one other specified axis (usually frequency axis, 
    for instance) via SciPy interpolation. Wraps HEALPIX and SciPy interpolations
    into one routine.

    Inputs:

    indata      [numpy array] input data to be interpolated. Must be of shape 
                (nhpy x nax1 x nax2 x ...). Currently works only for 
                (nhpy x nax1). nhpy is a HEALPIX compatible npix

    theta_phi   [numpy array] spherical angle locations (in radians) at which
                the healpix data is to be interpolated to at each of the other 
                given axes. It must be of size nang x 2 where nang is the number 
                of spherical angle locations, 2 denotes theta and phi. If set to
                None (default), no healpix interpolation is performed

    inloc_axis  [numpy array] locations along the axis specified in axis (to be 
                interpolated with SciPy) in which indata is specified. It 
                should be of size nax1, nax2, ... or naxm. Currently it works 
                only if set to nax1

    outloc_axis [numpy array] locations along the axis specified in axis to be 
                interpolated to with SciPy. The axis over which this 
                interpolation is to be done is specified in axis. It must be of
                size nout. If this is set exactly equal to inloc_axis, no 
                interpolation along this axis is performed

    axis        [integer] axis along which SciPy interpolation is to be done. 
                If set to -1 (default), the interpolation happens over the last
                axis. Since the first axis of indata is reserved for the healpix
                pixels, axis must be set to 1 or above (upto indata.ndim-1).

    kind        [str or int] Specifies the kind of interpolation as a 
                string ('linear', 'nearest', 'zero', 'slinear', 'quadratic', 
                'cubic' where 'slinear', 'quadratic' and 'cubic' refer to a 
                spline interpolation of first, second or third order) or as an 
                integer specifying the order of the spline interpolator to use. 
                Default is 'linear'.

    bounds_error 
                [bool, optional] If True, a ValueError is raised any time 
                interpolation is attempted on a value outside of the range of x 
                (where extrapolation is necessary). If False, out of bounds 
                values are assigned fill_value. By default, an error is raised.

    fill_value  [float] If provided, then this value will be used to fill in 
                for requested points outside of the data range. If not provided, 
                then the default is NaN.

    assume_sorted 
                [bool] If False, values of inloc_axis can be in any order and 
                they are sorted first. If True, inloc_axis has to be an array 
                of monotonically increasing values.
    
    nest        [bool] if True, the is assumed to be in NESTED ordering.

    Outputs:

    HEALPIX interpolated and SciPy interpolated output. Will be of size
    nang x ... x nout x ... x naxm. Currently returns an array of shape 
    nang x nout
    -----------------------------------------------------------------------------
    """

    try:
        indata
    except NameError:
        raise NameError('input data not specified')

    if not isinstance(indata, np.ndarray):
        raise TypeError('input data must be a numpy array')

    if theta_phi is not None:
        if not isinstance(theta_phi, np.ndarray):
            raise TypeError('output locations must be a numpy array')

        if theta_phi.ndim != 2:
            raise ValueError('Output locations must be a 2D array')

    if axis == -1:
        axis = indata.ndim - 1

    if (axis < 1) or (axis >= indata.ndim):
        raise ValueError('input axis out of range')

    if theta_phi is not None:
        intermediate_data_shape = list(indata.shape)
        intermediate_data_shape[0] = theta_phi.shape[0]
        intermediate_data_shape = tuple(intermediate_data_shape)
        
        intermediate_data = np.zeros(intermediate_data_shape, dtype=np.float64)
        for ax in range(1,indata.ndim):
            for i in range(indata.shape[ax]):
                intermediate_data[:,i] = hp.get_interp_val(indata[:,i], theta_phi[:,0], theta_phi[:,1], nest=nest)
    else:
        intermediate_data = np.copy(indata)

    if outloc_axis is not None:
        if inloc_axis is not None:
            outloc_axis = outloc_axis.flatten()
            inloc_axis = inloc_axis.flatten()
            eps = 1e-8
            if (outloc_axis.size == inloc_axis.size) and (np.abs(inloc_axis-outloc_axis).max() <= eps):
                outdata = intermediate_data
            else:
                if kind == 'fft':
                    df_inp = np.mean(np.diff(inloc_axis))
                    df_out = np.mean(np.diff(outloc_axis))
                    ntau = df_inp / df_out * inloc_axis.size
                    ntau = np.round(ntau).astype(int)
                    tau_inp = spectral_axis(inloc_axis.size, delx=df_inp, shift=True)
                    fftinp = np.fft.fft(intermediate_data, axis=axis)
                    fftinp_shifted = np.fft.fftshift(fftinp, axes=axis)
                    if fftinp.size % 2 == 0:
                        fftinp_shifted[:,0] = 0.0 # Blank out the N/2 element (0 element when FFT-shifted) for conjugate symmetry
                    npad = ntau - inloc_axis.size
                    if npad % 2 == 0:
                        npad_before = npad/2
                        npad_after = npad/2
                    else:
                        npad_before = npad/2 + 1
                        npad_after = npad/2

                    fftinp_shifted_padded = np.pad(fftinp_shifted, [(0,0), (npad_before, npad_after)], mode='constant')
                    fftinp_padded = np.fft.ifftshift(fftinp_shifted_padded, axes=axis)
                    ifftout = np.fft.ifft(fftinp_padded, axis=axis) * (1.0 * ntau / inloc_axis.size)
                    eps_imag = 1e-10
                    if np.any(np.abs(ifftout.imag) > eps_imag):
                        raise ValueError('Significant imaginary component has been introduced unintentionally during the FFT based interpolation. Debug the code.')
                    else:
                        ifftout = ifftout.real
                    fout = spectral_axis(ntau, delx=tau_inp[1]-tau_inp[0], shift=True)
                    fout -= fout.min()
                    fout += inloc_axis.min() 
                    ind_outloc, ind_fout, dfreq = LKP.find_1NN(fout.reshape(-1,1), outloc_axis.reshape(-1,1), distance_ULIM=0.5*(fout[1]-fout[0]), remove_oob=True)
                    outdata = ifftout[:,ind_fout]
                    
                    # npad = 2 * (outloc_axis.size - inloc_axis.size)
                    # dt_inp = DSP.spectral_axis(2*inloc_axis.size, delx=inloc_axis[1]-inloc_axis[0], shift=True)
                    # dt_out = DSP.spectral_axis(2*outloc_axis.size, delx=outloc_axis[1]-outloc_axis[0], shift=True)
                    # fftinp = np.fft.fft(np.pad(intermediate_data, [(0,0), (0,inloc_axis.size)], mode='constant'), axis=axis) * (1.0 * outloc_axis.size / inloc_axis.size)
                    # fftinp = np.fft.fftshift(fftinp, axes=axis)
                    # fftinp[0,0] = 0.0  # Blank out the N/2 element for conjugate symmetry
                    # fftout = np.pad(fftinp, [(0,0), (npad/2, npad/2)], mode='constant')
                    # fftout = np.fft.ifftshift(fftout, axes=axis)
                    # outdata = np.fft.ifft(fftout, axis=axis)
                    # outdata = outdata[0,:outloc_axis.size]
                else:
                    interp_func = interpolate.interp1d(inloc_axis, intermediate_data, axis=axis, kind=kind, bounds_error=bounds_error, fill_value=fill_value, assume_sorted=assume_sorted)
                    outdata = interp_func(outloc_axis)
        else:
            raise ValueError('input inloc_axis not specified')
    else:
        outdata = intermediate_data

    return outdata

##############################################################################################################


@units.quantity_input(freq='frequency')
def calc_z(freq):
    """Calculate the redshift from a given frequency or frequncies.

    Parameters
    ----------
    freq : Astropy Quantity Object units equivalent to frequency
        The frequency to calculate the redshift of 21cm emission

    Returns
    -------
    redshift : float
        The redshift consistent with 21cm observations of the input frequency.

    """
    return (f21 / freq).si.value - 1


def calc_freq(redshift):
    """Calculate the frequency or frequencies of a given 21cm redshift.

    Parameters
    ----------
    redshift : float
        The redshift of the expected 21cm emission

    Returns
    -------
    freq : Astropy Quantity Object units equivalent to frequency
        Frequency of the emission in the rest frame of emission

    """
    return f21 / (1 + redshift)
    
def u2kperp(u, z, cosmo=None):
    """Convert baseline length u to k_perpendicular.

    Parameters
    ----------
    u : float
        The baseline separation of two interferometric antennas in units of wavelength
    z : float
        The redshift of the expected 21cm emission.
    cosmo : Astropy Cosmology Object
        The assumed cosmology of the universe.
        Defaults to WMAP9 year in "little h" units

    Returns
    -------
    kperp : Astropy Quantity units equivalent to wavenumber
        The spatial fluctuation scale perpendicular to the line of sight probed by the baseline length u.

    """
    if cosmo is None:
        cosmo = default_cosmology.get()
    return 2 * np.pi * u / cosmo.comoving_transverse_distance(z)


@units.quantity_input(kperp='wavenumber')
def kperp2u(kperp, z, cosmo=None):
    """Convert comsological k_perpendicular to baseline length u.

    Parameters
    ----------
    kperp : Astropy Quantity units equivalent to wavenumber
        The spatial fluctuation scale perpendicular to the line of sight.
    z : float
        The redshift of the expected 21cm emission.
    cosmo : Astropy Cosmology Object
        The assumed cosmology of the universe.
        Defaults to WMAP9 year in "little h" units

    Returns
    -------
    u : float
        The baseline separation of two interferometric antennas in units of
        wavelength which probes the spatial scale given by kperp

    """
    if cosmo is None:
        cosmo = default_cosmology.get()
    return kperp * cosmo.comoving_transverse_distance(z) / (2 * np.pi)


@units.quantity_input(eta='time')
def eta2kparr(eta, z, cosmo=None):
    """Conver delay eta to k_parallel (comoving 1./Mpc along line of sight).

    Parameters
    ----------
    eta : Astropy Quantity object with units equivalent to time.
        The inteferometric delay observed in units compatible with time.
    z : float
        The redshift of the expected 21cm emission.
    cosmo : Astropy Cosmology Object
        The assumed cosmology of the universe.
        Defaults to WMAP9 year in "little h" units

    Returns
    -------
    kparr : Astropy Quantity units equivalent to wavenumber
        The spatial fluctuation scale parallel to the line of sight probed by the input delay (eta).

    """
    if cosmo is None:
        cosmo = default_cosmology.get()
    return (eta * (2 * np.pi * cosmo.H0 * f21 * cosmo.efunc(z))
            / (const.c * (1 + z)**2)).to('1/Mpc')


@units.quantity_input(kparr='wavenumber')
def kparr2eta(kparr, z, cosmo=None):
    """Convert k_parallel (comoving 1/Mpc along line of sight) to delay eta.

    Parameters
    ----------
    kparr : Astropy Quantity units equivalent to wavenumber
        The spatial fluctuation scale parallel to the line of sight
    z : float
        The redshift of the expected 21cm emission.
    cosmo : Astropy Cosmology Object
        The assumed cosmology of the universe.
        Defaults to WMAP9 year in "little h" units

    Returns
    -------
    eta : Astropy Quantity units equivalent to time
        The inteferometric delay which probes the spatial scale given by kparr.

    """
    if cosmo is None:
        cosmo = default_cosmology.get()
    return (kparr * const.c * (1 + z)**2
            / (2 * np.pi * cosmo.H0 * f21 * cosmo.efunc(z))).to('s')


def X2Y(z, cosmo=None):
    """Convert units from interferometric delay power to comsological power.

    Converts power spectrum from interferometric units of eta, u to
    cosmological k_par, k_perp.

    Parameters
    ----------
    z : float
        redshift for cosmological conversion
    cosmo : Astropy Cosmology Object
        The assumed cosmology of the universe.
        Defaults to WMAP9 year in "little h" units

    Returns
    -------
    X2Y : Astropy Quantity units of Mpc^3*s/sr
        The power spectrum unit conversion factor

    """
    if cosmo is None:
        cosmo = default_cosmology.get()
    X2Y = const.c * (1 + z)**2 * cosmo.comoving_transverse_distance(z)**2
    X2Y /= cosmo.H0 * f21 * cosmo.efunc(z) * units.sr
    return X2Y.to('Mpc^3*s/sr')

@units.quantity_input(freqs='frequency')
def jy_to_mk(freqs):
    """Calculate the Jy/sr to mK conversion lambda^2/(2 * K_boltzman).

    Parameters
    ----------
    freqs : Astropy Quantity with units equivalent to frequency
        frequencies where the conversion should be calculated.

    Returns
    -------
    Astropy Quantity
        The conversion factor from Jy to mK * sr at the given frequencies.

    """
    jy2t = units.sr * const.c.to('m/s')**2 / (2 * freqs.to('1/s')**2
                                              * const.k_B)
    return jy2t.to('mK*sr/Jy')

def bootstrap_array(array, nboot=1000, axis=0):
    """Bootstrap resample the input array along the given axis.

    Randomly sample, with replacement along the given axis `nboot` times.
    Output array will always have N+1 dimensions with the extra axis immediately following the bootstrapped axis.

    Parameters
    ----------
    array : numpy array
        N-dimensional array to bootstrap resample.
    nboot : int
        Number of resamples to draw.
    axis : int
        Axis along which resampling is performed

    Returns
    -------
    array : numpy array
        The resampled array, if input is N-D output is N+1-D, extra dimension is added imediately suceeding the sampled axis.

    Raises
    ------
    ValueError
        If `axis` parameter is greater than the dimensions of the input array.

    """
    if axis >= len(np.shape(array)):
        raise ValueError("Specified axis must be shorter than the length "
                         "of input array.\n"
                         "axis value was {0} but array has {1} dimensions"
                         .format(axis, len(np.shape(array))))

    sample_inds = np.random.choice(array.shape[axis],
                                   size=(array.shape[axis], nboot),
                                   replace=True)
    return np.take(array, sample_inds, axis=axis)


def get_data_array(uv, reds, squeeze=True):
    """Remove data from pyuvdata object and store in numpy array.

    Uses UVData.get_data function to create a matrix of data of shape (Npols, Nbls, Ntimes, Nfreqs).
    Only valid to call on a set of redundant baselines with the same number of times.

    Parameters
    ----------
    uv : UVdata object, or subclass
        Data object which can support uv.get_data(ant_1, ant_2, pol)
    reds: list of ints
        list of all redundant baselines of interest as baseline numbers.
    squeeze : bool
        set true to squeeze the polarization dimension.
        This has no effect for data with Npols > 1.

    Returns
    -------
    data_array :complex arrary
        (Nbls , Ntimes, Nfreqs) numpy array or (Npols, Nbls, Ntimes, Nfreqs) if squeeze == False

    """
    data_shape = (uv.Npols, len(reds), uv.Ntimes, uv.Nfreqs)
    data_array = np.zeros(data_shape, dtype=np.complex)

    for count, baseline in enumerate(reds):
        tmp_data = uv.get_data(baseline, squeeze='none')
        # Keep the polarization dimenions and squeeze out spw
        tmp_data = np.squeeze(tmp_data, axis=1)
        # Reorder to: Npols, Ntimes, Nfreqs

        data_array[:, count] = np.transpose(tmp_data, [2, 0, 1])

    if squeeze:
        if data_array.shape[0] == 1:
            data_array = np.squeeze(data_array, axis=0)

    return data_array


def get_nsample_array(uv, reds, squeeze=True):
    """Remove nsamples from pyuvdata object and store in numpy array.

    Uses UVData.get_nsamples function to create a matrix of data of shape (Npols, Nbls, Ntimes, Nfreqs).
    Only valid to call on a set of redundant baselines with the same number of times.

    Parameters
    ----------
    uv : UVdata object, or subclass
        Data object which can support uv.get_data(ant_1, ant_2, pol)
    reds: list of ints
        list of all redundant baselines of interest as baseline numbers.
    squeeze : bool
        set true to squeeze the polarization dimension.
        This has no effect for data with Npols > 1.

    Returns
    -------
    nsample_array : float array
        (Nbls, Ntimes, Nfreqs) numpy array or (Npols, Nbls, Ntimes, Nfreqs) if squeeze == False

    """
    nsample_shape = (uv.Npols, len(reds), uv.Ntimes, uv.Nfreqs)
    nsample_array = np.zeros(nsample_shape, dtype=np.float)

    for count, baseline in enumerate(reds):
        tmp_data = uv.get_nsamples(baseline, squeeze='none')
        # Keep the polarization dimenions and squeeze out spw
        tmp_data = np.squeeze(tmp_data, axis=1)
        # Reorder to: Npols, Ntimes, Nfreqs
        nsample_array[:, count] = np.transpose(tmp_data, [2, 0, 1])

    if squeeze:
        if nsample_array.shape[0] == 1:
            nsample_array = np.squeeze(nsample_array, axis=0)

    return nsample_array


def get_flag_array(uv, reds, squeeze=True):
    """Remove nsamples from pyuvdata object and store in numpy array.

    Uses UVData.get_flags function to create a matrix of data of shape (Npols, Nbls, Ntimes, Nfreqs).
    Only valid to call on a set of redundant baselines with the same number of times.

    Parameters
    ----------
    uv : UVdata object, or subclass
        Data object which can support uv.get_data(ant_1, ant_2, pol)
    reds: list of ints
        list of all redundant baselines of interest as baseline numbers.
    squeeze : bool
        set true to squeeze the polarization dimension.
        This has no effect for data with Npols > 1.

    Returns
    -------
    flag_array : bool array
        (Nbls, Ntimes, Nfreqs) numpy array  (Npols, Nbls, Ntimes, Nfreqs) if squeeze == False

    """
    flag_shape = (uv.Npols, len(reds), uv.Ntimes, uv.Nfreqs)
    flag_array = np.zeros(flag_shape, dtype=np.bool)
    reds = np.array(reds)

    for count, baseline in enumerate(reds):
        tmp_data = uv.get_flags(baseline, squeeze='none')
        # Keep the polarization dimenions and squeeze out spw
        tmp_data = np.squeeze(tmp_data, axis=1)
        # Reorder to: Npols, Ntimes, Nfreqs
        flag_array[:, count] = np.transpose(tmp_data, [2, 0, 1])

    if squeeze:
        if flag_array.shape[0] == 1:
            flag_array = np.squeeze(flag_array, axis=0)

    return flag_array


def get_integration_time(uv, reds, squeeze=True):
    """Extract the integration_time array from pyuvdata objectself.

    Extracts the integration time array from a UVdata object to create a matrix of shape (Npols, Nbls, Ntimes, Nfreqs).
    Only valid to call on a set of redundant baselines with the same number of times.

    Parameters
    ----------
    uv : UVdata object, or subclass
        Data object which can support uv.get_data(ant_1, ant_2, pol)
    reds: list of ints
        list of all redundant baselines of interest as baseline numbers.
    squeeze : bool
        set true to squeeze the polarization dimension.
        This has no effect for data with Npols > 1.

    Returns
    -------
    integration_time : float array
        (Nbls, Ntimes) numpy array of integration times.

    """
    shape = (len(reds), uv.Ntimes)
    integration_time = np.zeros(shape, dtype=np.float)
    reds = np.array(reds)

    for count, baseline in enumerate(reds):
        blt_inds, conj_inds, pol_inds = uv._key2inds(baseline)
        # The integration doesn't care about conjugation, just need all the
        # times associated with this baseline
        inds = np.concatenate([blt_inds, conj_inds])
        inds.sort()
        integration_time[count] = uv.integration_time[inds]

    return integration_time

def beam3Dvol(beam, freqs, freq_wts=None, hemisphere=True):

    """
    ----------------------------------------------------------------------------
    Compute 3D volume relevant for power spectrum given an antenna power 
    pattern. It is estimated by summing square of the beam in angular and 
    frequency coordinates and in units of "Sr Hz".

    Inputs:

    beam        [numpy array] Antenna power pattern with peak normalized to 
                unity. It can be of shape (npix x nchan) or (npix x 1) or 
                (npix,). npix must be a HEALPix compatible value. nchan is the
                number of frequency channels, same as the size of input freqs.
                If it is of shape (npix x 1) or (npix,), the beam will be 
                assumed to be identical for all frequency channels.

    freqs       [list or numpy array] Frequency channels (in Hz) of size nchan

    freq_wts    [numpy array] Frequency weights to be applied to the
                beam. Must be of shape (nchan,) or (nwin, nchan)

    Keyword Inputs:

    hemisphere  [boolean] If set to True (default), the 3D volume will be 
                estimated using the upper hemisphere. If False, the full sphere
                is used.

    Output:

    The product Omega x bandwdith (in Sr Hz) computed using the integral of 
    squared power pattern. It is of shape (nwin,)
    ----------------------------------------------------------------------------
    """

    try:
        beam, freqs
    except NameError:
        raise NameError('Both inputs beam and freqs must be specified')

    if not isinstance(beam, np.ndarray):
        raise TypeError('Input beam must be a numpy array')

    if not isinstance(freqs, (list, np.ndarray)):
        raise TypeError('Input freqs must be a list or numpy array')
    freqs = np.asarray(freqs).astype(np.float).reshape(-1)
    if freqs.size < 2:
        raise ValueError('Input freqs does not have enough elements to determine frequency resolution')

    if beam.ndim > 2:
        raise ValueError('Invalid dimensions for beam')
    elif beam.ndim == 2:
        if beam.shape[1] != 1:
            if beam.shape[1] != freqs.size:
                raise ValueError('Dimensions of beam do not match the number of frequency channels')
    elif beam.ndim == 1:
        beam = beam.reshape(-1,1)
    else:
        raise ValueError('Invalid dimensions for beam')

    if freq_wts is not None:
        if not isinstance(freq_wts, np.ndarray):
            raise TypeError('Input freq_wts must be a numpy array')
        if freq_wts.ndim == 1:
            freq_wts = freq_wts.reshape(1,-1)
        elif freq_wts.ndim > 2:
            raise ValueError('Input freq_wts must be of shape nwin x nchan')

        freq_wts = np.asarray(freq_wts).astype(np.float).reshape(-1,freqs.size)
        if freq_wts.shape[1] != freqs.size:
            raise ValueError('Input freq_wts does not have shape compatible with freqs')
    else:
        freq_wts = np.ones(freqs.size, dtype=np.float).reshape(1,-1)

    eps = 1e-10
    if beam.max() > 1.0+eps:
        raise ValueError('Input beam maximum exceeds unity. Input beam should be normalized to peak of unity')

    nside = hp.npix2nside(beam.shape[0])
    domega = hp.nside2pixarea(nside, degrees=False)
    df = freqs[1] - freqs[0]
    bw = df * freqs.size
    weighted_beam = beam[:,np.newaxis,:] * freq_wts[np.newaxis,:,:]

    theta, phi = hp.pix2ang(nside, np.arange(beam.shape[0]))
    if hemisphere:
        ind, = np.where(theta <= np.pi/2)  # Select upper hemisphere
    else:
        ind = np.arange(beam.shape[0])

    omega_bw = domega * df * np.nansum(weighted_beam[ind,:,:]**2, axis=(0,2))
    if np.any(omega_bw > 4*np.pi*bw):
        raise ValueError('3D volume estimated from beam exceeds the upper limit. Check normalization of the input beam')

    return omega_bw

## A function for waterfall plot ## 

def waterfall(
    vis, freq=None, lsts=None,
    vmax=None, vrange=None, title=None,
):
    """
    A wrapper around the uvtools' waterfall function providing some
    extra labelling and plot adjustment.
    """
    freq = freq/1e6
    lsts = lsts
    fig, ax = plt.subplots(
        2,1, sharex=True, sharey=True, figsize=(12,10)
    )

    if title is not None:
        ax[0].set_title(title, fontsize=12)
    plt.sca(ax[0])
    uvtools.plot.waterfall(
        vis, mode='abs', mx=vmax, drng=vrange,
        extent=(freq.min(), freq.max(), lsts.min(), lsts.max())
    )
    plt.colorbar(label=r'(Vis/Jy)')
    plt.ylabel("LST", fontsize=12)

    plt.sca(ax[1])
    uvtools.plot.waterfall(
        vis,
        mode='phs',
        extent=(freq.min(), freq.max(), lsts.min(), lsts.max())
    )
    plt.colorbar(label='Phase [rad]')
    plt.xlabel("Frequency [MHz]", fontsize=12)
    plt.ylabel("LST", fontsize=12)
    plt.tight_layout()
    plt.show()

def delay_waterfall(
    vis, delay=None, lsts=None,
    vmax=None, vrange=None, title=None,
):
    """
    A wrapper around the uvtools' waterfall function providing some
    extra labelling and plot adjustment.
    """
    fig, ax = plt.subplots(
        2,1, sharex=True, sharey=True, figsize=(12,10)
    )

    if title is not None:
        ax[0].set_title(title, fontsize=12)
    plt.sca(ax[0])
    uvtools.plot.waterfall(
        vis, mode='abs', mx=vmax, drng=vrange,
        extent=(delay.min(), delay.max(), lsts.min(), lsts.max())
    )
    plt.colorbar(label='Vis/Jy)')
    plt.ylabel("LST", fontsize=12)

    plt.sca(ax[1])
    uvtools.plot.waterfall(
        vis,
        mode='phs',
        extent=(delay.min(), delay.max(), lsts.min(), lsts.max())
    )
    plt.colorbar(label='Phase [rad]')
    plt.xlabel("Delay [ns]", fontsize=12)
    plt.ylabel("LST", fontsize=12)
    plt.tight_layout()
    plt.show()  

def delay_transform(data_array,flag_array = None,freq_array = None, delay_array = None, inverse = False, taper=windows.blackmanharris, shift = False):
        """Perform a delay transform on the stored data array.

        If data is set to frequency domain, fourier transforms to delay space.
        If data is set to delay domain, inverse fourier transform to frequency space.
        """      
        
        
        if freq_array is not None:
           delta_x = abs(np.diff(freq_array[0])[0])
        else:
            delta_x = abs(np.diff(delay_array[0])[0])
        float_flags = np.logical_not(flag_array).astype(float)
        data_array = data_array*float_flags
        axis = -1
        if isinstance(data_array, units.Quantity):
            unit = data_array.unit
        else:
             unit = 1.

        if not isinstance(delta_x, units.Quantity):
           raise ValueError('delta_x must be an astropy Quantity object. '
                         'value was : {df}'.format(df=delta_x))

        n_axis = data_array.shape[axis]
        data_shape = np.ones_like(data_array.shape)
        data_shape[axis] = n_axis
        # win = taper(n_axis).reshape(1, n_axis)
        win = np.broadcast_to(taper(n_axis), data_shape)

        # Fourier Transforms should have a delta_x term multiplied
        # This is the proper normalization of the FT but is not
        # accounted for in an fft.
        if not inverse:
           if not shift:

              fourier_array = np.fft.fft(data_array * win, axis=axis)
              fourier_array = fourier_array * delta_x.si * unit
           else:
               fourier_array = np.fft.fft(data_array * win, axis=axis) 
               
               fourier_array = np.fft.fftshift(fourier_array, axes=axis)
               fourier_array = fourier_array * delta_x.si * unit
                 
        else:
            if not shift:
               fourier_array = np.fft.ifft(data_array, axis=axis)
               fourier_array = fourier_array / win * delta_x.si * unit 
            else:
                fourier_array = np.fft.ifft(data_array, axis=axis)
                
                fourier_array = np.fft.ifftshift(fourier_array, axes=axis)
                fourier_array = fourier_array / win * delta_x.si * unit
                

        return fourier_array      

def delay_transform_without_window(data_array,freq_array = None, delay_array = None, flag_array = None,inverse = False,shift = False):
        """Perform a delay transform without any tapering window on the stored data array.

        If data is set to frequency domain, fourier transforms to delay space.
        If data is set to delay domain, inverse fourier transform to frequency space.
        """      
        
        if freq_array is not None:
           delta_x = abs(np.diff(freq_array[0])[0])
        else:
            delta_x = abs(np.diff(delay_array[0])[0])
        
        float_flags = np.logical_not(flag_array).astype(float)
        data_array = data_array*float_flags
        axis = -1
        if isinstance(data_array, units.Quantity):
            unit = data_array.unit
        else:
             unit = 1.

        if not isinstance(delta_x, units.Quantity):
           raise ValueError('delta_x must be an astropy Quantity object. '
                         'value was : {df}'.format(df=delta_x))

        n_axis = data_array.shape[axis]
        data_shape = np.ones_like(data_array.shape)
        data_shape[axis] = n_axis

        # Fourier Transforms should have a delta_x term multiplied
        # This is the proper normalization of the FT but is not
        # accounted for in an fft.
        if not inverse:
           if not shift:

              fourier_array = np.fft.fft(data_array, axis=axis)
              fourier_array = fourier_array * delta_x.si * unit
           else:
               fourier_array = np.fft.fft(data_array, axis=axis)
                  
               fourier_array = np.fft.fftshift(fourier_array, axes=axis)
               fourier_array = fourier_array * delta_x.si * unit                     
                
        else:
            if not shift:
               fourier_array = np.fft.ifft(data_array, axis=axis)
               fourier_array = fourier_array * delta_x.si * unit 
            else:
                fourier_array = np.fft.ifft(data_array, axis=axis)
                
                fourier_array = np.fft.ifftshift(fourier_array, axes=axis)
                fourier_array = fourier_array * delta_x.si * unit                  
                

        return fourier_array 
            

def remove_auto_correlations(data_array, axes=(0, 1)):
    """Remove the auto-corrlation term from input array.

    Takes an N x N array, removes the diagonal components, and returns a flattened N(N-1) dimenion in place of the array.
    If uses on a M dimensional array, returns an M-1 array.

    Parameters
    ----------
    data_array : array
        Array shaped like (Nbls, Nbls, Ntimes, Nfreqs). Removes same baseline diagonal along the specifed axes.
    axes : tuple of int, length 2
        axes over which the diagonal will be removed.

    Returns
    -------
    data_out : array with the same type as `data_array`.
        (Nbls * (Nbls-1), Ntimes, Nfreqs) array.
        if input has pols: (Npols, Nbls * (Nbls -1), Ntimes, Nfreqs)

    Raises
    ------
    ValueError
        If axes is not a length 2 tuple.
        If axes are not adjecent (e.g. axes=(2,7)).
        If axes do not have the same shape.

    """
    if not np.shape(axes)[0] == 2:
        raise ValueError("Shape must be a length 2 tuple/array/list of "
                         "axis indices.")
    if axes[0] != axes[1] - 1:
        raise ValueError("Axes over which diagonal components are to be "
                         "remove must be adjacent.")
    if data_array.shape[axes[0]] != data_array.shape[axes[1]]:
        raise ValueError("The axes over which diagonal components are to be "
                         "removed must have the same shape.")
    n_inds = data_array.shape[axes[0]]
    # make a boolean index array with True off the diagonal and
    # False on the diagonal.
    indices = np.logical_not(np.diag(np.ones(n_inds, dtype=bool)))
    # move the axes so axes[0] is the 0th axis and axis 1 is the 1th
    data_array = np.moveaxis(data_array, axes[0], 0)
    data_array = np.moveaxis(data_array, axes[1], 1)
    data_out = data_array[indices]
    # put the compressed axis back in the original spot
    data_out = np.moveaxis(data_out, 0, axes[0])
    return data_out

def diagonal_term(data_array, axes=(0, 1)):
    """Remove the off-diagonal term from input array nad gives the diagonal only.

    Takes an N x N array, removes the diagonal components, and returns a flattened N dimenion in place of the array.
    If uses on a M dimensional array, returns an M-1 array.

    Parameters
    ----------
    data_array : array
        Array shaped like (Nbls, Nbls, Ntimes, Nfreqs). Removes same baseline off-diagonal along the specifed axes.
    axes : tuple of int, length 2
        axes over which the diagonal will be stored.

    Returns
    -------
    data_out : array with the same type as `data_array`.
        (Nbls, Ntimes, Nfreqs) array.
        if input has pols: (Nbls, Ntimes, Nfreqs)

    Raises
    ------
    ValueError
        If axes is not a length 2 tuple.
        If axes are not adjecent (e.g. axes=(2,7)).
        If axes do not have the same shape.

    """
    if not np.shape(axes)[0] == 2:
        raise ValueError("Shape must be a length 2 tuple/array/list of "
                         "axis indices.")
    if axes[0] != axes[1] - 1:
        raise ValueError("Axes over which diagonal components are to be "
                         "remove must be adjacent.")
    if data_array.shape[axes[0]] != data_array.shape[axes[1]]:
        raise ValueError("The axes over which diagonal components are to be "
                         "removed must have the same shape.")
    n_inds = data_array.shape[axes[0]]
    # make a boolean index array with True off the diagonal and
    # False on the diagonal.
    indices = np.diag(np.ones(n_inds, dtype=bool))
    # move the axes so axes[0] is the 0th axis and axis 1 is the 1th
    data_array = np.moveaxis(data_array, axes[0], 0)
    data_array = np.moveaxis(data_array, axes[1], 1)
    data_out = data_array[indices]
    # put the compressed axis back in the original spot
    data_out = np.moveaxis(data_out, 0, axes[0])
    return data_out

def cross_multiply_array(array_1, array_2=None, axis=0):
    """Cross multiply the arrays along the given axis.

    Cross multiplies along axis and computes array_1.conj() * array_2
    if axis has length M then a new axis of size M will be inserted directly succeeding the original.

    Parameters
    ----------
    array_1 : array_like
        N-dimensional array_like
    array_2 : array_like, optional
        N-dimenional array.
        Defaults to copy of array_1
    axis : int
        Axis along which to cross multiply

    Returns
    -------
    cross_array : array_like
        N+1 Dimensional array

    Raises
    ------
    ValueError
        If input arrays have different shapes.

    """
    if isinstance(array_1, list):
        array_1 = np.asarray(array_1)

    if array_2 is None:
        array_2 = copy.deepcopy(array_1)

    if isinstance(array_2, list):
        array_2 = np.asarray(array_2)

    unit_1, unit_2 = 1., 1.
    if isinstance(array_1, units.Quantity):
        unit_1 = array_1.unit
        array_1 = array_1.value

    if isinstance(array_2, units.Quantity):
        unit_2 = array_2.unit
        array_2 = array_2.value

    if array_2.shape != array_1.shape:
        raise ValueError("array_1 and array_2 must have the same shapes. "
                         "array_1 has shape {a1} but array_2 has shape {a2}"
                         .format(a1=np.shape(array_1), a2=np.shape(array_2)))

    cross_array = (np.expand_dims(array_1, axis=axis).conj()
                   * np.expand_dims(array_2, axis=axis + 1)
                   * unit_1 * unit_2)

    return cross_array

def complex1dClean_arg_splitter(args, **kwargs):
    return complex1dClean(*args, **kwargs)

def complex1dClean(inp, kernel, cbox=None, gain=0.1, maxiter=10000,
                   threshold=5e-3, threshold_type='relative', verbose=False,
                   progressbar=False, pid=None, progressbar_yloc=0):

    """
    ----------------------------------------------------------------------------
    Hogbom CLEAN algorithm applicable to 1D complex array

    Inputs:

    inp      [numpy vector] input 1D array to be cleaned. Can be complex.

    kernel   [numpy vector] 1D array that acts as the deconvolving kernel. Can 
             be complex. Must be of same size as inp

    cbox     [boolean array] 1D boolean array that acts as a mask for pixels 
             which should be cleaned. Same size as inp. Only pixels with values 
             True are to be searched for maxima in residuals for cleaning and 
             the rest are not searched for. Default=None (means all pixels are 
             to be searched for maxima while cleaning)

    gain     [scalar] gain factor to be applied while subtracting clean 
             component from residuals. This is the fraction of the maximum in 
             the residuals that will be subtracted. Must lie between 0 and 1.
             A lower value will have a smoother convergence but take a longer 
             time to converge. Default=0.1

    maxiter  [scalar] maximum number of iterations for cleaning process. Will 
             terminate if the number of iterations exceed maxiter. Default=10000

    threshold 
             [scalar] represents the cleaning depth either as a fraction of the
             maximum in the input (when thershold_type is set to 'relative') or
             the absolute value (when threshold_type is set to 'absolute') in 
             same units of input down to which inp should be cleaned. Value must 
             always be positive. When threshold_type is set to 'relative', 
             threshold must lie between 0 and 1. Default=5e-3 (found to work 
             well and converge fast) assuming threshold_type is set to 'relative'

    threshold_type
             [string] represents the type of threshold specified by value in 
             input threshold. Accepted values are 'relative' and 'absolute'. If
             set to 'relative' the threshold value is the fraction (between 0
             and 1) of maximum in input down to which it should be cleaned. If 
             set to 'asbolute' it is the actual value down to which inp should 
             be cleaned. Default='relative'

    verbose  [boolean] If set to True (default), print diagnostic and progress
             messages. If set to False, no such messages are printed.

    progressbar 
             [boolean] If set to False (default), no progress bar is displayed

    pid      [string or integer] process identifier (optional) relevant only in
             case of parallel processing and if progressbar is set to True. If
             pid is not specified, it defaults to the Pool process id

    progressbar_yloc
             [integer] row number where the progressbar is displayed on the
             terminal. Default=0

    Output:

    outdict  [dictionary] It consists of the following keys and values at
             termination:
             'termination' [dictionary] consists of information on the 
                           conditions for termination with the following keys 
                           and values:
                           'threshold' [boolean] If True, the cleaning process
                                       terminated because the threshold was 
                                       reached
                           'maxiter'   [boolean] If True, the cleaning process
                                       terminated because the number of 
                                       iterations reached maxiter
                           'inrms<outrms'
                                       [boolean] If True, the cleaning process
                                       terminated because the rms inside the 
                                       clean box is below the rms outside of it
             'iter'        [scalar] number of iterations performed before 
                           termination
             'rms'         [numpy vector] rms of the residuals as a function of
                           iteration
             'inrms'       [numpy vector] rms of the residuals inside the clean 
                           box as a function of iteration
             'outrms'      [numpy vector] rms of the residuals outside the clean 
                           box as a function of iteration
             'res'         [numpy array] uncleaned residuals at the end of the
                           cleaning process. Complex valued and same size as 
                           inp
             'cc'          [numpy array] clean components at the end of the
                           cleaning process. Complex valued and same size as 
                           inp
    ----------------------------------------------------------------------------
    """

    try:
        inp, kernel
    except NameError:
        raise NameError('Inputs inp and kernel not specified')

    if not isinstance(inp, np.ndarray):
        raise TypeError('inp must be a numpy array')
    if not isinstance(kernel, np.ndarray):
        raise TypeError('kernel must be a numpy array')

    if threshold_type not in ['relative', 'absolute']:
        raise ValueError('invalid specification for threshold_type')

    if not isinstance(threshold, (int,float)):
        raise TypeError('input threshold must be a scalar')
    else:
        threshold = float(threshold)
        if threshold <= 0.0:
            raise ValueError('input threshold must be positive')

    inp = inp.flatten()
    kernel = kernel.flatten()
    kernel /= np.abs(kernel).max()
    kmaxind = np.argmax(np.abs(kernel))  ### this gives the index of the max value of the kernel/dirty beam ##

    if inp.size != kernel.size:
        raise ValueError('inp and kernel must have same size')

    if cbox is None:
        cbox = np.ones(inp.size, dtype=np.bool)
    elif isinstance(cbox, np.ndarray):
        cbox = cbox.flatten()
        if cbox.size != inp.size:
            raise ValueError('Clean box must be of same size as input')
        cbox = np.where(cbox > 0.0, True, False)
        # cbox = cbox.astype(np.int)
    else:
        raise TypeError('cbox must be a numpy array')
    cbox = cbox.astype(np.bool)

    if threshold_type == 'relative':
        lolim = threshold
    else:
        lolim = threshold / np.abs(inp).max()

    if lolim >= 1.0:
        raise ValueError('incompatible value specified for threshold')

    # inrms = [np.std(inp[cbox])]
    inrms = [np.median(np.abs(inp[cbox] - np.median(inp[cbox])))]
    if inp.size - np.sum(cbox) <= 2:
        outrms = None
    else:
        # outrms = [np.std(inp[np.invert(cbox)])]
        outrms = [np.median(np.abs(inp[np.invert(cbox)] - np.median(inp[np.invert(cbox)])))]

    if not isinstance(gain, float):
        raise TypeError('gain must be a floating point number')
    else:
        if (gain <= 0.0) or (gain >= 1.0):
            raise TypeError('gain must lie between 0 and 1')

    if not isinstance(maxiter, int):
        raise TypeError('maxiter must be an integer')
    else:
        if maxiter <= 0:
            raise ValueError('maxiter must be positive')

    cc = np.zeros_like(inp)   ## this stores the 'CLEAN' components 
    res = np.copy(inp)        ### This stores the residual 
    cond3 = False
   # prevrms = np.std(res)
   # currentrms = [np.std(res)]
    prevrms = np.median(np.abs(res - np.median(res)))
    currentrms = [np.median(np.abs(res - np.median(res)))]
    itr = 0
    terminate = False

    if progressbar:
        if pid is None:
            pid = MP.current_process().name
        else:
            pid = '{0:0d}'.format(pid)
        progressbar_loc = (0, progressbar_yloc)
        writer = WM.Writer(progressbar_loc)
        progress = PGB.ProgressBar(widgets=[pid+' ', PGB.Percentage(), PGB.Bar(marker='-', left=' |', right='| '), PGB.Counter(), '/{0:0d} Iterations '.format(maxiter), PGB.ETA()], maxval=maxiter, fd=writer).start()
    while not terminate:
        itr += 1
        indmaxres = np.argmax(np.abs(res*cbox))  ## This gives the position of 'peak' value of the data in first loop, cause I have copied the data in res. In 2nd loop it gives the 'peak' in residual. 
        maxres = res[indmaxres]                ## Value of the 'peak' in the data. 
        
        ccval = gain * maxres     ## now I multiply the gain (0.01 in my case, you may give anything in the start) with the peak. 
        cc[indmaxres] += ccval     ## I put the gain*peak value to the Component list at the position of the peak. 
        res = res - ccval * np.roll(kernel, indmaxres-kmaxind) ## rolling shift the kernel/dirty beam to the 'peak' position 
        
        prevrms = np.copy(currentrms[-1])
       # currentrms += [np.std(res)]
        currentrms += [np.median(np.abs(res - np.median(res)))]

       # inrms += [np.std(res[cbox])]
        inrms += [np.median(np.abs(res[cbox] - np.median(res[cbox])))]
            
        # cond1 = np.abs(maxres) <= inrms[-1]
        cond1 = np.abs(maxres) <= lolim * np.abs(inp).max()
        cond2 = itr >= maxiter
        terminate = cond1 or cond2
        if outrms is not None:
            # outrms += [np.std(res[np.invert(cbox)])]
            outrms += [np.median(np.abs(res[np.invert(cbox)] - np.median(res[np.invert(cbox)])))]
            cond3 = inrms[-1] <= outrms[-1]
            terminate = terminate or cond3

        if progressbar:
            progress.update(itr)
    if progressbar:
        progress.finish()

    inrms = np.asarray(inrms)
    currentrms = np.asarray(currentrms)
    if outrms is not None:
        outrms = np.asarray(outrms)
        
    outdict = {'termination':{'threshold': cond1, 'maxiter': cond2, 'inrms<outrms': cond3}, 'iter': itr, 'rms': currentrms, 'inrms': inrms, 'outrms': outrms, 'cc': cc, 'res': res}

    return outdict




def downsampler(inp, factor, axis=-1, verbose=False, method='interp',
                kind='linear', fill_value=np.nan):

    """
    -----------------------------------------------------------------------------
    Routine to downsample a given input sequence along a specific dimension 
    where the input could be multi-dimensional (up to 8 dimensions)

    Inputs:

    inp           [Numpy array] array which has to be downsampled. Cannot have
                  more than 8 dimensions

    factor        [scalar] downsampling factor. positive integer or floating
                  point number greater than or equal to unity. If an integer, 
                  output is simply a sampled subset of the input. If not an 
                  integer, downsampling is obtained by interpolation.

    Keyword Inputs:

    axis          [scalar] Integer specifying the axis along which the array is
                  to be downsampled. Default = -1, the last axis.

    verbose       [Boolean] If set to True, will print progress and/or
                  diagnostic messages. If False, will suppress printing such
                  messages. Default = True

    method        [string] Specifies the method for resampling. Accepted values
                  are 'FFT' and 'interp' (default) for FFT-based and 
                  interpolation based techniques respectively. If method chosen
                  is 'interp' then value in input keyword kind determines the
                  the kind of interpolation. 

    kind          [string] Spcifies the kind of interpolation. Applies only if 
                  value of keyword input method is set to 'interp'. This is 
                  used only if factor is not an integer thus requiring 
                  interpolation. Accepted values are 'linear', 'quadratic' and 
                  'cubic'. Default = 'linear'

    fill_value    [scalar] Value to fill locations outside the index range of 
                  input array. Default = NaN
    -----------------------------------------------------------------------------
    """

    try:
        inp
    except NameError:
        raise NameError('No input specified. Aborting downsampler().')

    try:
        factor
    except NameError:
        if verbose:
            print('No downsampling factor specified. No downsampling performed on input.')
        return input

    if not isinstance(inp, np.ndarray):
        raise TypeError('Input should be a numpy array. Aborting downsampler().')

    if not isinstance(factor, (int, float)):
        raise TypeError('Downsampling factor must be a scalar value.')

    if factor < 1.0:
        raise ValueError('Downsampling factor must be greater than 1.')

    if (axis < -inp.ndim) or (axis > inp.ndim):
        raise IndexError('The axis specified does not exist in the input. Aborting downsampler().')

    if inp.ndim > 8:
        raise ValueError('The routine cannot handle inputs with more than 8 dimensions. Aborting downsampler().')

    axis = range(inp.ndim)[axis]
    if (factor % 1) == 0:
        factor = int(factor)
        if inp.ndim == 1:
            return inp[::factor]
        elif inp.ndim == 2:
            if (axis + 8) % 8 == 0:
                return inp[::factor,:]
            elif (axis + 8) % 8 == 1:
                return inp[:,::factor]
        elif inp.ndim == 3:
            if (axis + 8) % 8 == 0:
                return inp[::factor,:,:]
            elif (axis + 8) % 8 == 1:
                return inp[:,::factor,:]
            elif (axis + 8) % 8 == 2:
                return inp[:,:,::factor]
        elif inp.ndim == 4:
            if (axis + 8) % 8 == 0:
                return inp[::factor,:,:,:]
            elif (axis + 8) % 8 == 1:
                return inp[:,::factor,:,:]
            elif (axis + 8) % 8 == 2:
                return inp[:,:,::factor,:]
            elif (axis + 8) % 8 == 3:
                return inp[:,:,:,::factor]
        elif inp.ndim == 5:
            if (axis + 8) % 8 == 0:
                return inp[::factor,:,:,:,:]
            elif (axis + 8) % 8 == 1:
                return inp[:,::factor,:,:,:]
            elif (axis + 8) % 8 == 2:      
                return inp[:,:,::factor,:,:]
            elif (axis + 8) % 8 == 3:      
                return inp[:,:,:,::factor,:]
            elif (axis + 8) % 8 == 4:      
                return inp[:,:,:,:,::factor]
        elif inp.ndim == 6:
            if (axis + 8) % 8 == 0:
                return inp[::factor,:,:,:,:,:]
            elif (axis + 8) % 8 == 1:
                return inp[:,::factor,:,:,:,:]
            elif (axis + 8) % 8 == 2:      
                return inp[:,:,::factor,:,:,:]
            elif (axis + 8) % 8 == 3:      
                return inp[:,:,:,::factor,:,:]
            elif (axis + 8) % 8 == 4:      
                return inp[:,:,:,:,::factor,:]
            elif (axis + 8) % 8 == 5:      
                return inp[:,:,:,:,:,::factor]
        elif inp.ndim == 7:
            if (axis + 8) % 8 == 0:
                return inp[::factor,:,:,:,:,:,:]
            elif (axis + 8) % 8 == 1:
                return inp[:,::factor,:,:,:,:,:]
            elif (axis + 8) % 8 == 2:      
                return inp[:,:,::factor,:,:,:,:]
            elif (axis + 8) % 8 == 3:      
                return inp[:,:,:,::factor,:,:,:]
            elif (axis + 8) % 8 == 4:      
                return inp[:,:,:,:,::factor,:,:]
            elif (axis + 8) % 8 == 5:      
                return inp[:,:,:,:,:,::factor,:]
            elif (axis + 8) % 8 == 6:      
                return inp[:,:,:,:,:,:,::factor]
        elif inp.ndim == 8:
            if (axis + 8) % 8 == 0:
                return inp[::factor,:,:,:,:,:,:,:]
            elif (axis + 8) % 8 == 1:
                return inp[:,::factor,:,:,:,:,:,:]
            elif (axis + 8) % 8 == 2:      
                return inp[:,:,::factor,:,:,:,:,:]
            elif (axis + 8) % 8 == 3:      
                return inp[:,:,:,::factor,:,:,:,:]
            elif (axis + 8) % 8 == 4:      
                return inp[:,:,:,:,::factor,:,:,:]
            elif (axis + 8) % 8 == 5:      
                return inp[:,:,:,:,:,::factor,:,:]
            elif (axis + 8) % 8 == 6:      
                return inp[:,:,:,:,:,:,::factor,:]
            elif (axis + 8) % 8 == 7:      
                return inp[:,:,:,:,:,:,:,::factor]
    else:
        if method == 'interp':
            if verbose:
                print('Determining the interpolating function for downsampling.')
            tol = 1e-10
            reqd_inds = np.arange(0, inp.shape[axis]-1+tol, factor)

            # intpfunc = interpolate.interp1d(np.arange(inp.shape[axis]), inp,
            #                                 kind=kind, fill_value=fill_value,
            #                                 axis=axis) 
            # result = intpfunc(reqd_inds)

            result = interpolate_array(inp, np.arange(inp.shape[axis]), reqd_inds, axis=axis, kind=kind)
        elif method in ['FFT', 'fft']:
            nout = np.round(inp.shape[axis] / factor).astype(int)
            result = signal.resample(inp, nout, t=None, axis=axis, window=None)
        else:
            raise ValueError('Invalid method specified for downsampling')

        if verbose:
            print('Returning the downsampled data.')
        return result



def interpolate_array(inparray, inploc, outloc, axis=-1, kind='linear'):

    """
    -----------------------------------------------------------------------------
    Interpolate a multi-dimensional array along one of its dimensions. It acts 
    as a wrapper to scipy.interpolate.interp1d but applies boundary conditions 
    differently

    Inputs:

    inparray    [numpy array] Multi-dimensional input array which will be used 
                in determining the interpolation function

    inploc      [numpy array] Locations using which the interpolation function
                is determined. It must be of size equal to the dimension of 
                input array along which interpolation is to be determined 
                specified by axis keyword input. It must be a list or numpy 
                array

    outloc      [list or numpy array] Locations at which interpolated array is
                to be determined along the specified axis. It must be a scalar, 
                list or numpy array. If any of outloc is outside the range of
                inploc, the first and the last cubes from the inparray will
                be used as boundary values

    axis        [scalar] Axis along which interpolation is to be performed. 
                Default=-1 (last axis)

    kind        [string or integer] Specifies the kind of interpolation as a 
                string ('linear', 'nearest', 'zero', 'slinear', 'quadratic', 
                'cubic' where 'slinear', 'quadratic' and 'cubic' refer to a 
                spline interpolation of first, second or third order) or as an 
                integer specifying the order of the spline interpolator to use. 
                Default is 'linear'.

    Output:

    outarray    [numpy array] Output array after interpolation 
    -----------------------------------------------------------------------------
    """

    assert isinstance(inparray, np.ndarray), 'Input array inparray must be a numpy array'
    assert isinstance(inploc, (list, np.ndarray)), 'Input locations must be a list or numpy array'
    assert isinstance(outloc, (int, float, list, np.ndarray)), 'Output locations must be a scalar, list or numpy array'
    assert isinstance(axis, int), 'Interpolation axis must be an integer'
    assert isinstance(kind, str), 'Kind of interpolation must be a string'

    inploc = np.asarray(inploc).reshape(-1)
    outloc = np.asarray(outloc).reshape(-1)
    assert axis < inparray.ndim, 'Insufficient dimensions in inparray for interpolation'
    assert inparray.shape[axis]==inploc.size, 'Dimension of interpolation axis of inparray is mismatched with number of locations at which interpolation is requested'

    interp_required = True
    if inploc.size == outloc.size:
        if np.allclose(inploc, outloc):
            interp_required = False
            return inparray # no interpolation required, just return outarray=inparray
    if interp_required:
        inbound_ind = np.where(np.logical_and(outloc >= inploc.min(), outloc <= inploc.max()))[0]
        outbound_low_ind = np.where(outloc < inploc.min())[0]
        outbound_high_ind = np.where(outloc > inploc.max())[0]
    
        outarray = None
        if inbound_ind.size > 0:
            interpfunc = interpolate.interp1d(inploc, inparray, kind=kind, axis=axis, copy=False, assume_sorted=False)
            outarray = interpfunc(outloc[inbound_ind])
        if outbound_low_ind.size > 0:
            if outarray is None:
                outarray = np.repeat(np.take(inparray, [np.argmin(inploc)], axis=axis), outbound_low_ind.size, axis=axis)
            else:
                outarray = np.concatenate((np.repeat(np.take(inparray, [np.argmin(inploc)], axis=axis), outbound_low_ind.size, axis=axis), outarray), axis=axis)
        if outbound_high_ind.size > 0:
            if outarray is None:
                outarray = np.repeat(np.take(inparray, [np.argmax(inploc)], axis=axis), outbound_high_ind.size, axis=axis)
            else:
                outarray = np.concatenate((outarray, np.repeat(np.take(inparray, [np.argmax(inploc)], axis=axis), outbound_high_ind.size, axis=axis)), axis=axis)
    
        return outarray


