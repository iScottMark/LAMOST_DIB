import os
import numpy as np
from numpy.typing import ArrayLike
from astropy.time import Time
from astropy.io import fits
from uncertainties import ufloat


# =============================================================================
# The following functions can be reused in other projects.
def compute_distance(x: ArrayLike, y: ArrayLike) -> float:
    """
    Compute the distance between two vectors with the same length.

    Parameters
    ----------
    x : ArrayLike
        The first vector.
    y : ArrayLike
        The second vector.

    Returns
    -------
    distance : float
        The distance between the two vectors.
    """
    return np.sum(np.abs(x - y))


def shift_wave(wave, rv):
    """Shift the wavelength into the rest frame according to the radial velocity. Refer to
    https://github.com/tingyuansen/The_Payne/blob/e4bebc7eccb75c52dd0e7b419b5ab19b9a2f9df0/The_Payne/utils.py#L82
    # ! But the Ting's code is contradictory to the explanation in the docstring.

    Parameters
    ----------
    wave : array-like
        The wavelength array.
    rv : float
        The radial velocity.

    Returns
    -------
    rest_wave : array-like
        The rest wavelength array.
    interp_flux : array-like
        The interpolated flux array.
    """
    c = 299792.458  # the speed of light in km/s
    doppler_factor = np.sqrt((1 - rv / c) / (1 + rv / c))
    rest_wave = wave * doppler_factor
    return rest_wave


"""
This module implements the conversion of wavelengths between vacuum and air
Reference: Donald Morton (2000, ApJ. Suppl., 130, 403)
VALD3 link: http://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion
"""


# the code is copied from https://github.com/hypergravity/laspec/blob/master/laspec/wavelength.py

def vac2air(wave_vac):
    """
    Parameters
    ----------
    wave_vac:
        wavelength (A) in vacuum

    Return
    ------
    wave_air:
        wavelength (A) in air
    """
    wave_vac = np.array(wave_vac)
    s = 1e4 / wave_vac
    n = 1 + 0.0000834254 + \
        0.02406147 / (130 - s ** 2) + \
        0.00015998 / (38.9 - s ** 2)
    return wave_vac / n


def air2vac(wave_air):
    """
    Parameters
    ----------
    wave_air:
        wavelength (A) in air

    Return
    ------
    wave_vac:
        wavelength (A) in vacuum
    """
    wave_air = np.array(wave_air)
    s = 1e4 / wave_air
    n = 1 + 0.00008336624212083 + \
        0.02408926869968 / (130.1065924522 - s ** 2) + \
        0.0001599740894897 / (38.92568793293 - s ** 2)
    return wave_air * n


def cut_array(x, y, yerr, xmin, xmax):
    idx = np.where((x >= xmin) & (x <= xmax))
    x_cut = x[idx]
    y_cut = y[idx]
    yerr_cut = yerr[idx]

    return x_cut, y_cut, yerr_cut


def rebin_array(x, y, x_rebin):
    y_rebin = np.interp(x_rebin, x, y)
    return y_rebin


def specNormg(spec, order=5, niter=20, lowrej=0.5, highrej=5):
    """
    Georges Kordopatis
    version: 29/03/2019
    INPUT:
    order: integer - order of the polynomial to fit the continuum
    niter: integer - number of iterations
    lowrej: float - a point is rejected when goes lowrej times sigma under the fit
    highrej: float - a point is rejected when goes highrej times sigma above the fit
    spec: fltarr - input spectrum
    OUTPUT: continuum at the same dimensions as spec and normalised spectrum
    """
    npix = len(spec)
    original = spec.copy()
    spectrum = spec.copy()
    x0 = np.arange(npix)

    for i in range(0, niter):
        # fc=np.polyfit(x0, spectrum, order)
        # contin=np.poly1d(fc)
        seo = np.polynomial.polynomial.Polynomial.fit(x0, spectrum, order)
        # may have the issue of accuracy of the polynomial using different versions of numpy
        contin = seo.linspace(n=npix)[1]

        residual = spectrum - contin
        sigma = np.nanstd(residual)

        f = (spectrum < contin - lowrej * sigma) | (spectrum > contin + highrej * sigma)
        spectrum[f] = contin[f]

    continuum = contin
    normed = original / continuum
    return continuum, normed


def pre_detect(wave, flux_norm, snr, lower_lambda, upper_lambda):
    '''Pre-detection of the DIB 6283
    Parameters
    ----------
    wave : array-like
        The wavelength array
    flux_norm : array-like
        The normalized flux array
    snr : float
        The signal-to-noise ratio
    lower_lambda : float
        The lower detection boundary of the wavelength range
    upper_lambda : float
        The upper detection boundary of the wavelength range

    Returns
    -------
    detect_flag : int
        0 or 1, 0 for undetected, 1 for detected
    wave_c : float
        the central wavelength of the DIB
    a_v : float
        the absorption depth of the DIB
    threshold : float
        the threshold of the absorption depth
    '''
    t = (wave >= lower_lambda) & (wave <= upper_lambda)
    r = flux_norm[t]
    w = wave[t]

    # DISCARDED METHOD
    # i = np.argmin(r)
    # if i-2 < 0:
    #     lowy = 1. - np.nanmean(r[i:i+5])
    #     lowx = np.nanmean(w[i:i+5])
    # elif i+3 > r.shape[0]:
    #     lowy = 1. - np.nanmean(r[i-4:i+1])
    #     lowx = np.nanmean(w[i-4:i+1])
    # else:
    #     lowy = 1. - np.nanmean(r[i-2:i+3])
    #     lowx = np.nanmean(w[i-2:i+3])

    lowy = 1. - np.min(r)
    lowx = w[np.argmin(r)]

    sdv = 1. / snr
    threshold = 1 * sdv  # 3 sigma which can be modified if needed

    a_v = np.abs(lowy)
    wave_c = lowx

    if a_v >= threshold:
        detect_flag = 1
    else:
        detect_flag = 0
    return detect_flag, wave_c, a_v, threshold


def dynamic_mask(wave, flux, cw, width):
    ''' Dynamically determine the range of mask

    Parameters
    ----------
    wave : array-like
        The wavelength array
    flux : array-like
        The flux array
    cw : float or list-like
        The central wavelength of the DIB
    width : float
        The width needs to remain

    Returns
    -------
    flux_copy : array-like
        The masked flux array
    '''
    flux_copy = flux.copy()  # ! NOTE: The original flux array will be modified if a copy is not made
    # if type(cw) != list:
    if not isinstance(cw, list):
        lower_w = cw - width
        upper_w = cw + width
    else:
        lower_w = cw[0] - width
        upper_w = cw[1] + width
    mask = (wave <= lower_w) | (wave >= upper_w)
    flux_copy[mask] = 1.

    return flux_copy


def ew_measure(depth, depth_err, sigma, sigma_err):
    '''Measure the equivalent width of a DIB and give the error by error propagation

    Parameters
    ----------
    depth : float
        The absorption depth of the DIB
    depth_err : float
        The error of the absorption depth
    sigma : float
        The standard deviation of the absorption with Gaussian Profile
    sigma_err : float
        The error of the standard deviation

    Returns
    -------
    ew : float
        The equivalent width of the DIB
    ew_err : float
        The error of the equivalent width
    '''

    # ew_err = np.sqrt((np.sqrt(2.0*np.pi) * sigma * depth_err)**2 + (np.sqrt(2.0*np.pi) * depth * sigma_err)**2)
    u = ufloat(depth, depth_err)
    v = ufloat(sigma, sigma_err)
    y = np.abs(np.sqrt(2.0 * np.pi) * u * v)
    ew = y.n
    ew_err = y.s
    return ew, ew_err


def get_weighted_avg(arr, weight):
    mask_arr = np.ma.array(arr, mask=np.isinf(arr))
    mask_avg = np.average(mask_arr, axis=0, weights=weight)
    return np.ma.getdata(mask_avg)


# =============================================================================
# Some functions are specific to process the data of LAMOST.

# Data Path
def assemble_fitsname(lmjd, planid, spid, fiberid):
    mmmmm = lmjd
    yyyy = planid
    xx = spid
    fff = fiberid
    fitsname = f'spec-{mmmmm:05d}-{yyyy}_sp{xx:02d}-{fff:03d}.fits.gz'

    return fitsname


def assemble_prefix(mjd, planid):
    """
    ! NOTE: In LAMOST, the directory is organized by MJD rather than LMJD.
            However, the LMJD is one of the components of the FITS file name.
    """
    utc = Time(mjd, format='mjd').ymdhms
    year = utc[0]
    month = utc[1]
    day = utc[2]
    prefix = f'{year}{month:02d}{day:02d}/{planid}/'

    return prefix


def assemble_path(lmjd, mjd, planid, spid, fiberid):
    prefix = assemble_prefix(mjd, planid)
    fitsname = assemble_fitsname(lmjd, planid, spid, fiberid)
    path = prefix + fitsname

    return path


def get_spec_path(spec_info, data_dir):
    assemble_list = [spec_info['lmjd'], spec_info['mjd'], 
                     spec_info['planid'].decode(), spec_info['spid'], spec_info['fiberid']]
    relative_path = assemble_path(*assemble_list)
    absolute_path = os.path.join(data_dir, relative_path)
    return absolute_path


def read_fits(path):
    with fits.open(path) as f:
        # refer to the LAMOST LRS DOCs for the data structure
        # http://www.lamost.org/dr10/v1.0/doc/lr-data-production-description_cn#s2.4.2
        data = np.array(f[1].data)
        flux = data[0][0]
        ivar = data[0][1]  # inverse variance = 1 / error^2
        wave = data[0][2]
        ormask = data[0][4]
        flux_norm = data[0][5]  # a normalized flux provided by LAMOST pipeline

        flux_new = flux[np.where(ormask == 0)]
        flux_norm_new = flux_norm[np.where(ormask == 0)]
        wave_new = wave[np.where(ormask == 0)]
        ivar_new = ivar[np.where(ormask == 0)]
    return wave_new, flux_new, flux_norm_new, ivar_new
