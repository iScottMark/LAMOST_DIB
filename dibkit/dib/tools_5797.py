from scipy.optimize import curve_fit
import numpy as np
import emcee


# =========================================================================
# Define the curve fitting method

def get_indi_5797_sg_cf(wave, flux_mask, mu0, a0, sigma0=2.0):
    ''' 'INDI'vidually Measure the DIB 5797 as a 'S'ingle 'G'aussian profile using the 'C'urve 'F'itting method

    Parameters
    ----------
    wave : array
        The wavelength of the spectrum which has been preprocessed and processed
    flux_mask : array
        The masked flux
    mu0 : float
        The initial guess of the central wavelength of the DIB 5797 after pre-detection
    a0 : float
        The initial guess of the absorption depth of the DIB 5797 after pre-detection
    sigma0 : float, Default = 2.0
        The initial guess of the standard deviation of the DIB 5797 after pre-detection

    Returns
    -------
    params : array
        The fitted parameters of the DIB 5797 gaussian profile
        params[0] : The absorption depth
        params[1] : The central wavelength
        params[2] : The standard deviation
    perr : array
        The error of the fitted parameters of the DIB gaussian profile, i.e., the standard deviation of the `params`
    '''
    initial_guess = [-a0, mu0, sigma0]
    bounds = ([-0.3, 5790, 0.1],
              [0.0, 5804, 5.0])
    params, pcov = curve_fit(gaussian_c, wave, flux_mask, p0=initial_guess, bounds=bounds, maxfev=100000)
    perr = np.sqrt(np.diag(pcov))

    return np.array(params), np.array(perr)


# =========================================================================
# Define the MCMC method

def get_indi_5797_sg_mcmc(wave, flux_mask, params, flux_err=0.0):
    ''' 'INDI'vidually Measure the DIB 5797 as a 'S'ingle 'G'aussian profile using the 'M'arkov 'C'hain 'M'onte 'C'arlo method

    Parameters
    ----------
    wave : array
        The wavelength of the spectrum which has been preprocessed and processed
    flux_mask : array
        The masked flux
    params : array
        The DIB 5797 parameters obtained from the curve fitting method
    flux_err : array, Default = 0.0
        The error of the flux. Actually, it is the inverse variance = 1 / error^2

    Returns
    -------
    pfit : array
        The fitted parameters of the DIB 5797 gaussian profile
    perr : array
        The error of the fitted parameters of the DIB 5797 gaussian profile
        The error is calculated as the mean value of the 16th and 84th percentiles of the samples
    samples : array
        The samples of the fitted parameters of the DIB 5797 gaussian profile
    '''
    nwalkers = 100
    ndim = params.shape[0]

    if np.sum(flux_err) == 0.0:
        flux_err = 1e-3 * np.ones_like(flux_mask)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_5797_sg, args=(params, wave, flux_mask, flux_err))
    pos = params + np.random.normal(0, 0.01, (nwalkers, ndim))
    # Do the burn-in
    s1, _, _ = sampler.run_mcmc(pos, 50)
    # Do the production
    sampler.reset()
    sampler.run_mcmc(s1, 200);
    samples = sampler.flatchain
    results = np.percentile(samples, [16, 50, 84], axis=0)
    pfit = results[1]
    perr = np.abs(results[0] - results[2]) / 2.0

    return pfit, perr, samples


# =========================================================================
# Define some Gaussian profile functions
def gaussian(x, a, b, c):
    return a * np.exp(-(x - b) ** 2 / (2 * c ** 2))


def gaussian_c(x, a, b, c):
    return gaussian(x, a, b, c) + 1.0


# =========================================================================
# Define the log probability function for the MCMC
# DIB 5797 with a single Gaussian profile
def pr_a(a, a0):
    width = 0.5 * np.abs(a0)
    return np.exp(-(a - a0) ** 2 / (2.0 * width ** 2))


def pr_mu(mu, mu0):
    width = 0.5
    return np.exp(-(mu - mu0) ** 2 / (2.0 * width ** 2))


def pr_sigma(sigma, sigma0):
    # sigma - measured width for Gaussian profile
    width = 0.5
    return np.exp(-(sigma - sigma0) ** 2 / (2.0 * width ** 2))


def lnlike_5797_sg(p, x, y, yerr):
    inv_sigma2 = 1.0 / yerr ** 2
    return -0.5 * (np.sum((y - gaussian_c(x, *p)) ** 2 * inv_sigma2 - np.log(inv_sigma2)))


def lnprior_5797_sg(p, p_guess):
    # unpack the parameters:
    a_5797, mu_5797, sigma_5797 = p
    a0_5797, mu0_5797, sigma0_5797 = p_guess

    flag_a_5797 = (a_5797 >= -0.3) & (a_5797 < 0.0)
    flag_mu_5797 = (mu_5797 >= 5790.0) & (mu_5797 < 5804.0)
    flag_sigma_5797 = (sigma_5797 >= 0.1) & (sigma_5797 < 5)

    if flag_a_5797 & flag_mu_5797 & flag_sigma_5797:
        lp_a_5797 = np.log(pr_a(a_5797, a0_5797))
        lp_mu_5797 = np.log(pr_mu(mu_5797, mu0_5797))
        lp_sigma_5797 = np.log(pr_sigma(sigma_5797, sigma0_5797))
        return lp_a_5797 + lp_mu_5797 + lp_sigma_5797
    return -np.inf


def lnprob_5797_sg(p, p_guess, x, y, yerr):
    lp = lnprior_5797_sg(p, p_guess)
    if not np.isfinite(lp):
        return -np.inf
    total = lp + lnlike_5797_sg(p, x, y, yerr)
    if np.isnan(total):
        return -np.inf
    return total
