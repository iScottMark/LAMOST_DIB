import numpy as np

from dibkit.core.utils import pre_detect, dynamic_mask, ew_measure

from dibkit.dib.tools_5780 import get_indi_5780_sg_cf, get_indi_5780_dg_cf, get_indi_5780_sg_mcmc, get_indi_5780_dg_mcmc
from dibkit.dib.tools_5797 import get_indi_5797_sg_cf, get_indi_5797_sg_mcmc
from dibkit.dib.tools_6614 import get_6614_cf, get_6614_mcmc
from dibkit.dib.tools_simu import get_simu_sg_cf, get_simu_sg_mcmc, get_simu_dg_cf, get_simu_dg_mcmc


def measure_dib5780_lite(wave, flux, flux_err, snr):
    ''' Measure the DIB 5780 without considering the blending with DIB 5778

    Parameters
    ----------
    wave : array
        The wavelength of the spectrum which has been preprocessed and processed
    flux : array
        The renormalized flux of the spectrum
    flux_err : array
        The error of the flux. Actually, it is the inverse variance = 1 / error^2
    snr: float
        The signal-to-noise ratio of the spectrum in the 'r' waveband

    Returns
    -------
    dib_dict : dict
        The measurement results of the DIB 5780, DIB 5797 and DIB 5778
    '''
    # =========================================================================
    # Convert the data type to float64 for the following calculation like MCMC
    x, y, y_err = np.float64(wave), np.float64(flux), np.float64(flux_err)

    # Pre-detection for the DIB 5780
    is_5780_detect, mu0_5780, a0_5780, eps = pre_detect(x, y, snr, lower_lambda=5775, upper_lambda=5785)
    is_5797_detect, mu0_5797, a0_5797, _ = pre_detect(x, y, snr, lower_lambda=5793, upper_lambda=5801)

    # =========================================================================
    # Fit the DIB 5780 and DIB 5797 separately
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Only fit the DIB 5780
    if is_5780_detect == 1:
        y_mask = dynamic_mask(x, y, mu0_5780, width=10)  # dynamically determine the range of mask

        # fit DIB 5780 as a single gaussian profile
        p_indi_5780_sg_cf, perr_indi_5780_sg_cf = get_indi_5780_sg_cf(x, y_mask, mu0_5780, a0_5780)
        p_indi_5780_sg_mc, perr_indi_5780_sg_mc, _ = get_indi_5780_sg_mcmc(x, y_mask, p_indi_5780_sg_cf, y_err)

    else:
        p_indi_5780_sg_cf = -1.0 * np.ones(3)
        p_indi_5780_sg_mc = -1.0 * np.ones(3)
        perr_indi_5780_sg_cf = -1.0 * np.ones(3)
        perr_indi_5780_sg_mc = -1.0 * np.ones(3)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Only fit the DIB 5797
    if is_5797_detect == 1:
        y_mask = dynamic_mask(x, y, mu0_5797, width=6)  # dynamically determine the range of mask

        # fit DIB 5797 as a single gaussian profile
        p_indi_5797_sg_cf, perr_indi_5797_sg_cf = get_indi_5797_sg_cf(x, y_mask, mu0_5797, a0_5797)
        p_indi_5797_sg_mc, perr_indi_5797_sg_mc, _ = get_indi_5797_sg_mcmc(x, y_mask, p_indi_5797_sg_cf, y_err)
    else:
        p_indi_5797_sg_cf = -1.0 * np.ones(3)
        p_indi_5797_sg_mc = -1.0 * np.ones(3)
        perr_indi_5797_sg_cf = -1.0 * np.ones(3)
        perr_indi_5797_sg_mc = -1.0 * np.ones(3)

    # =========================================================================
    # Fit the DIB 5780 and DIB 5797 simultaneously
    if is_5780_detect == 1 and is_5797_detect == 1:
        y_mask = dynamic_mask(x, y, [mu0_5780, mu0_5797], width=6)  # dynamically determine the range of mask

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # fit DIB 5780 as a single gaussian profile
        p_simu_sg_cf, perr_simu_sg_cf = get_simu_sg_cf(x, y_mask, [mu0_5780, mu0_5797], [a0_5780, a0_5797])
        p_simu_sg_mc, perr_simu_sg_mc, _ = get_simu_sg_mcmc(x, y_mask, p_simu_sg_cf, y_err)

    else:
        p_simu_sg_cf = -1.0 * np.ones(6)
        p_simu_sg_mc = -1.0 * np.ones(6)
        perr_simu_sg_cf = -1.0 * np.ones(6)
        perr_simu_sg_mc = -1.0 * np.ones(6)

    # =========================================================================
    # Measure the equivalent width of DIB 5780 and DIB 5797

    if is_5780_detect == 1:
        # DIB 5780
        ew_5780_sg_cf, ew_err_5780_sg_cf = ew_measure(p_indi_5780_sg_cf[0], perr_indi_5780_sg_cf[0],
                                                      p_indi_5780_sg_cf[2], perr_indi_5780_sg_cf[2])
        ew_5780_sg_mc, ew_err_5780_sg_mc = ew_measure(p_indi_5780_sg_mc[0], perr_indi_5780_sg_mc[0],
                                                      p_indi_5780_sg_mc[2], perr_indi_5780_sg_mc[2])
    else:
        ew_5780_sg_cf, ew_err_5780_sg_cf = -1.0, -1.0
        ew_5780_sg_mc, ew_err_5780_sg_mc = -1.0, -1.0

    if is_5797_detect == 1:
        # DIb 5797
        ew_5797_sg_cf, ew_err_5797_sg_cf = ew_measure(p_indi_5797_sg_cf[0], perr_indi_5797_sg_cf[0],
                                                      p_indi_5797_sg_cf[2], perr_indi_5797_sg_cf[2])
        ew_5797_sg_mc, ew_err_5797_sg_mc = ew_measure(p_indi_5797_sg_mc[0], perr_indi_5797_sg_mc[0],
                                                      p_indi_5797_sg_mc[2], perr_indi_5797_sg_mc[2])
    else:
        ew_5797_sg_cf, ew_err_5797_sg_cf = -1.0, -1.0
        ew_5797_sg_mc, ew_err_5797_sg_mc = -1.0, -1.0

    if is_5780_detect == 1 and is_5797_detect == 1:
        # DIB 5780
        ew_5780_sg_cf_simu, ew_err_5780_sg_cf_simu = ew_measure(p_simu_sg_cf[0], perr_simu_sg_cf[0], p_simu_sg_cf[2],
                                                                perr_simu_sg_cf[2])
        ew_5780_sg_mc_simu, ew_err_5780_sg_mc_simu = ew_measure(p_simu_sg_mc[0], perr_simu_sg_mc[0], p_simu_sg_mc[2],
                                                                perr_simu_sg_mc[2])

        # DIB 5797
        ew_5797_sg_cf_simu, ew_err_5797_sg_cf_simu = ew_measure(p_simu_sg_cf[3], perr_simu_sg_cf[3], p_simu_sg_cf[5],
                                                                perr_simu_sg_cf[5])
        ew_5797_sg_mc_simu, ew_err_5797_sg_mc_simu = ew_measure(p_simu_sg_mc[3], perr_simu_sg_mc[3], p_simu_sg_mc[5],
                                                                perr_simu_sg_mc[5])

    else:
        ew_5780_sg_cf_simu, ew_err_5780_sg_cf_simu = -1.0, -1.0
        ew_5780_sg_mc_simu, ew_err_5780_sg_mc_simu = -1.0, -1.0
        ew_5797_sg_cf_simu, ew_err_5797_sg_cf_simu = -1.0, -1.0
        ew_5797_sg_mc_simu, ew_err_5797_sg_mc_simu = -1.0, -1.0

    # =========================================================================
    # Assemble the results as a dictionary
    dib_dict = {
        # 'obsid': obsid,
        'threshold': eps,
        'is_5780_detect': is_5780_detect,
        'is_5797_detect': is_5797_detect,

        # DIB 5780
        'a0_5780': a0_5780,
        'mu0_5780': mu0_5780,

        'a_5780_sg_cf': -p_indi_5780_sg_cf[0],
        'aerr_5780_sg_cf': perr_indi_5780_sg_cf[0],
        'mu_5780_sg_cf': p_indi_5780_sg_cf[1],
        'muerr_5780_sg_cf': perr_indi_5780_sg_cf[1],
        'sig_5780_sg_cf': p_indi_5780_sg_cf[2],
        'sigerr_5780_sg_cf': perr_indi_5780_sg_cf[2],
        'ew_5780_sg_cf': ew_5780_sg_cf,
        'ewerr_5780_sg_cf': ew_err_5780_sg_cf,

        'a_5780_sg_mc': -p_indi_5780_sg_mc[0],
        'aerr_5780_sg_mc': perr_indi_5780_sg_mc[0],
        'mu_5780_sg_mc': p_indi_5780_sg_mc[1],
        'muerr_5780_sg_mc': perr_indi_5780_sg_mc[1],
        'sig_5780_sg_mc': p_indi_5780_sg_mc[2],
        'sigerr_5780_sg_mc': perr_indi_5780_sg_mc[2],
        'ew_5780_sg_mc': ew_5780_sg_mc,
        'ewerr_5780_sg_mc': ew_err_5780_sg_mc,

        'a_5780_sg_cf_simu': -p_simu_sg_cf[0],
        'aerr_5780_sg_cf_simu': perr_simu_sg_cf[0],
        'mu_5780_sg_cf_simu': p_simu_sg_cf[1],
        'muerr_5780_sg_cf_simu': perr_simu_sg_cf[1],
        'sig_5780_sg_cf_simu': p_simu_sg_cf[2],
        'sigerr_5780_sg_cf_simu': perr_simu_sg_cf[2],
        'ew_5780_sg_cf_simu': ew_5780_sg_cf_simu,
        'ewerr_5780_sg_cf_simu': ew_err_5780_sg_cf_simu,

        'a_5780_sg_mc_simu': -p_simu_sg_mc[0],
        'aerr_5780_sg_mc_simu': perr_simu_sg_mc[0],
        'mu_5780_sg_mc_simu': p_simu_sg_mc[1],
        'muerr_5780_sg_mc_simu': perr_simu_sg_mc[1],
        'sig_5780_sg_mc_simu': p_simu_sg_mc[2],
        'sigerr_5780_sg_mc_simu': perr_simu_sg_mc[2],
        'ew_5780_sg_mc_simu': ew_5780_sg_mc_simu,
        'ewerr_5780_sg_mc_simu': ew_err_5780_sg_mc_simu,

        # DIB 5797
        'a0_5797': a0_5797,
        'mu0_5797': mu0_5797,

        'a_5797_sg_cf': -p_indi_5797_sg_cf[0],
        'aerr_5797_sg_cf': perr_indi_5797_sg_cf[0],
        'mu_5797_sg_cf': p_indi_5797_sg_cf[1],
        'muerr_5797_sg_cf': perr_indi_5797_sg_cf[1],
        'sig_5797_sg_cf': p_indi_5797_sg_cf[2],
        'sigerr_5797_sg_cf': perr_indi_5797_sg_cf[2],
        'ew_5797_sg_cf': ew_5797_sg_cf,
        'ewerr_5797_sg_cf': ew_err_5797_sg_cf,

        'a_5797_sg_mc': -p_indi_5797_sg_mc[0],
        'aerr_5797_sg_mc': perr_indi_5797_sg_mc[0],
        'mu_5797_sg_mc': p_indi_5797_sg_mc[1],
        'muerr_5797_sg_mc': perr_indi_5797_sg_mc[1],
        'sig_5797_sg_mc': p_indi_5797_sg_mc[2],
        'sigerr_5797_sg_mc': perr_indi_5797_sg_mc[2],
        'ew_5797_sg_mc': ew_5797_sg_mc,
        'ewerr_5797_sg_mc': ew_err_5797_sg_mc,

        'a_5797_sg_cf_simu': -p_simu_sg_cf[3],
        'aerr_5797_sg_cf_simu': perr_simu_sg_cf[3],
        'mu_5797_sg_cf_simu': p_simu_sg_cf[4],
        'muerr_5797_sg_cf_simu': perr_simu_sg_cf[4],
        'sig_5797_sg_cf_simu': p_simu_sg_cf[5],
        'sigerr_5797_sg_cf_simu': perr_simu_sg_cf[5],
        'ew_5797_sg_cf_simu': ew_5797_sg_cf_simu,
        'ewerr_5797_sg_cf_simu': ew_err_5797_sg_cf_simu,

        'a_5797_sg_mc_simu': -p_simu_sg_mc[3],
        'aerr_5797_sg_mc_simu': perr_simu_sg_mc[3],
        'mu_5797_sg_mc_simu': p_simu_sg_mc[4],
        'muerr_5797_sg_mc_simu': perr_simu_sg_mc[4],
        'sig_5797_sg_mc_simu': p_simu_sg_mc[5],
        'sigerr_5797_sg_mc_simu': perr_simu_sg_mc[5],
        'ew_5797_sg_mc_simu': ew_5797_sg_mc_simu,
        'ewerr_5797_sg_mc_simu': ew_err_5797_sg_mc_simu,
    }

    return dib_dict


def measure_dib5780(wave, flux, flux_err, snr):
    ''' Measure the DIB 5780

    Parameters
    ----------
    wave : array
        The wavelength of the spectrum which has been preprocessed and processed
    flux : array
        The renormalized flux of the spectrum
    flux_err : array
        The error of the flux. Actually, it is the inverse variance = 1 / error^2
    snr: float
        The signal-to-noise ratio of the spectrum in the 'r' waveband

    Returns
    -------
    dib_dict : dict
        The measurement results of the DIB 5780, DIB 5797 and DIB 5778
    '''
    # =========================================================================
    # Convert the data type to float64 for the following calculation like MCMC
    x, y, y_err = np.float64(wave), np.float64(flux), np.float64(flux_err)

    # Pre-detection for the DIB 5780
    is_5780_detect, mu0_5780, a0_5780, eps = pre_detect(x, y, snr, lower_lambda=5775, upper_lambda=5785)
    is_5797_detect, mu0_5797, a0_5797, _ = pre_detect(x, y, snr, lower_lambda=5793, upper_lambda=5801)

    # =========================================================================
    # Fit the DIB 5780 and DIB 5797 separately
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Only fit the DIB 5780
    if is_5780_detect == 1:
        y_mask = dynamic_mask(x, y, mu0_5780, width=10)  # dynamically determine the range of mask

        # fit DIB 5780 as a single gaussian profile
        p_indi_5780_sg_cf, perr_indi_5780_sg_cf = get_indi_5780_sg_cf(x, y_mask, mu0_5780, a0_5780)
        p_indi_5780_sg_mc, perr_indi_5780_sg_mc, _ = get_indi_5780_sg_mcmc(x, y_mask, p_indi_5780_sg_cf, y_err)

        # fit DIB 5780 blended with the wider DIB 5778 as a double gaussian profile
        p_indi_5780_dg_cf, perr_indi_5780_dg_cf = get_indi_5780_dg_cf(x, y_mask, mu0_5780, a0_5780)
        p_indi_5780_dg_mc, perr_indi_5780_dg_mc, _ = get_indi_5780_dg_mcmc(x, y_mask, p_indi_5780_dg_cf, y_err)
    else:
        p_indi_5780_sg_cf = -1.0 * np.ones(3)
        p_indi_5780_sg_mc = -1.0 * np.ones(3)
        perr_indi_5780_sg_cf = -1.0 * np.ones(3)
        perr_indi_5780_sg_mc = -1.0 * np.ones(3)
        p_indi_5780_dg_cf = -1.0 * np.ones(6)
        p_indi_5780_dg_mc = -1.0 * np.ones(6)
        perr_indi_5780_dg_cf = -1.0 * np.ones(6)
        perr_indi_5780_dg_mc = -1.0 * np.ones(6)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Only fit the DIB 5797
    if is_5797_detect == 1:
        y_mask = dynamic_mask(x, y, mu0_5797, width=6)  # dynamically determine the range of mask

        # fit DIB 5797 as a single gaussian profile
        p_indi_5797_sg_cf, perr_indi_5797_sg_cf = get_indi_5797_sg_cf(x, y_mask, mu0_5797, a0_5797)
        p_indi_5797_sg_mc, perr_indi_5797_sg_mc, _ = get_indi_5797_sg_mcmc(x, y_mask, p_indi_5797_sg_cf, y_err)
    else:
        p_indi_5797_sg_cf = -1.0 * np.ones(3)
        p_indi_5797_sg_mc = -1.0 * np.ones(3)
        perr_indi_5797_sg_cf = -1.0 * np.ones(3)
        perr_indi_5797_sg_mc = -1.0 * np.ones(3)

    # =========================================================================
    # Fit the DIB 5780 and DIB 5797 simultaneously
    if is_5780_detect == 1 and is_5797_detect == 1:
        y_mask = dynamic_mask(x, y, [mu0_5780, mu0_5797], width=6)  # dynamically determine the range of mask

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # fit DIB 5780 as a single gaussian profile
        p_simu_sg_cf, perr_simu_sg_cf = get_simu_sg_cf(x, y_mask, [mu0_5780, mu0_5797], [a0_5780, a0_5797])
        p_simu_sg_mc, perr_simu_sg_mc, _ = get_simu_sg_mcmc(x, y_mask, p_simu_sg_cf, y_err)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # fit DIB 5780 blended with the wider DIB 5778 as a double gaussian profile
        p_simu_dg_cf, perr_simu_dg_cf = get_simu_dg_cf(x, y_mask, [mu0_5780, mu0_5797], [a0_5780, a0_5797])
        p_simu_dg_mc, perr_simu_dg_mc, _ = get_simu_dg_mcmc(x, y_mask, p_simu_dg_cf, y_err)
    else:
        p_simu_sg_cf = -1.0 * np.ones(6)
        p_simu_sg_mc = -1.0 * np.ones(6)
        perr_simu_sg_cf = -1.0 * np.ones(6)
        perr_simu_sg_mc = -1.0 * np.ones(6)
        p_simu_dg_cf = -1.0 * np.ones(9)
        p_simu_dg_mc = -1.0 * np.ones(9)
        perr_simu_dg_cf = -1.0 * np.ones(9)
        perr_simu_dg_mc = -1.0 * np.ones(9)

    # =========================================================================
    # Measure the equivalent width of DIB 5780 and DIB 5797

    if is_5780_detect == 1:
        # DIB 5778
        ew_5778_dg_cf, ew_err_5778_dg_cf = ew_measure(p_indi_5780_dg_cf[0], perr_indi_5780_dg_cf[0],
                                                      p_indi_5780_dg_cf[2], perr_indi_5780_dg_cf[2])
        ew_5778_dg_mc, ew_err_5778_dg_mc = ew_measure(p_indi_5780_dg_mc[0], perr_indi_5780_dg_mc[0],
                                                      p_indi_5780_dg_mc[2], perr_indi_5780_dg_mc[2])

        # DIB 5780
        ew_5780_sg_cf, ew_err_5780_sg_cf = ew_measure(p_indi_5780_sg_cf[0], perr_indi_5780_sg_cf[0],
                                                      p_indi_5780_sg_cf[2], perr_indi_5780_sg_cf[2])
        ew_5780_sg_mc, ew_err_5780_sg_mc = ew_measure(p_indi_5780_sg_mc[0], perr_indi_5780_sg_mc[0],
                                                      p_indi_5780_sg_mc[2], perr_indi_5780_sg_mc[2])
        ew_5780_dg_cf, ew_err_5780_dg_cf = ew_measure(p_indi_5780_dg_cf[3], perr_indi_5780_dg_cf[3],
                                                      p_indi_5780_dg_cf[5], perr_indi_5780_dg_cf[5])
        ew_5780_dg_mc, ew_err_5780_dg_mc = ew_measure(p_indi_5780_dg_mc[3], perr_indi_5780_dg_mc[3],
                                                      p_indi_5780_dg_mc[5], perr_indi_5780_dg_mc[5])
    else:
        ew_5778_dg_cf, ew_err_5778_dg_cf = -1.0, -1.0
        ew_5778_dg_mc, ew_err_5778_dg_mc = -1.0, -1.0
        ew_5780_sg_cf, ew_err_5780_sg_cf = -1.0, -1.0
        ew_5780_sg_mc, ew_err_5780_sg_mc = -1.0, -1.0
        ew_5780_dg_cf, ew_err_5780_dg_cf = -1.0, -1.0
        ew_5780_dg_mc, ew_err_5780_dg_mc = -1.0, -1.0

    if is_5797_detect == 1:
        # DIb 5797
        ew_5797_sg_cf, ew_err_5797_sg_cf = ew_measure(p_indi_5797_sg_cf[0], perr_indi_5797_sg_cf[0],
                                                      p_indi_5797_sg_cf[2], perr_indi_5797_sg_cf[2])
        ew_5797_sg_mc, ew_err_5797_sg_mc = ew_measure(p_indi_5797_sg_mc[0], perr_indi_5797_sg_mc[0],
                                                      p_indi_5797_sg_mc[2], perr_indi_5797_sg_mc[2])
    else:
        ew_5797_sg_cf, ew_err_5797_sg_cf = -1.0, -1.0
        ew_5797_sg_mc, ew_err_5797_sg_mc = -1.0, -1.0

    if is_5780_detect == 1 and is_5797_detect == 1:
        # DIB 5778
        ew_5778_dg_cf_simu, ew_err_5778_dg_cf_simu = ew_measure(p_simu_dg_cf[0], perr_simu_dg_cf[0], p_simu_dg_cf[2],
                                                                perr_simu_dg_cf[2])
        ew_5778_dg_mc_simu, ew_err_5778_dg_mc_simu = ew_measure(p_simu_dg_mc[0], perr_simu_dg_mc[0], p_simu_dg_mc[2],
                                                                perr_simu_dg_mc[2])

        # DIB 5780
        ew_5780_sg_cf_simu, ew_err_5780_sg_cf_simu = ew_measure(p_simu_sg_cf[0], perr_simu_sg_cf[0], p_simu_sg_cf[2],
                                                                perr_simu_sg_cf[2])
        ew_5780_sg_mc_simu, ew_err_5780_sg_mc_simu = ew_measure(p_simu_sg_mc[0], perr_simu_sg_mc[0], p_simu_sg_mc[2],
                                                                perr_simu_sg_mc[2])
        ew_5780_dg_cf_simu, ew_err_5780_dg_cf_simu = ew_measure(p_simu_dg_cf[3], perr_simu_dg_cf[3], p_simu_dg_cf[5],
                                                                perr_simu_dg_cf[5])
        ew_5780_dg_mc_simu, ew_err_5780_dg_mc_simu = ew_measure(p_simu_dg_mc[3], perr_simu_dg_mc[3], p_simu_dg_mc[5],
                                                                perr_simu_dg_mc[5])

        # DIB 5797
        ew_5797_sg_cf_simu, ew_err_5797_sg_cf_simu = ew_measure(p_simu_sg_cf[3], perr_simu_sg_cf[3], p_simu_sg_cf[5],
                                                                perr_simu_sg_cf[5])
        ew_5797_sg_mc_simu, ew_err_5797_sg_mc_simu = ew_measure(p_simu_sg_mc[3], perr_simu_sg_mc[3], p_simu_sg_mc[5],
                                                                perr_simu_sg_mc[5])
        ew_5797_dg_cf_simu, ew_err_5797_dg_cf_simu = ew_measure(p_simu_dg_cf[6], perr_simu_dg_cf[6], p_simu_dg_cf[8],
                                                                perr_simu_dg_cf[8])
        ew_5797_dg_mc_simu, ew_err_5797_dg_mc_simu = ew_measure(p_simu_dg_mc[6], perr_simu_dg_mc[6], p_simu_dg_mc[8],
                                                                perr_simu_dg_mc[8])
    else:
        ew_5778_dg_cf_simu, ew_err_5778_dg_cf_simu = -1.0, -1.0
        ew_5778_dg_mc_simu, ew_err_5778_dg_mc_simu = -1.0, -1.0
        ew_5780_sg_cf_simu, ew_err_5780_sg_cf_simu = -1.0, -1.0
        ew_5780_sg_mc_simu, ew_err_5780_sg_mc_simu = -1.0, -1.0
        ew_5780_dg_cf_simu, ew_err_5780_dg_cf_simu = -1.0, -1.0
        ew_5780_dg_mc_simu, ew_err_5780_dg_mc_simu = -1.0, -1.0
        ew_5797_sg_cf_simu, ew_err_5797_sg_cf_simu = -1.0, -1.0
        ew_5797_sg_mc_simu, ew_err_5797_sg_mc_simu = -1.0, -1.0
        ew_5797_dg_cf_simu, ew_err_5797_dg_cf_simu = -1.0, -1.0
        ew_5797_dg_mc_simu, ew_err_5797_dg_mc_simu = -1.0, -1.0

    # =========================================================================
    # Assemble the results as a dictionary
    dib_dict = {
        # 'obsid': obsid,
        'threshold': eps,
        'is_5780_detect': is_5780_detect,
        'is_5797_detect': is_5797_detect,

        # DIB 5780
        'a0_5780': a0_5780,
        'mu0_5780': mu0_5780,

        'a_5780_sg_cf': -p_indi_5780_sg_cf[0],
        'aerr_5780_sg_cf': perr_indi_5780_sg_cf[0],
        'mu_5780_sg_cf': p_indi_5780_sg_cf[1],
        'muerr_5780_sg_cf': perr_indi_5780_sg_cf[1],
        'sig_5780_sg_cf': p_indi_5780_sg_cf[2],
        'sigerr_5780_sg_cf': perr_indi_5780_sg_cf[2],
        'ew_5780_sg_cf': ew_5780_sg_cf,
        'ewerr_5780_sg_cf': ew_err_5780_sg_cf,

        'a_5780_sg_mc': -p_indi_5780_sg_mc[0],
        'aerr_5780_sg_mc': perr_indi_5780_sg_mc[0],
        'mu_5780_sg_mc': p_indi_5780_sg_mc[1],
        'muerr_5780_sg_mc': perr_indi_5780_sg_mc[1],
        'sig_5780_sg_mc': p_indi_5780_sg_mc[2],
        'sigerr_5780_sg_mc': perr_indi_5780_sg_mc[2],
        'ew_5780_sg_mc': ew_5780_sg_mc,
        'ewerr_5780_sg_mc': ew_err_5780_sg_mc,

        'a_5780_sg_cf_simu': -p_simu_sg_cf[0],
        'aerr_5780_sg_cf_simu': perr_simu_sg_cf[0],
        'mu_5780_sg_cf_simu': p_simu_sg_cf[1],
        'muerr_5780_sg_cf_simu': perr_simu_sg_cf[1],
        'sig_5780_sg_cf_simu': p_simu_sg_cf[2],
        'sigerr_5780_sg_cf_simu': perr_simu_sg_cf[2],
        'ew_5780_sg_cf_simu': ew_5780_sg_cf_simu,
        'ewerr_5780_sg_cf_simu': ew_err_5780_sg_cf_simu,

        'a_5780_sg_mc_simu': -p_simu_sg_mc[0],
        'aerr_5780_sg_mc_simu': perr_simu_sg_mc[0],
        'mu_5780_sg_mc_simu': p_simu_sg_mc[1],
        'muerr_5780_sg_mc_simu': perr_simu_sg_mc[1],
        'sig_5780_sg_mc_simu': p_simu_sg_mc[2],
        'sigerr_5780_sg_mc_simu': perr_simu_sg_mc[2],
        'ew_5780_sg_mc_simu': ew_5780_sg_mc_simu,
        'ewerr_5780_sg_mc_simu': ew_err_5780_sg_mc_simu,

        'a_5780_dg_cf': -p_indi_5780_dg_cf[3],
        'aerr_5780_dg_cf': perr_indi_5780_dg_cf[3],
        'mu_5780_dg_cf': p_indi_5780_dg_cf[4],
        'muerr_5780_dg_cf': perr_indi_5780_dg_cf[4],
        'sig_5780_dg_cf': p_indi_5780_dg_cf[5],
        'sigerr_5780_dg_cf': perr_indi_5780_dg_cf[5],
        'ew_5780_dg_cf': ew_5780_dg_cf,
        'ewerr_5780_dg_cf': ew_err_5780_dg_cf,

        'a_5780_dg_mc': -p_indi_5780_dg_mc[3],
        'aerr_5780_dg_mc': perr_indi_5780_dg_mc[3],
        'mu_5780_dg_mc': p_indi_5780_dg_mc[4],
        'muerr_5780_dg_mc': perr_indi_5780_dg_mc[4],
        'sig_5780_dg_mc': p_indi_5780_dg_mc[5],
        'sigerr_5780_dg_mc': perr_indi_5780_dg_mc[5],
        'ew_5780_dg_mc': ew_5780_dg_mc,
        'ewerr_5780_dg_mc': ew_err_5780_dg_mc,

        'a_5780_dg_cf_simu': -p_simu_dg_cf[3],
        'aerr_5780_dg_cf_simu': perr_simu_dg_cf[3],
        'mu_5780_dg_cf_simu': p_simu_dg_cf[4],
        'muerr_5780_dg_cf_simu': perr_simu_dg_cf[4],
        'sig_5780_dg_cf_simu': p_simu_dg_cf[5],
        'sigerr_5780_dg_cf_simu': perr_simu_dg_cf[5],
        'ew_5780_dg_cf_simu': ew_5780_dg_cf_simu,
        'ewerr_5780_dg_cf_simu': ew_err_5780_dg_cf_simu,

        'a_5780_dg_mc_simu': -p_simu_dg_mc[3],
        'aerr_5780_dg_mc_simu': perr_simu_dg_mc[3],
        'mu_5780_dg_mc_simu': p_simu_dg_mc[4],
        'muerr_5780_dg_mc_simu': perr_simu_dg_mc[4],
        'sig_5780_dg_mc_simu': p_simu_dg_mc[5],
        'sigerr_5780_dg_mc_simu': perr_simu_dg_mc[5],
        'ew_5780_dg_mc_simu': ew_5780_dg_mc_simu,
        'ewerr_5780_dg_mc_simu': ew_err_5780_dg_mc_simu,

        # DIB 5797
        'a0_5797': a0_5797,
        'mu0_5797': mu0_5797,

        'a_5797_sg_cf': -p_indi_5797_sg_cf[0],
        'aerr_5797_sg_cf': perr_indi_5797_sg_cf[0],
        'mu_5797_sg_cf': p_indi_5797_sg_cf[1],
        'muerr_5797_sg_cf': perr_indi_5797_sg_cf[1],
        'sig_5797_sg_cf': p_indi_5797_sg_cf[2],
        'sigerr_5797_sg_cf': perr_indi_5797_sg_cf[2],
        'ew_5797_sg_cf': ew_5797_sg_cf,
        'ewerr_5797_sg_cf': ew_err_5797_sg_cf,

        'a_5797_sg_mc': -p_indi_5797_sg_mc[0],
        'aerr_5797_sg_mc': perr_indi_5797_sg_mc[0],
        'mu_5797_sg_mc': p_indi_5797_sg_mc[1],
        'muerr_5797_sg_mc': perr_indi_5797_sg_mc[1],
        'sig_5797_sg_mc': p_indi_5797_sg_mc[2],
        'sigerr_5797_sg_mc': perr_indi_5797_sg_mc[2],
        'ew_5797_sg_mc': ew_5797_sg_mc,
        'ewerr_5797_sg_mc': ew_err_5797_sg_mc,

        'a_5797_sg_cf_simu': -p_simu_sg_cf[3],
        'aerr_5797_sg_cf_simu': perr_simu_sg_cf[3],
        'mu_5797_sg_cf_simu': p_simu_sg_cf[4],
        'muerr_5797_sg_cf_simu': perr_simu_sg_cf[4],
        'sig_5797_sg_cf_simu': p_simu_sg_cf[5],
        'sigerr_5797_sg_cf_simu': perr_simu_sg_cf[5],
        'ew_5797_sg_cf_simu': ew_5797_sg_cf_simu,
        'ewerr_5797_sg_cf_simu': ew_err_5797_sg_cf_simu,

        'a_5797_sg_mc_simu': -p_simu_sg_mc[3],
        'aerr_5797_sg_mc_simu': perr_simu_sg_mc[3],
        'mu_5797_sg_mc_simu': p_simu_sg_mc[4],
        'muerr_5797_sg_mc_simu': perr_simu_sg_mc[4],
        'sig_5797_sg_mc_simu': p_simu_sg_mc[5],
        'sigerr_5797_sg_mc_simu': perr_simu_sg_mc[5],
        'ew_5797_sg_mc_simu': ew_5797_sg_mc_simu,
        'ewerr_5797_sg_mc_simu': ew_err_5797_sg_mc_simu,

        'a_5797_dg_cf_simu': -p_simu_dg_cf[6],
        'aerr_5797_dg_cf_simu': perr_simu_dg_cf[6],
        'mu_5797_dg_cf_simu': p_simu_dg_cf[7],
        'muerr_5797_dg_cf_simu': perr_simu_dg_cf[7],
        'sig_5797_dg_cf_simu': p_simu_dg_cf[8],
        'sigerr_5797_dg_cf_simu': perr_simu_dg_cf[8],
        'ew_5797_dg_cf_simu': ew_5797_dg_cf_simu,
        'ewerr_5797_dg_cf_simu': ew_err_5797_dg_cf_simu,

        'a_5797_dg_mc_simu': -p_simu_dg_mc[6],
        'aerr_5797_dg_mc_simu': perr_simu_dg_mc[6],
        'mu_5797_dg_mc_simu': p_simu_dg_mc[7],
        'muerr_5797_dg_mc_simu': perr_simu_dg_mc[7],
        'sig_5797_dg_mc_simu': p_simu_dg_mc[8],
        'sigerr_5797_dg_mc_simu': perr_simu_dg_mc[8],
        'ew_5797_dg_mc_simu': ew_5797_dg_mc_simu,
        'ewerr_5797_dg_mc_simu': ew_err_5797_dg_mc_simu,

        # DIB 5778
        'a_5778_dg_cf': -p_indi_5780_dg_cf[0],
        'aerr_5778_dg_cf': perr_indi_5780_dg_cf[0],
        'mu_5778_dg_cf': p_indi_5780_dg_cf[1],
        'muerr_5778_dg_cf': perr_indi_5780_dg_cf[1],
        'sig_5778_dg_cf': p_indi_5780_dg_cf[2],
        'sigerr_5778_dg_cf': perr_indi_5780_dg_cf[2],
        'ew_5778_dg_cf': ew_5778_dg_cf,
        'ewerr_5778_dg_cf': ew_err_5778_dg_cf,

        'a_5778_dg_mc': -p_indi_5780_dg_mc[0],
        'aerr_5778_dg_mc': perr_indi_5780_dg_mc[0],
        'mu_5778_dg_mc': p_indi_5780_dg_mc[1],
        'muerr_5778_dg_mc': perr_indi_5780_dg_mc[1],
        'sig_5778_dg_mc': p_indi_5780_dg_mc[2],
        'sigerr_5778_dg_mc': perr_indi_5780_dg_mc[2],
        'ew_5778_dg_mc': ew_5778_dg_mc,
        'ewerr_5778_dg_mc': ew_err_5778_dg_mc,

        'a_5778_dg_cf_simu': -p_simu_dg_cf[0],
        'aerr_5778_dg_cf_simu': perr_simu_dg_cf[0],
        'mu_5778_dg_cf_simu': p_simu_dg_cf[1],
        'muerr_5778_dg_cf_simu': perr_simu_dg_cf[1],
        'sig_5778_dg_cf_simu': p_simu_dg_cf[2],
        'sigerr_5778_dg_cf_simu': perr_simu_dg_cf[2],
        'ew_5778_dg_cf_simu': ew_5778_dg_cf_simu,
        'ewerr_5778_dg_cf_simu': ew_err_5778_dg_cf_simu,

        'a_5778_dg_mc_simu': -p_simu_dg_mc[0],
        'aerr_5778_dg_mc_simu': perr_simu_dg_mc[0],
        'mu_5778_dg_mc_simu': p_simu_dg_mc[1],
        'muerr_5778_dg_mc_simu': perr_simu_dg_mc[1],
        'sig_5778_dg_mc_simu': p_simu_dg_mc[2],
        'sigerr_5778_dg_mc_simu': perr_simu_dg_mc[2],
        'ew_5778_dg_mc_simu': ew_5778_dg_mc_simu,
        'ewerr_5778_dg_mc_simu': ew_err_5778_dg_mc_simu,
    }

    return dib_dict


def measure_dib6614(wave, flux, flux_err, snr):
    ''' Measure the DIB 6614

    Parameters
    ----------
    wave : array
        The wavelength of the spectrum which has been preprocessed and processed
    flux : array
        The renormalized flux of the spectrum
    flux_err : array
        The error of the flux. Actually, it is the inverse variance = 1 / error^2
    snr: float
        The signal-to-noise ratio of the spectrum in the 'r' waveband

    Returns
    -------
    dib_dict : dict
        The measurement results of the DIB 6614
    '''
    # =========================================================================
    # Convert the data type to float64 for the following calculation like MCMC
    x, y, y_err = np.float64(wave), np.float64(flux), np.float64(flux_err)

    # Pre-detection for the DIB 6614
    is_6614_detect, mu0_6614, a0_6614, eps = pre_detect(x, y, snr, lower_lambda=6608, upper_lambda=6620)

    # =========================================================================
    # Fit the DIB 6614
    if is_6614_detect == 1:
        y_mask = dynamic_mask(x, y, mu0_6614, width=6)  # dynamically determine the range of mask

        # fit the DIB 6614 as a single gaussian profile
        p_6614_cf, perr_6614_cf = get_6614_cf(x, y_mask, mu0_6614, a0_6614)
        p_6614_mc, perr_6614_mc, _ = get_6614_mcmc(x, y_mask, p_6614_cf, y_err)
    else:
        p_6614_cf = -1.0 * np.ones(3)
        p_6614_mc = -1.0 * np.ones(3)
        perr_6614_cf = -1.0 * np.ones(3)
        perr_6614_mc = -1.0 * np.ones(3)

    # =========================================================================
    # Measure the equivalent width of DIB 6614
    if is_6614_detect == 1:
        ew_6614_cf, ew_err_6614_cf = ew_measure(p_6614_cf[0], perr_6614_cf[0], p_6614_cf[2], perr_6614_cf[2])
        ew_6614_mc, ew_err_6614_mc = ew_measure(p_6614_mc[0], perr_6614_mc[0], p_6614_mc[2], perr_6614_mc[2])
    else:
        ew_6614_cf, ew_err_6614_cf = -1.0, -1.0
        ew_6614_mc, ew_err_6614_mc = -1.0, -1.0

    # =========================================================================
    # Assemble the results as a dictionary
    dib_dict = {
        # 'obsid': obsid,
        'threshold': eps,
        'is_6614_detect': is_6614_detect,

        # DIB 6614
        'a0_6614': a0_6614,
        'mu0_6614': mu0_6614,

        'a_6614_cf': -p_6614_cf[0],
        'aerr_6614_cf': perr_6614_cf[0],
        'mu_6614_cf': p_6614_cf[1],
        'muerr_6614_cf': perr_6614_cf[1],
        'sig_6614_cf': p_6614_cf[2],
        'sigerr_6614_cf': perr_6614_cf[2],
        'ew_6614_cf': ew_6614_cf,
        'ewerr_6614_cf': ew_err_6614_cf,

        'a_6614_mc': -p_6614_mc[0],
        'aerr_6614_mc': perr_6614_mc[0],
        'mu_6614_mc': p_6614_mc[1],
        'muerr_6614_mc': perr_6614_mc[1],
        'sig_6614_mc': p_6614_mc[2],
        'sigerr_6614_mc': perr_6614_mc[2],
        'ew_6614_mc': ew_6614_mc,
        'ewerr_6614_mc': ew_err_6614_mc,
    }

    return dib_dict
