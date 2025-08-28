import os
import yaml
import numpy as np
import pandas as pd
from astropy.io import fits
from pathlib import Path

from dibkit.core.utils import assemble_path, shift_wave, vac2air, cut_array, rebin_array, specNormg


def get_spec_path(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    data_dir = config['PATH']['data_dir']
    return os.path.join(ROOT, data_dir)


def get_config_dib_region(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    region_dict = config['dib_region']
    return region_dict


def get_dib_region(config: dict):
    return np.arange(config['xmin'], config['xmax'], config['step'])


def get_config_similar_region(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    region_dict = config['similar_region']
    return region_dict


def get_similar_region(config: dict):
    return np.arange(config['xmin'], config['xmax'], config['step'])


def get_config_neighbor(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    neighbor_dict = config['neighbor']
    return neighbor_dict


# GLOBAL VARIABLES
ROOT = Path(__file__).resolve().parent.parent.parent
data_config_path = os.path.join(ROOT, 'configs', 'data.yml')
dib_config_path = os.path.join(ROOT, 'configs', 'dib.yml')
neighbor_config_path = os.path.join(ROOT, 'configs', 'neighbor.yml')

# unpack the data config
spec_relative_path = get_spec_path(config_path=data_config_path)
# unpack the similar region config
similar_region_dict = get_config_similar_region(config_path=dib_config_path)
similar_region = get_similar_region(config=similar_region_dict)
# unpack the neighbor config
neighbor_config = get_config_neighbor(config_path=neighbor_config_path)
lower_bound = neighbor_config['num_lo']
upper_bound = neighbor_config['num_hi']
ratio_take = neighbor_config['num_ratio']


# index of the mask region
def _load_id_mask(config_path: str = dib_config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    mask_region_list = config['mask_region_list']
    wave_rebin = similar_region
    id_mask = np.zeros_like(wave_rebin, dtype=bool)
    for i in mask_region_list:
        id_mask = id_mask | ((wave_rebin >= i[0]) & (wave_rebin < i[1]))
    return id_mask


_id_mask = _load_id_mask()


class Spectra(object):

    def __init__(self, spec_info: pd.Series, data_dir: str = spec_relative_path):
        self.__spec_info = spec_info
        self.__data_dir = data_dir

    @property
    def spec_path(self):
        assemble_list = [self.__spec_info['lmjd'], self.__spec_info['mjd'], self.__spec_info['planid'].decode(),
                         self.__spec_info['spid'], self.__spec_info['fiberid']]
        relative_path = assemble_path(*assemble_list)
        absolute_path = os.path.join(self.__data_dir, relative_path)
        return absolute_path

    def read_fits(self):
        with fits.open(self.spec_path) as f:
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

    def get_similar_preprocess_spec(self, wave_rebin: np.ndarray = similar_region):
        # Read fits file
        wave, _, flux_norm, ivar = self.read_fits()
        # Shift the wavelength to the rest frame
        wave_rest = shift_wave(wave, self.__spec_info['rv'])
        # Convert the wavelength from vacuum to air
        wave_rest_air = vac2air(wave_rest)
        # Cut the spectrum to the range of 4500-7500 A
        wave_cut, flux_cut, ivar_cut = cut_array(wave_rest_air, flux_norm, ivar,
                                                 xmin=similar_region_dict['xmin'],
                                                 xmax=similar_region_dict['xmax'])
        # Rebin for the same pixel size
        flux_rebin = rebin_array(wave_cut, flux_cut, x_rebin=wave_rebin)
        # Mask the regions which are not used for the computation of the similarity
        flux_rebin_copy = flux_rebin.copy()
        flux_mask = flux_rebin_copy[~_id_mask]
        return flux_mask  # return any intermediate result, like `flux_rebin`, `flux_cut`, etc. if needed

    def get_measure_preprocess_spec(self, region_dict: dict):
        # Read fits file
        wave, flux, _, ivar = self.read_fits()
        # Shift the wavelength to the rest frame
        wave_rest = shift_wave(wave, self.__spec_info['rv'])
        # Convert the wavelength from vacuum to air
        wave_rest_air = vac2air(wave_rest)
        # Cut the spectrum to the range
        wave_cut, flux_cut, ivar_cut = cut_array(wave_rest_air, flux, ivar,
                                                 xmin=region_dict['xmin'],
                                                 xmax=region_dict['xmax'])
        # Rebin for the same pixel size
        dib_region = np.arange(region_dict['xmin'], region_dict['xmax'], region_dict['step'])
        flux_rebin = rebin_array(wave_cut, flux_cut, x_rebin=dib_region)
        ivar_rebin = rebin_array(wave_cut, ivar_cut, x_rebin=dib_region)
        # Local renormalization
        flux_cont, flux_norm = specNormg(flux_rebin)

        # Compute the flux error
        ivar_norm = flux_cont ** 2 * ivar_rebin     # TODO: reference
        err_flux_norm = 1 / np.sqrt(ivar_norm)

        return flux_norm, err_flux_norm
    
    def get_measure_preprocess_spec2(self, region_dict: dict):
        # Read fits file
        wave, flux, flux_norm_lasp, ivar = self.read_fits()
        # Shift the wavelength to the rest frame
        wave_rest = shift_wave(wave, self.__spec_info['rv'])
        # Convert the wavelength from vacuum to air
        wave_rest_air = vac2air(wave_rest)
        # Cut the spectrum to the range
        wave_cut, flux_cut, ivar_cut = cut_array(wave_rest_air, flux, ivar,
                                                xmin=region_dict['xmin'],
                                                xmax=region_dict['xmax'])
        _, flux_norm_cut_lasp, _ = cut_array(wave_rest_air, flux_norm_lasp, ivar,
                                        xmin=region_dict['xmin'],
                                        xmax=region_dict['xmax'])
        # Rebin for the same pixel size
        dib_region = np.arange(region_dict['xmin'], region_dict['xmax'], region_dict['step'])
        flux_rebin = rebin_array(wave_cut, flux_cut, x_rebin=dib_region)
        ivar_rebin = rebin_array(wave_cut, ivar_cut, x_rebin=dib_region)
        flux_norm_cut_lasp = rebin_array(wave_cut, flux_norm_cut_lasp, x_rebin=dib_region)
        # Local renormalization
        flux_cont, flux_norm = specNormg(flux_rebin)

        # Compute the flux error
        ivar_norm = flux_cont ** 2 * ivar_rebin     # TODO: reference
        err_flux_norm = 1 / np.sqrt(ivar_norm)

        return dib_region, flux_rebin, flux_norm, flux_norm_cut_lasp
