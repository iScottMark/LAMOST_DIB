import numpy as np
import pandas as pd
from uncertainties import unumpy

from dibkit.core.utils import get_weighted_avg
from dibkit.core.spec import Spectra


class CIS(object):

    def __init__(self, dib_region: dict, target_info: pd.Series, closest_neighbor_info: pd.DataFrame):
        self.__dib_region = dib_region
        self.__target_info = target_info
        self.__closest_neighbor_info = closest_neighbor_info

        self.flux_template, self.err_flux_template = self.get_flux_template()
        self.flux_target, self.err_flux_target = self.get_flux_target()

        u = unumpy.uarray(self.flux_target, self.err_flux_target)
        v = unumpy.uarray(self.flux_template, self.err_flux_template)
        z = u / v
        self.flux_cis = unumpy.nominal_values(z)
        self.err_flux_cis = unumpy.std_devs(z)

    def get_flux_template_arr(self):
        flux_template_arr = []
        err_flux_template_arr = []
        snr_template_arr = []
        for i in range(len(self.__closest_neighbor_info)):
            try:
                template_info = self.__closest_neighbor_info.iloc[i]
                flux_template, err_flux_template = Spectra(template_info).get_measure_preprocess_spec(self.__dib_region)

                flux_template_arr.append(flux_template)
                err_flux_template_arr.append(err_flux_template)
                snr_template_arr.append(template_info['snrr'])
            except FileNotFoundError:
                continue
        flux_template_arr = np.array(flux_template_arr)
        err_flux_template_arr = np.array(err_flux_template_arr)
        snr_template_arr = np.array(snr_template_arr)
        return flux_template_arr, err_flux_template_arr, snr_template_arr

    def get_flux_template(self):
        flux_template_arr, err_flux_template_arr, snr_template_arr = self.get_flux_template_arr()
        flux_template_weighted = get_weighted_avg(flux_template_arr, snr_template_arr)
        err_flux_template_weighted = get_weighted_avg(err_flux_template_arr, snr_template_arr)
        return flux_template_weighted, err_flux_template_weighted

    def get_flux_target(self):
        flux_target, err_flux_target = Spectra(self.__target_info).get_measure_preprocess_spec(self.__dib_region)
        return flux_target, err_flux_target
