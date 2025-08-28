import numpy as np
import pandas as pd
import yaml
import os
from pathlib import Path

from dibkit.core.utils import compute_distance
from dibkit.core.spec import Spectra
from dibkit.core.sample import load_catalog


# PARSE CONFIG FILES
def get_config_threshold(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    threshold = config['threshold']
    delta_teff = threshold['delta_teff']
    delta_logg = threshold['delta_logg']
    delta_feh = threshold['delta_feh']
    return delta_teff, delta_logg, delta_feh


def get_config_neighbor(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    neighbor_dict = config['neighbor']
    return neighbor_dict


# GLOBAL VARIABLES
ROOT = Path(__file__).resolve().parent.parent.parent
neighbor_config_path = os.path.join(ROOT, 'configs', 'neighbor.yml')
# unpack the threshold of stellar parameters
threshold_tuple = get_config_threshold(config_path=neighbor_config_path)
# unpack the neighbor config
neighbor_config = get_config_neighbor(config_path=neighbor_config_path)
lower_bound = neighbor_config['num_lo']
upper_bound = neighbor_config['num_hi']
ratio_take = neighbor_config['num_ratio']


class Neighbor(object):
    """
    This class is used to find the nearest neighbors of a given target
    """

    def __init__(self, reference_dataset: pd.DataFrame, threshold: tuple = threshold_tuple):
        """
        # TODO
        """
        # assume the keys of the reference dataset are 'teff', 'logg', 'feh'
        # protected attributes

        self._target_info = pd.Series()
        self._reference_dataset = reference_dataset
        self._threshold = threshold
        # unpack the reference stellar parameter space & threshold
        self._param_space: np.ndarray = (
            self._reference_dataset[['teff', 'logg', 'feh']].values)  # modify the keys as needed
        self._delta_teff = threshold[0]
        self._delta_logg = threshold[1]
        self._delta_feh = threshold[2]

        # private attributes
        self.__n_match: int = 0
        self.__idx_match: np.ndarray = np.array([])

        # public attributes
        self.closest_neighbor_info = pd.DataFrame()
        self.candidate_info = pd.DataFrame()
        self.neighbor_info = pd.DataFrame()

    def set_target(self, target_series: pd.Series):
        self._target_info = target_series
        self._match_in_param_space()
        if self.__n_match >= lower_bound:
            self.neighbor_info = self.get_neighbor_info()
            self.candidate_info = self.get_candidate_info()
            self.closest_neighbor_info = self.get_closest_neighbor_info()

    @property
    def target_info(self):
        return self._target_info

    def _match_in_param_space(self):
        """
        # TODO:
        """
        # unpack the target information
        teff = self._target_info['teff']
        logg = self._target_info['logg']
        feh = self._target_info['feh']

        idx_teff = (self._param_space[:, 0] >= teff - self._delta_teff) & \
                   (self._param_space[:, 0] <= teff + self._delta_teff)
        idx_logg = (self._param_space[:, 1] >= logg - self._delta_logg) & \
                   (self._param_space[:, 1] <= logg + self._delta_logg)
        idx_feh = (self._param_space[:, 2] >= feh - self._delta_feh) & \
                  (self._param_space[:, 2] <= feh + self._delta_feh)

        idx = idx_teff & idx_logg & idx_feh
        n_match = np.count_nonzero(idx)     # or `idx.sum()`

        self.__n_match = n_match
        self.__idx_match = idx

    @property
    def n_match(self) -> int:
        return self.__n_match

    def get_neighbor_info(self) -> pd.DataFrame:
        return self._reference_dataset.iloc[self.__idx_match]

    def get_candidate_info(self,
                           low: int = lower_bound,
                           high: int = upper_bound) -> pd.DataFrame:
        if self.__n_match < low:
            df_candidate = pd.DataFrame()
        elif self.__n_match > high:
            df_candidate = self.neighbor_info.sort_values(by='snrr', ascending=False).head(high)
        else:
            df_candidate = self.neighbor_info.sort_values(by='snrr', ascending=False)
        return df_candidate

    def get_closest_neighbor_info(self, ratio: float = ratio_take) -> pd.DataFrame:
        d_list = []  # store the distance between the target and each candidate neighbor
        num_candidate = self.candidate_info.shape[0]
        flux_target = Spectra(self._target_info).get_similar_preprocess_spec()

        for i in range(num_candidate):
            template_info: pd.Series = self.candidate_info.iloc[i]
            try:
                # ! The size of the normalized flux of some lamost pipelined spectra
                # ! are NOT the same as the wavelength array
                # ! I cannot understand this is why.
                flux_template = Spectra(template_info).get_similar_preprocess_spec()
                d = compute_distance(flux_target, flux_template)
            except Exception:
                d = np.inf
            d_list.append(d)

        df_tmp = self.candidate_info
        df_tmp['distance'] = d_list
        df_closest = df_tmp.sort_values(by='distance', ascending=True).head(int(num_candidate * ratio))
        return df_closest


if __name__ == '__main__':

    df_reference, df_target = load_catalog()
    target_info = df_target.iloc[6]
    neighbor = Neighbor(df_reference)
    neighbor.set_target(target_info)
    print(neighbor.n_match)
    print(neighbor.closest_neighbor_info.head(10)['distance'])
