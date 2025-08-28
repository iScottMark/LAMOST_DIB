import os
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
from astropy.table import Table


# GLOBAL VARIABLES
ROOT = Path(__file__).resolve().parent.parent.parent
data_config_path = os.path.join(ROOT, 'configs', 'data.yml')


class Catalog(object):
    """
    Class to handle a catalog.

    Attributes
    ----------
    # TODO
    # TODO: code the test cases
    """

    def __init__(self, path: str) -> None:
        """
        Initialize a catalog object.

        Parameters
        ----------
        path : str
            Path to the catalog.
        """
        # Private attributes and methods
        self.__path = path
        print(f'Loading catalog from {self.__path}...')
        self._catalog = self.__load_catalog()
        print('Catalog loaded successfully.')
        # Public attributes
        self.reference_catalog: pd.DataFrame = pd.DataFrame()
        self.target_catalog: pd.DataFrame = pd.DataFrame()

    def __load_catalog(self) -> pd.DataFrame:
        """
        Load catalog from a fits file.

        Returns
        -------
        catalog : pd.DataFrame
            General catalog as a pandas DataFrame.
        """
        return Table.read(self.__path, format='fits').to_pandas()

    @property
    def catalog(self) -> pd.DataFrame:
        """
        Return the catalog.

        Returns
        -------
        catalog : pd.DataFrame
            General catalog as a pandas DataFrame.
        """
        return self._catalog

    def get_reference_catalog(self, filter_idx: pd.Series) -> pd.DataFrame:
        """
        Get the reference catalog.

        Parameters
        ----------
        filter_idx : pd.Series
            Boolean series to filter the catalog, e.g., (df['A'] > 10) & (df['B'] < 20).

        Returns
        -------
        reference_catalog : pd.DataFrame
            Reference catalog as a pandas DataFrame.
        """
        self.reference_catalog = self._catalog[filter_idx]
        return self.reference_catalog

    def get_target_catalog(self, filter_idx: pd.Series) -> pd.DataFrame:
        """
        Get the target catalog.

        Parameters
        ----------
        filter_idx : pd.Series
            Boolean series to filter the catalog, e.g., (df['A'] > 10) & (df['B'] < 20).

        Returns
        -------
        target_catalog : pd.DataFrame
            Target catalog as a pandas DataFrame.
        """
        self.target_catalog = self._catalog[filter_idx]
        return self.target_catalog

    def get_spec_info(self, lamost_obsid: int) -> pd.Series:
        """
        Get the spectrum information of a given observation ID.

        Parameters
        ----------
        lamost_obsid : int
            Unique observation ID in LAMOST.

        Returns
        -------
        spec_info : pd.Series
            Spectrum information of the observation and measurement.
        """
        return self._catalog[self._catalog['obsid'] == lamost_obsid].iloc[0]

    @property
    def reference_parameter_space(self) -> np.ndarray:
        """
        Stellar parameter space of the reference catalog.

        Returns
        -------
        reference_stellar_parameter_space : np.ndarray
            Stellar parameter space (Teff, logg, [Fe/H]) of the reference catalog as a numpy array.
                1st column: Teff,
                2nd column: logg,
                3rd column: [Fe/H].
        """
        return self.reference_catalog[['teff', 'logg', 'feh']].values   # the keyword should be changed as needed


def load_catalog(config_path=data_config_path):
    # parse the config file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    catalog_path: str = config['PATH']['cool_catalog_path']

    lamost_general_catalog = Catalog(path=os.path.join(ROOT, catalog_path))
    df = lamost_general_catalog.catalog
    print(df.shape)

    # reference and target catalog
    # common filter conditions
    teff_err_limit: float = config['FILTER']['common']['teff_err_limit']
    logg_err_limit: float = config['FILTER']['common']['logg_err_limit']
    feh_err_limit: float = config['FILTER']['common']['feh_err_limit']
    rv_err_limit: float = config['FILTER']['common']['rv_err_limit']
    filter_common = ((df['teff_err'] < teff_err_limit) & (df['teff_err'] > 0.0)) & \
                    ((df['logg_err'] < logg_err_limit) & (df['logg_err'] > 0.0)) & \
                    ((df['feh_err'] < feh_err_limit) & (df['feh_err'] > 0.0)) & \
                    ((df['rv_err'] < rv_err_limit) & (df['rv_err'] > 0.0))
    # reference filter conditions
    snrr_lo: float = config['FILTER']['reference']['snrr_lo']
    ebv_up: float = config['FILTER']['reference']['ebv_up']
    b_lo: float = config['FILTER']['reference']['b_lo']
    filter_reference = (df['snrr'] > snrr_lo) & (df['ebv'] < ebv_up) & ((df['b'] > b_lo) | (df['b'] < -b_lo))
    # target filter conditions
    snrr_lo: float = config['FILTER']['target']['snrr_lo']
    filter_target = (df['snrr'] > snrr_lo)

    reference_catalog = lamost_general_catalog.get_reference_catalog(filter_common & filter_reference)
    target_catalog = lamost_general_catalog.get_target_catalog(filter_common & filter_target)

    return reference_catalog, target_catalog


if __name__ == '__main__':
    # load the dataset
    df_reference, df_target = load_dataset()

    # print the results
    print(df_reference.shape)
    print(df_target.shape)
