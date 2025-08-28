''' 
This script is used for batch measuring the DIB in cool stellar spectra of LAMOST
'''
import os
import sys
import warnings
from pathlib import Path

import yaml
import time
from datetime import datetime

import h5py
from tqdm import tqdm
import pandas as pd
import numpy as np
from astropy.table import Table

# self-defined modules
sys.path.append(str(Path(__file__).resolve().parent.parent))  # add the parent directory to the path
from dibkit.core import utils

# from dibkit.dib.measure import measure_dib5780
from dibkit.dib.measure import measure_dib5780_lite
from dibkit.dib.measure import measure_dib6614


def get_config_dib_region(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    region_dict = config['dib_region']
    return region_dict


def get_dib_region(config: dict):
    return np.arange(config['xmin'], config['xmax'], config['step'])


# GLOBAL VARIABLES
ROOT = Path(__file__).resolve().parent.parent
# unpack the data config
data_config_path = os.path.join(ROOT, 'configs', 'data.yml')
with open(data_config_path, 'r') as f:
    data_config = yaml.safe_load(f)
    input_catalog_path = data_config['PATH']['hot_catalog_path']
    data_dir = data_config['PATH']['data_dir']
# unpack the dib config
dib_config_path = os.path.join(ROOT, 'configs', 'dib.yml')
dib_region_dict = get_config_dib_region(config_path=dib_config_path)
dib_region = get_dib_region(config=dib_region_dict)
wave_min = dib_region_dict['xmin']
wave_max = dib_region_dict['xmax']


def set_env():
    # close the multithreading for numpy
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'


def input_chunk():
    args = sys.argv[1:]
    assert len(args) == 2, 'Please provide only 2 arguments'
    arg_start = int(args[0])
    arg_end = int(args[1])
    return arg_start, arg_end


def get_file_paths(arg_start: int, arg_end: int, config_path: str = data_config_path) -> tuple:
    # parse the config file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    output_dir = config['PATH']['output_hot_dir']
    output_prefix = config['PATH']['output_prefix']

    log_file = os.path.join(ROOT, output_dir, f'{output_prefix}_{arg_start}_{arg_end}.log')
    csv_file = os.path.join(ROOT, output_dir, f'{output_prefix}_{arg_start}_{arg_end}.csv')
    h5_file = os.path.join(ROOT, output_dir, f'{output_prefix}_{arg_start}_{arg_end}.h5')

    return log_file, csv_file, h5_file


def remove_files(*file_paths):
    for file in file_paths:
        if os.path.exists(file):
            os.remove(file)


def main(dib_name: str = '5780', is_debug: bool = False, is_save_h5: bool = True):
    # -------------------------------------------------------------------------
    # Set environment
    set_env()
    
    # -------------------------------------------------------------------------
    # Input the start and end index of the chunk
    if is_debug:
        arg_start, arg_end = 39075, 39077
    else:
        arg_start, arg_end = input_chunk()
        
    # -------------------------------------------------------------------------
    # Initialize
    # input catalog
    df = Table.read(os.path.join(ROOT, input_catalog_path)).to_pandas()
    df_sub = df.iloc[arg_start:arg_end]
    num_loop = df_sub.shape[0]
    result_list = []
    
    # check whether files exist
    log_file, csv_file, h5_file = get_file_paths(arg_start, arg_end)
    remove_files(log_file, csv_file, h5_file)  # remove the files if they exist    
    
    # TODO: create hdf5 file
    if is_save_h5:
        pass
    
    # -------------------------------------------------------------------------
    # Start
    time_start = time.time()
    time_begin = datetime.now()
    print(f'begin at: {time_begin.strftime("%Y-%m-%d %H:%M:%S")}')
    
    # -------------------------------------------------------------------------
    # Loop
    for i in tqdm(range(num_loop)):
        spec_info = df_sub.iloc[i]
        try:
            # 1. read the spectrum from .fits file
            abs_spec_path = utils.get_spec_path(spec_info, os.path.join(ROOT, data_dir))
            wave_vac, flux, flux_norm, ivar = utils.read_fits(abs_spec_path)
            # transform from vacuum to air
            wave_air = utils.vac2air(wave_vac)
            
            # 2. local continuum normalization
            # cut
            wave_cut, flux_cut, ivar_cut = utils.cut_array(wave_air, flux, ivar,
                                                           xmin=wave_min,
                                                           xmax=wave_max)
            # rebin
            flux_rebin = utils.rebin_array(wave_cut, flux_cut, dib_region)
            ivar_rebin = utils.rebin_array(wave_cut, ivar_cut, dib_region)
            # normalize
            flux_cont, flux_norm = utils.specNormg(flux_rebin, order=5)
            # flux error
            ivar_norm = flux_cont ** 2 * ivar_rebin
            err_flux_norm = 1 / np.sqrt(ivar_norm)
            
            # 3. measure DIB
            snr = spec_info['snrr']
            if dib_name == '5780':
                result = measure_dib5780_lite(dib_region, flux_norm, err_flux_norm, snr)
            elif dib_name == '6614':
                result = measure_dib6614(dib_region, flux_norm, err_flux_norm, snr)
            
            # 4. save the result
            cis_dict = {'obsid': spec_info['obsid']}
            cis_dict.update(result)
            result_list.append(cis_dict)
            
            # TODO: 5. store the continuum-normalized spectrum and its error
            if is_save_h5:
                pass
        
        except Exception as e:
            print(f'error: {spec_info["obsid"]}')
            print(e)
            # save log file
            with open(log_file, 'a') as f:
                f.write(f'error: {spec_info["obsid"]}\n{e}\n\n')
            continue
    
    # -------------------------------------------------------------------------
    # save
    if is_save_h5:
        pass
    df_result = pd.DataFrame(result_list).round(6)
    df_result.to_csv(csv_file, index=False)

    # -------------------------------------------------------------------------
    # Finish
    time_end = time.time()
    time_finish = datetime.now()
    print(f'finish at: {time_finish.strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'totally cost: {time_end - time_start:.2f}s')        


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    # main()
    main(dib_name='5780', is_debug=True, is_save_h5=False)
