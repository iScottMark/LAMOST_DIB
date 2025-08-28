''' 
This script is used for batch measuring the DIB in the cool stellar spectra of LAMOST
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

# self-defined modules
sys.path.append(str(Path(__file__).resolve().parent.parent))  # add the parent directory to the path

from dibkit.core.sample import load_catalog
from dibkit.core.neighbor import Neighbor
from dibkit.core.cis import CIS

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


def get_config_neighbor(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    neighbor_dict = config['neighbor']
    return neighbor_dict


# GLOBAL VARIABLES
ROOT = str(Path(__file__).resolve().parent.parent)
data_config_path = os.path.join(ROOT, 'configs', 'data.yml')
dib_config_path = os.path.join(ROOT, 'configs', 'dib.yml')
neighbor_config_path = os.path.join(ROOT, 'configs', 'neighbor.yml')

# unpack the dib config
dib_region_dict = get_config_dib_region(config_path=dib_config_path)
dib_region = get_dib_region(config=dib_region_dict)
# unpack the neighbor config
neighbor_config = get_config_neighbor(config_path=neighbor_config_path)
lower_bound = neighbor_config['num_lo']
upper_bound = neighbor_config['num_hi']
ratio_take = neighbor_config['num_ratio']


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
    output_dir = config['PATH']['output_cool_dir']
    output_prefix = config['PATH']['output_prefix']

    log_file = os.path.join(ROOT, output_dir, f'{output_prefix}_{arg_start}_{arg_end}.log')
    csv_file = os.path.join(ROOT, output_dir, f'{output_prefix}_{arg_start}_{arg_end}.csv')
    h5_file = os.path.join(ROOT, output_dir, f'{output_prefix}_{arg_start}_{arg_end}.h5')

    return log_file, csv_file, h5_file


def remove_files(*file_paths):
    for file in file_paths:
        if os.path.exists(file):
            os.remove(file)


def create_hdf5(h5_file, num_loops, len_flux, len_neighbor):
    hf = h5py.File(h5_file, 'a')
    # store the stacked template spectra
    hf.create_dataset('obsid', shape=(0,), dtype='i', maxshape=(num_loops,))
    hf.create_dataset('flux_target', shape=(0, len_flux), dtype='f', maxshape=(num_loops, len_flux))
    hf.create_dataset('flux_template', shape=(0, len_flux), dtype='f', maxshape=(num_loops, len_flux))
    hf.create_dataset('err_flux_target', shape=(0, len_flux), dtype='f', maxshape=(num_loops, len_flux))
    hf.create_dataset('err_flux_template', shape=(0, len_flux), dtype='f', maxshape=(num_loops, len_flux))
    # store the OBSID and distance of the template which is used to construct a template spectrum
    hf.create_dataset('obsid_each_template', shape=(0, len_neighbor), dtype='i', maxshape=(num_loops, len_neighbor))
    hf.create_dataset('distance_each_template', shape=(0, len_neighbor), dtype='f',
                      maxshape=(num_loops, len_neighbor))
    return hf


def write_hdf5(hdf5, len_flux, len_neighbor, obsid, flux_target, flux_template, err_flux_target, err_flux_template,
               obsid_each_template, distance_each_template):
    hdf5['obsid'].resize((hdf5['obsid'].shape[0] + 1,))
    hdf5['flux_target'].resize((hdf5['flux_target'].shape[0] + 1, len_flux))
    hdf5['flux_template'].resize((hdf5['flux_template'].shape[0] + 1, len_flux))
    hdf5['err_flux_target'].resize((hdf5['err_flux_target'].shape[0] + 1, len_flux))
    hdf5['err_flux_template'].resize((hdf5['err_flux_template'].shape[0] + 1, len_flux))

    hdf5['obsid'][-1] = obsid
    hdf5['flux_target'][-1] = flux_target
    hdf5['flux_template'][-1] = flux_template
    hdf5['err_flux_target'][-1] = err_flux_target
    hdf5['err_flux_template'][-1] = err_flux_template

    hdf5['obsid_each_template'].resize((hdf5['obsid_each_template'].shape[0] + 1, len_neighbor))
    hdf5['distance_each_template'].resize((hdf5['distance_each_template'].shape[0] + 1, len_neighbor))

    if len(obsid_each_template) < len_neighbor:
        # pad the array with -1 to make the length of the array equal to len_neighbor
        obsid_each_template = np.pad(obsid_each_template, (0, len_neighbor - len(obsid_each_template)), 'constant',
                                     constant_values=(0, int(-1)))
        distance_each_template = np.pad(distance_each_template, (0, len_neighbor - len(distance_each_template)),
                                        'constant', constant_values=(0, int(-1)))

    hdf5['obsid_each_template'][-1] = obsid_each_template
    hdf5['distance_each_template'][-1] = distance_each_template

    return hdf5


def main(dib_name: str = '5780', is_debug: bool = False, is_save_h5: bool = True):
    # -------------------------------------------------------------------------
    # Set environment
    set_env()

    # -------------------------------------------------------------------------
    # Input the start and end index of the chunk
    if is_debug:
        # arg_start, arg_end = 30, 32
        arg_start, arg_end = 909198, 909200
    else:
        arg_start, arg_end = input_chunk()

    # -------------------------------------------------------------------------
    # Build target catalog and reference catalogs, which needs a lot of memory and a few minutes
    df_reference, df_target = load_catalog()

    # -------------------------------------------------------------------------
    # Initialize
    df_target_sub = df_target.iloc[arg_start:arg_end]
    num_loops = df_target_sub.shape[0]
    result_list = []

    # check whether files exist
    log_file, csv_file, h5_file = get_file_paths(arg_start, arg_end)
    remove_files(log_file, csv_file, h5_file)  # remove the files if they exist

    # create hdf5 file
    if is_save_h5:
        hf = create_hdf5(h5_file, num_loops,
                         len_flux=len(dib_region),
                         len_neighbor=int(ratio_take * upper_bound))

    # -------------------------------------------------------------------------
    # Start
    time_start = time.time()
    time_begin = datetime.now()
    print(f'begin at: {time_begin.strftime("%Y-%m-%d %H:%M:%S")}')

    # -------------------------------------------------------------------------
    # Loop
    # Instantiate the Neighbor class
    neighbor = Neighbor(df_reference)
    for i in tqdm(range(num_loops)):
        target_info = df_target_sub.iloc[i]
        try:
            neighbor.set_target(target_info)
            n_match = neighbor.n_match
            if n_match < lower_bound:
                continue
            # Match neighbors in the parameter space
            df_closest_neighbor = neighbor.closest_neighbor_info
            cis = CIS(dib_region=dib_region_dict,
                      target_info=target_info,
                      closest_neighbor_info=df_closest_neighbor)

            flux_cis, err_flux_cis = cis.flux_cis, cis.err_flux_cis
            # TODO: directly read the data from the hdf5 file, which has saved the processed data
            # if not is_save_h5:
            #     flux_cis, err_flux_cis = cis.flux_cis, cis.err_flux_cis
            # else:
            #     pass

            # TODO: `flux_cis` is different from the one derived from the former method
            # Measure the DIB in Cool ISM residual Spectrum (CIS)
            if dib_name == '5780':
                result = measure_dib5780_lite(dib_region, flux_cis, err_flux_cis, target_info['snrr'])
            elif dib_name == '6614':
                result = measure_dib6614(dib_region, flux_cis, err_flux_cis, target_info['snrr'])
            else:
                raise ValueError(f'At present, `dib_name` only supports either "5780" or "6614", \
                    but got {dib_name}')

            # Store the final results
            cis_dict = {'obsid': target_info['obsid'], 'n_match': n_match}
            cis_dict.update(result)
            result_list.append(cis_dict)

            # Store the stacked template spectra
            if is_save_h5:
                flux_target, err_flux_target = cis.flux_target, cis.err_flux_target
                flux_template, err_flux_template = cis.flux_template, cis.err_flux_template
                hf = write_hdf5(hdf5=hf,
                                len_flux=len(dib_region),
                                len_neighbor=int(ratio_take * upper_bound),
                                obsid=target_info['obsid'],
                                flux_target=flux_target, flux_template=flux_template,
                                err_flux_target=err_flux_target, err_flux_template=err_flux_template,
                                obsid_each_template=df_closest_neighbor['obsid'].values,
                                distance_each_template=df_closest_neighbor['distance'].values)

        except Exception as e:
            print(f'error: {target_info["obsid"]}')
            print(e)
            # save log file
            with open(log_file, 'a') as f:
                f.write(f'error: {target_info["obsid"]}\n{e}\n\n')
            continue

    # -------------------------------------------------------------------------
    # Save
    if is_save_h5:
        hf.close()
    df = pd.DataFrame(result_list).round(6)
    df.to_csv(csv_file, index=False)

    # -------------------------------------------------------------------------
    # Finish
    time_end = time.time()
    time_finish = datetime.now()
    print(f'finish at: {time_finish.strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'totally cost: {time_end - time_start:.2f}s')


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    # main()
    main(dib_name='5780', is_debug=True, is_save_h5=True)
