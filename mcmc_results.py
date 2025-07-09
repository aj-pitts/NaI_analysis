import numpy as np
from astropy.io import fits
from tqdm import tqdm
import os
import re
from astropy.table import Table
from modules import util, file_handler

def sort_paths(mcmc_paths):
    def extract_run_number(filepath):
        match = re.search(r"-run-(\d+)\.fits$", filepath)
        return int(match.group(1)) if match else float('inf')

    sorted_paths = sorted(mcmc_paths, key=extract_run_number)
    return sorted_paths

def combine_mcmc_results(mcmc_paths, verbose=False):
    sorted_paths = sort_paths(mcmc_paths)

    records = []
    dtype=[
        ('id', int), 
        ('bin', int), 
        ('velocities', float),
        ('lambda samples', object), 
        ('percentiles', object)
    ]

    items = sorted_paths
    iterator = tqdm(sorted_paths, desc="Combining MCMC results") if verbose else items

    for mcmc_fil in iterator:
        data = fits.open(mcmc_fil)
        data_table = Table(data[1].data)

        bins = data_table['bin']
        percentiles = data_table['percentiles']
        velocities = data_table['velocities']
        lambda_samples = np.array([row_samples[:,1000:,0].flatten() for row_samples in data_table['samples']])
        ids = np.arange(len(records), len(data_table)+len(records))

        records.extend(zip(ids, bins, velocities, lambda_samples, percentiles))

    return np.array(records, dtype=dtype)


def make_mcmc_results_cube(galname, bin_method, verbose=False, write_data=True):
    datapath_dict = file_handler.init_datapaths(galname, bin_method)
    mcmc_table = combine_mcmc_results(datapath_dict['MCMC'], verbose=verbose)
    cube = fits.open(datapath_dict['LOGCUBE'])
    if cube['primary'].header['dapqual'] == 30:
        raise ValueError(f"LOGCUBE flagged as CRITICAL in Primary header.")

    spatial_bins = cube['binid'].data[0]

    results_cube = np.zeros((4, spatial_bins.shape[0], spatial_bins.shape[1]))
    error_16th_percentile_cube = np.zeros_like(results_cube)
    error_84th_percentile_cube = np.zeros_like(results_cube)

    bins, inds = np.unique(mcmc_table['bin'],return_index=True)
    zipped_items = zip(bins,inds)
    iterator = tqdm(zipped_items, desc="constructing MCMC results cube") if verbose else zipped_items

    for ID, ind in iterator:
        if ID == -1:
            continue
        w = spatial_bins == ID
        y, x = np.where(w)

        percentiles = mcmc_table[ind]['percentiles']

        for i,cube in enumerate([results_cube, error_16th_percentile_cube, error_84th_percentile_cube]):
            cube[:,y,x] = percentiles[:,i, np.newaxis]
            

    mcmc_dict = {"MCMC Results":results_cube, "MCMC 16th Percentile":error_16th_percentile_cube, "MCMC 84th Percentile":error_84th_percentile_cube}

    HDU_name = "MCMC_RESULTS"
    mcmc_header_dict = {
        HDU_name:{
            "DESC":(f"{galname} {HDU_name.replace("_"," ")} Cube",""),
            "C01":("lambda_0", "Data in channel 1"),
            "C02":("log_N", "Data in channel 2"),
            "C03":("b_D", "Data in channel 3"),
            "C04":("C_f", "Data in channel 4"),
            "HDUCLASS":("CUBE", "Data format"),
            "BUNIT_01":("Angstrom", "Unit of pixel value in C01"),
            "BUNIT_02":("1 / cm^2", "Unit of pixel value in C02"),
            "BUNIT_03":("km / s", "Unit of pixel value in C03"),
            "BUNIT_04":(" ", "Unit of pixel value in C04"),
            "ERRDATA1":(f"MCMC_16TH_PERC", "Associated 16th percentile uncertainty values extension"),
            "ERRDATA2":(f"MCMC_84TH_PERC", "Associated 84th percentile uncertainty values extension"),
            "EXTNAME":(HDU_name, "Extension name"),
            "AUTHOR":("Andrew Pitts","")
        },
        f"MCMC_16TH_PERC":{
            "DESC":(f"{galname} {HDU_name.replace("_"," ")} 16th percentile uncertainty",""),
            "C01":("lambda_0", "Data in channel 1"),
            "C02":("log_N", "Data in channel 2"),
            "C03":("b_D", "Data in channel 3"),
            "C04":("C_f", "Data in channel 4"),
            "HDUCLASS":("CUBE", "Data format"),
            "BUNIT_01":("Angstrom", "Unit of pixel value in C01"),
            "BUNIT_02":("1 / cm^2", "Unit of pixel value in C02"),
            "BUNIT_03":("km / s", "Unit of pixel value in C03"),
            "BUNIT_04":(" ", "Unit of pixel value in C04"),
            "DATA":(HDU_name, "Associated data extension"),
            "ERRDATA2":(f"MCMC_84TH_PERC", "Associated 84th percentile uncertainty values extension"),
            "EXTNAME":(f"MCMC_16TH_PERC", "Extension name"),
            "AUTHOR":("Andrew Pitts","")
        },
        f"MCMC_84TH_PERC":{
            "DESC":(f"{galname} {HDU_name.replace("_"," ")} 84th percentile uncertainty",""),
            "C01":("lambda_0", "Data in channel 1"),
            "C02":("log_N", "Data in channel 2"),
            "C03":("b_D", "Data in channel 3"),
            "C04":("C_f", "Data in channel 4"),
            "HDUCLASS":("CUBE", "Data format"),
            "BUNIT_01":("Angstrom", "Unit of pixel value in C01"),
            "BUNIT_02":("1 / cm^2", "Unit of pixel value in C02"),
            "BUNIT_03":("km / s", "Unit of pixel value in C03"),
            "BUNIT_04":(" ", "Unit of pixel value in C03"),
            "DATA":(HDU_name, "Associated data extension"),
            "ERRDATA1":(f"MCMC_16TH_PERC", "Associated 16th percentile uncertainty values extension"),
            "EXTNAME":(f"MCMC_84TH_PERC", "Extension name"),
            "AUTHOR":("Andrew Pitts","")
        }
    }
    if write_data:
        mcmc_cubedict = file_handler.standard_map_dict(galname, mcmc_dict, custom_header_dict=mcmc_header_dict)
        file_handler.write_maps_file(galname, bin_method, [mcmc_cubedict], verbose=verbose, preserve_standard_order=True)
    else:
        return mcmc_dict, mcmc_header_dict




def mcmc_percentiles(mcmc_paths, include_burnin = False, save = None, verbose = True):
    sorted_paths = sort_paths(mcmc_paths)

    s = 1000
    if include_burnin:
        s=0

    records = []

    dtype = [
        ('id', int),
        ('bin', int),
        ('lambda', object),
        ('logN', object),
        ('bD', object),
        ('Cf', object)
    ]

    items = sorted_paths
    iterator = tqdm(sorted_paths, desc="Extracting MCMC Percentiles") if verbose else items

    for fil in iterator:
        data = fits.open(fil)
        data_table = Table(data[1].data)
        data_table.keep_columns(['bin', 'samples'])

        ids = np.arange(len(records), len(data_table)+len(records))
        bins = data_table['bin']

        
        lambda_samples = [row_samples[:,s:,0].flatten() for row_samples in data_table['samples']]
        logN_samples = [row_samples[:,s:,1].flatten() for row_samples in data_table['samples']]
        bD_samples = [row_samples[:,s:,2].flatten() for row_samples in data_table['samples']]
        Cf_samples = [row_samples[:,s:,3].flatten() for row_samples in data_table['samples']]

        records.extend(zip(ids, bins, lambda_samples, logN_samples, bD_samples, Cf_samples))

    structured_array = np.array(records, dtype=dtype)

    if save:
        np.save(save, structured_array)
        util.verbose_print(verbose, f"Saved MCMC Percentiles to {save}")

    return structured_array