import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table, join, Column
import pandas as pd
from glob import glob
import os
import re
from datetime import datetime
import argparse
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from modules import defaults, file_handler, util, plotter


def get_ew_cut(snr):
    if snr<= 30:
        return np.inf
    elif snr <= 38:
        return 0.5
    elif snr <= 64:
        return 0.275
    else:
        return 0.05

def NaD_snr_map(cube_fil, maps_fil, z_guess, verbose=False):
    cube = fits.open(cube_fil)
    maps = fits.open(maps_fil)
    if cube['primary'].header['dapqual'] == 30:
        raise ValueError(f"LOGCUBE flagged as CRITICAL in Primary header.")
    

    spatial_bins = cube['BINID'].data[0]
    fluxcube = cube['FLUX'].data
    ivarcube = cube['IVAR'].data
    wave = cube['WAVE'].data

    stellarvel = maps['STELLAR_VEL'].data
    stellarvel_mask = maps['STELLAR_VEL_MASK'].data #DAPPIXMASK
    stellarvel_ivar = maps['STELLAR_VEL_IVAR'].data


    z = np.zeros_like(stellarvel)
    c = 2.998e5
    for ID in np.unique(spatial_bins):
        w = ID == spatial_bins
        sv = stellarvel[w][0]
        z[w] = (sv * (1+z_guess))/c + z_guess

    snr_map = np.zeros_like(stellarvel)
    #region = (5880, 5910)
    windows = [(5865, 5875), (5915, 5925)]

    for ID in np.unique(spatial_bins):
        w = ID == spatial_bins
        y_inds, x_inds = np.where(w)
        
        z_bin = z[w][0]
        wave_bin = wave / (1+z_bin)

        #wave_window = (wave_bin>=region[0]) & (wave_bin<=region[1])
        wave_window = (wave_bin>=windows[0][0]) & (wave_bin<=windows[0][1]) | (wave_bin>=windows[1][0]) & (wave_bin<=windows[1][1])
        wave_inds = np.where(wave_window)[0]
        
        flux_arr = fluxcube[wave_inds, y_inds[0], x_inds[0]]
        ivar_arr = ivarcube[wave_inds, y_inds[0], x_inds[0]]
        sigma_arr = 1/np.sqrt(ivar_arr)


        snr_map[w] = np.median(flux_arr / sigma_arr)

    return snr_map



def combine_mcmc_results(mcmc_paths, verbose=False):
    def extract_run_number(filepath):
        match = re.search(r"-run-(\d+)\.fits$", filepath)
        return int(match.group(1)) if match else float('inf')
    
    sorted_paths = sorted(mcmc_paths, key=extract_run_number)

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


def make_mcmc_results_cube(cube_fil, mcmc_table, verbose=False):
    cube = fits.open(cube_fil)
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
            "DESC":(f"{args.galname} {HDU_name.replace("_"," ")} Cube",""),
            "C01":("lambda_0", "Data in channel 1"),
            "C02":("log_N", "Data in channel 2"),
            "C03":("b_D", "Data in channel 3"),
            "C04":("C_f", "Data in channel 4"),
            "HDUCLASS":("CUBE", "Data format"),
            "BUNIT_01":("Angstrom", "Unit of pixel value in C01"),
            "BUNIT_02":("1 / cm^2", "Unit of pixel value in C01"),
            "BUNIT_03":("km / s", "Unit of pixel value in C01"),
            "BUNIT_04":(" ", "Unit of pixel value in C01"),
            "ERRDATA1":(f"{HDU_name}_16TH_PERC", "Associated 16th percentile uncertainty values extension"),
            "ERRDATA2":(f"{HDU_name}_84TH_PERC", "Associated 84th percentile uncertainty values extension"),
            "EXTNAME":(HDU_name, "Extension name"),
            "AUTHOR":("Andrew Pitts","")
        },
        f"{HDU_name}_16TH_PERC":{
            "DESC":(f"{args.galname} {HDU_name.replace("_"," ")} 16th percentile uncertainty",""),
            "C01":("lambda_0", "Data in channel 1"),
            "C02":("log_N", "Data in channel 2"),
            "C03":("b_D", "Data in channel 3"),
            "C04":("C_f", "Data in channel 4"),
            "HDUCLASS":("CUBE", "Data format"),
            "BUNIT_01":("Angstrom", "Unit of pixel value in C01"),
            "BUNIT_02":("1 / cm^2", "Unit of pixel value in C01"),
            "BUNIT_03":("km / s", "Unit of pixel value in C01"),
            "BUNIT_04":(" ", "Unit of pixel value in C01"),
            "DATA":(HDU_name, "Associated data extension"),
            "ERRDATA2":(f"{HDU_name}_84TH_PERC", "Associated 84th percentile uncertainty values extension"),
            "EXTNAME":(f"{HDU_name}_16TH_PERC", "Extension name"),
            "AUTHOR":("Andrew Pitts","")
        },
        f"{HDU_name}_84TH_PERC":{
            "DESC":(f"{args.galname} {HDU_name.replace("_"," ")} 84th percentile uncertainty",""),
            "C01":("lambda_0", "Data in channel 1"),
            "C02":("log_N", "Data in channel 2"),
            "C03":("b_D", "Data in channel 3"),
            "C04":("C_f", "Data in channel 4"),
            "HDUCLASS":("CUBE", "Data format"),
            "BUNIT_01":("Angstrom", "Unit of pixel value in C01"),
            "BUNIT_02":("1 / cm^2", "Unit of pixel value in C01"),
            "BUNIT_03":("km / s", "Unit of pixel value in C01"),
            "BUNIT_04":(" ", "Unit of pixel value in C01"),
            "DATA":(HDU_name, "Associated data extension"),
            "ERRDATA1":(f"{HDU_name}_16TH_PERC", "Associated 16th percentile uncertainty values extension"),
            "EXTNAME":(f"{HDU_name}_84TH_PERC", "Extension name"),
            "AUTHOR":("Andrew Pitts","")
        }
    }
    return mcmc_dict, mcmc_header_dict


def make_vmap(cube_fil, maps_fil, snrmap, EW_file, mcmc_table, verbose=False):
    cube = fits.open(cube_fil)
    maps = fits.open(maps_fil)
    if cube['primary'].header['dapqual'] == 30:
        raise ValueError(f"LOGCUBE flagged as CRITICAL in Primary header.")

    lamrest = 5897.558

    binid = cube['BINID'].data[0]

    stellarvel = maps['STELLAR_VEL'].data
    stellarvel_mask = maps['STELLAR_VEL_MASK'].data #DAPPIXMASK

    ew_hdul = fits.open(EW_file)
    ewmap = ew_hdul['EQ_WIDTH_NAI'].data
    ewmap_mask = ew_hdul['EQ_WIDTH_NAI_MASK'].data

    vel_map = np.zeros(binid.shape) - 999.
    print(vel_map.shape)
    vel_map_error = np.zeros_like(vel_map) - 999.
    vmap_mask = np.zeros_like(vel_map)

    frac_map = np.zeros_like(vel_map)

    term_vel_map = np.zeros_like(vel_map)
    term_vel_map_mask = np.zeros_like(term_vel_map)
    
    ### TODO ###
    stellarvel_ivar = maps['STELLAR_VEL_IVAR'].data 
    term_vel_map_error = np.copy(vel_map_error) 
    ewmap_error = ew_hdul['EQ_WIDTH_NAI_ERR'].data 
    ############


    ## mask unused spaxels
    w = binid == -1
    vmap_mask[w] = 6

    bins, inds = np.unique(mcmc_table['bin'],return_index=True)

    zipped_items = zip(bins,inds)
    iterator = tqdm(zipped_items, desc='Constructing Velocity Map') if verbose else zipped_items
    for ID,ind in iterator:
        if ID == -1:
            continue
        w = binid == ID

        bin_check = util.check_bin_ID(ID, binid, DAPPIXMASK_list=[stellarvel_mask], stellar_velocity_map=stellarvel)
        vmap_mask[w] = bin_check

        if args.mask:
            if np.median(ewmap_mask[w]).astype(bool):
                pass
            else:
                sn = snrmap[w][0]
                ew_cut = get_ew_cut(sn)
                ew = ewmap[w][0]

                if ew<ew_cut:
                    vmap_mask[w] = 7

        velocity = mcmc_table[ind]['velocities']
        lambda_samples = mcmc_table[ind]['lambda samples']
        percentiles = mcmc_table[ind]['percentiles']

        bD = percentiles[2,0]
        
        if velocity == 0:
            frac = 0
        else:
            frac = np.sum(lambda_samples > lamrest)/lambda_samples.size if velocity>0 else np.sum(lambda_samples < lamrest)/lambda_samples.size

        if velocity < 0:
            if frac >= .95:
                term_vel_out = velocity - np.sqrt(abs(np.log(0.1))) * bD
        else:
            term_vel_map_mask[w] = 1
            term_vel_out = 0

        frac_map[w] = frac
        vel_map[w] = velocity
        term_vel_map[w] = term_vel_out

    vmap_name = "Vel Map"
    vmap_dict = {f"{vmap_name}":vel_map, f"{vmap_name} Confidence":frac_map, f"{vmap_name} Mask":vmap_mask, f"{vmap_name} Uncertainty":vel_map_error}

    term_vel_map_name = "Vout"
    term_velmap_dict = {f"{term_vel_map_name}":term_vel_map, f"{term_vel_map_name} Mask":term_vel_map_mask, f"{term_vel_map_name} Uncertainty":term_vel_map_error}

    return vmap_dict, term_velmap_dict

    
def get_args():
    parser = argparse.ArgumentParser(description="A script to create an Na I velocity map from the MCMC results of a beta-corrected galaxy.")

    parser.add_argument('galname', type=str, help="Input galaxy name.")
    parser.add_argument('bin_method', type=str, help="Input DAP spatial binning method.")
    parser.add_argument('-v','--verbose', help = "Print verbose outputs (default: False)", action='store_true', default = False)
    parser.add_argument('-m', '--mask', help = "Mask velocities by S/N & EW combination (default: False)", action='store_true',default=False)

    return parser.parse_args()


def main(args):
    verbose = args.verbose
    analysisplan_methods = defaults.analysis_plans()
    corr_key = 'BETA-CORR'

    # initialize directories and paths
    data_dir = defaults.get_data_path('local')
    local_dir = os.path.join(data_dir, 'local_outputs')
    gal_local_dir = os.path.join(local_dir, f"{args.galname}-{args.bin_method}", corr_key, analysisplan_methods)
    gal_figures_dir = os.path.join(local_dir, f"{args.galname}-{args.bin_method}", "figures")

    util.check_filepath([gal_local_dir, gal_figures_dir], mkdir=True, verbose=verbose)

    datapath_dict = file_handler.init_datapaths(args.galname, args.bin_method)

    ## acquire logcube and maps files from datapath dict, raise error if they were not found
    configuration = file_handler.parse_config(datapath_dict['CONFIG'], verbose=args.verbose)
    redshift = float(configuration['z'])
    util.verbose_print(args.verbose, f"Redshift z = {redshift} found in {datapath_dict['CONFIG']}")
    
    cubefil = datapath_dict['LOGCUBE']
    if cubefil is None:
        raise ValueError(f"LOGCUBE file not found for {args.galname}-{args.bin_method}-{corr_key}")
    
    mapsfil = datapath_dict['MAPS']
    if mapsfil is None:
        raise ValueError(f"MAPS file not found for {args.galname}-{args.bin_method}-{corr_key}")

    ## TODO: fix
    mcmcfils = datapath_dict['MCMC']
    if mcmcfils is None or len(mcmcfils)==0:
        raise ValueError(f"No MCMC Reults found.")

    ew_files = [f for f in os.listdir(gal_local_dir) if 'EW-map.fits' in f]
    if ew_files:
        ew_file = os.path.join(gal_local_dir,ew_files[0])
        util.verbose_print(verbose, f"Using EW File: {ew_file}")
    else:
        raise FileNotFoundError(f"No file containing 'EW-Map.fits' found in {gal_local_dir} \ncontents: {os.listdir(gal_local_dir)}")



    # combine the mcmc fits files into an astropy table
    mcmc_table = combine_mcmc_results(mcmc_paths=mcmcfils, verbose=verbose)

    # write mcmc results into datacubes
    mcmc_dict, mcmc_header_dict = make_mcmc_results_cube(cubefil, mcmc_table, verbose=verbose)

    # compute the Na D S/N
    snrmap = NaD_snr_map(cubefil, mapsfil, redshift, verbose)

    # measure Doppler V of line center
    vmap_dict, term_vmap_dict = make_vmap(cubefil, mapsfil, snrmap, ew_file, mcmc_table, verbose)

    print(vmap_dict['Vel Map'].shape)
    breakpoint()

    # structure the data for writing
    hdu_name = "NaI_VELOCITY"
    units = "km / s"

    mcmc_mapdict = file_handler.standard_map_dict(args.galname, mcmc_dict, custom_header_dict=mcmc_header_dict)

    velocity_mapdict = file_handler.standard_map_dict(args.galname, vmap_dict, HDU_keywords=hdu_name, unit_strs=units)

    hdu_name = "V_MAX_OUTFLOW"
    velocity_outflow_max_dict = file_handler.standard_map_dict(args.galname, term_vmap_dict, HDU_keywords=hdu_name, unit_strs=units)

    # write the data
    gal_dir = f"{args.galname}-{args.bin_method}"
    file_handler.map_file_handler(gal_dir, velocity_mapdict, gal_local_dir,
                                  verbose=args.verbose)
    file_handler.map_file_handler(gal_dir, velocity_outflow_max_dict, gal_local_dir,
                                  verbose=args.verbose)
    file_handler.map_file_handler(gal_dir, mcmc_mapdict, gal_local_dir, 
                                  verbose=args.verbose)
    
    plotter.map_plotter(vmap_dict['Vel Map'], vmap_dict['Vel Map Mask'], gal_figures_dir, hdu_name, r"$v_{\mathrm{Na\ D}}$",
                        r"$\left( \mathrm{km\ s^{-1}} \right)$", args.galname, args.bin_method, vmin=-200, vmax=200, cmap='seismic')

if __name__ == "__main__":
    args = get_args()
    main(args)