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
    elif snr <= 60:
        return 1.0
    elif snr <= 90:
        return 0.8
    else:
        return 0.6

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

    items = np.unique(spatial_bins)
    iterator = tqdm(np.unique(spatial_bins), desc="Constructing S/N Map") if verbose else items
    for ID in iterator:
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
            "BUNIT_02":("1 / cm^2", "Unit of pixel value in C02"),
            "BUNIT_03":("km / s", "Unit of pixel value in C03"),
            "BUNIT_04":(" ", "Unit of pixel value in C04"),
            "ERRDATA1":(f"MCMC_16TH_PERC", "Associated 16th percentile uncertainty values extension"),
            "ERRDATA2":(f"MCMC_84TH_PERC", "Associated 84th percentile uncertainty values extension"),
            "EXTNAME":(HDU_name, "Extension name"),
            "AUTHOR":("Andrew Pitts","")
        },
        f"MCMC_16TH_PERC":{
            "DESC":(f"{args.galname} {HDU_name.replace("_"," ")} 16th percentile uncertainty",""),
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
            "DESC":(f"{args.galname} {HDU_name.replace("_"," ")} 84th percentile uncertainty",""),
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
    return mcmc_dict, mcmc_header_dict


def make_vmap(cube_fil, maps_fil, EW_file, mcmc_table, redshift, verbose=False):
    cube = fits.open(cube_fil)
    maps = fits.open(maps_fil)
    if cube['primary'].header['dapqual'] == 30:
        raise ValueError(f"LOGCUBE flagged as CRITICAL in Primary header.")

    lamrest = 5897.558
    c = 2.998e5

    binid = cube['BINID'].data[0]

    stellarvel = maps['STELLAR_VEL'].data
    stellarvel_mask = maps['STELLAR_VEL_MASK'].data #DAPPIXMASK

    ew_hdul = fits.open(EW_file)
    ewmap = ew_hdul['EQ_WIDTH_NAI'].data
    ewmap_mask = ew_hdul['EQ_WIDTH_NAI_MASK'].data

    vel_map = np.zeros(binid.shape) - 999.

    vel_map_error = np.zeros((2, binid.shape[0], binid.shape[1])) - 999.
    vmap_mask = np.zeros_like(vel_map)

    frac_map = np.zeros_like(vel_map)
    
    ### TODO ###
    stellarvel_ivar = maps['STELLAR_VEL_IVAR'].data
    ewmap_error = ew_hdul['EQ_WIDTH_NAI_ERR'].data 
    ############

    # compute the Na D S/N
    if args.mask:
        snrmap = NaD_snr_map(cube_fil, maps_fil, redshift, verbose=verbose)

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

        lambda_percentiles = percentiles[0]

        lambda_16 = lambda_percentiles[1]
        lambda_84 = lambda_percentiles[2]

        velocity_16 = c * lambda_16 / lamrest
        velocity_84 = c * lambda_84 / lamrest

        vel_map_error[0][w] = velocity_16
        vel_map_error[1][w] = velocity_84


        if min(abs(velocity_84), abs(velocity_16)) >= 20: # ~ lambda unc of 0.4 Å
            vmap_mask[w] = 8
        
        if velocity == 0:
            frac = 0
        else:
            frac = np.sum(lambda_samples > lamrest)/lambda_samples.size if velocity > 0 else np.sum(lambda_samples < lamrest)/lambda_samples.size

        frac_map[w] = frac
        vel_map[w] = velocity

    vmap_name = "Vel Map"
    vmap_dict = {f"{vmap_name}":vel_map, f"{vmap_name} Confidence":frac_map, f"{vmap_name} Mask":vmap_mask, f"{vmap_name} Uncertainty":vel_map_error}

    return vmap_dict


def make_terminal_vmap(vmap_dict, mcmc_dict, cube_fil, verbose=False):
    cube = fits.open(cube_fil)

    binid = cube['BINID'].data[0]

    doppler_param = mcmc_dict['MCMC Results'][2]
    doppler_param_16 = mcmc_dict['MCMC 16th Percentile'][2]
    doppler_param_84 = mcmc_dict['MCMC 84th Percentile'][2]

    vmap = vmap_dict['Vel Map']
    vmap_mask = vmap_dict['Vel Map Mask']
    vmap_error = vmap_dict['Vel Map Uncertainty']
    frac = vmap_dict['Vel Map Confidence']

    term_vmap = np.zeros_like(binid)
    term_vmap_mask = np.zeros_like(binid)
    term_vmap_error = np.zeros((2, binid.shape[0], binid.shape[1]))

    items = np.unique(binid[1:])
    iterator = tqdm(items, desc="Constructing terminal velocity outflow map") if verbose else items
    for ID in iterator:
        w = ID == binid
        Y, X = np.where(w)
        y, x = Y[0], X[0]

        bin_v_mask = vmap_mask[y, x]
        if bool(bin_v_mask):
            term_vmap_mask[w] = bin_v_mask
        
        bin_frac = frac[y, x]
        bin_vel = vmap[y, x]
        bin_vel_mask = vmap_mask[y, x]
        bin_vel_error_16 = vmap_error[0 ,y, x]
        bin_vel_error_84 = vmap_error[1, y, x]


        term_vmap_mask[w] = bin_vel_mask
        
        if bin_vel >= 0:
            term_vmap_mask[w] = 9
            continue
        
        if bin_frac < .95:
            term_vmap_mask[w] = 10
            continue
        
        bin_bD = doppler_param[y, x]
        bD_upper = bin_bD + doppler_param_84[y, x]
        bD_lower = bin_bD - doppler_param_16[y, x]

        term_vmap[w] = abs(bin_vel) + (np.sqrt(abs(np.log(0.1))) * bin_bD)

        term_vmap_error[0][w] = np.sqrt( bin_vel_error_16**2 + (np.log(0.1) * bD_lower)**2 )
        term_vmap_error[1][w] = np.sqrt( bin_vel_error_84**2 + (np.log(0.1) * bD_upper)**2 )

    name = "Vout"
    return {f"{name}":term_vmap, f"{name} Mask":term_vmap_mask, f"{name} Uncertainty":term_vmap_error}
        

    
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

    datapath_dict = file_handler.init_datapaths(args.galname, args.bin_method, verbose=verbose)

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

    # measure Doppler V of line center
    vmap_dict = make_vmap(cubefil, mapsfil, ew_file, mcmc_table, redshift, verbose)
    term_vmap_dict = make_terminal_vmap(vmap_dict=vmap_dict, mcmc_dict=mcmc_dict, cube_fil=cubefil, 
                                        verbose=verbose)


    # structure the data for writing
    velocity_hduname = "V_NaI"
    additional_data = ["V_NaI_FRAC"]
    additional_units = ['']
    additinoal_description = ['Fractional confidence of NaI_VELOCITY']
    units = "km / s"

    mcmc_mapdict = file_handler.standard_map_dict(args.galname, mcmc_dict, custom_header_dict=mcmc_header_dict)

    velocity_mapdict = file_handler.standard_map_dict(args.galname, vmap_dict, HDU_keyword=velocity_hduname, IMAGE_units=units,
                                                      additional_keywords=additional_data, additional_units=additional_units, 
                                                      additional_descriptions=additinoal_description, asymmetric_error=True)

    vout_hdu_name = "V_MAX_OUT"
    additional_masks = [(9, 'Bin Sodium is redshifted'), (10,'Bin velocity confidence is < 95%')]
    velocity_outflow_max_dict = file_handler.standard_map_dict(args.galname, term_vmap_dict, HDU_keyword=vout_hdu_name, IMAGE_units=units, 
                                                               additional_mask_bits=additional_masks, asymmetric_error=True)

    # write the data
    gal_dir = f"{args.galname}-{args.bin_method}"

    file_handler.map_file_handler(gal_dir, [velocity_mapdict, velocity_outflow_max_dict, mcmc_mapdict], 
                                  gal_local_dir, verbose=args.verbose)
    
    # make the plots
    plotter.map_plotter(vmap_dict['Vel Map'], vmap_dict['Vel Map Mask'], gal_figures_dir, velocity_hduname, r"$v_{\mathrm{Na\ D}}$",
                        r"$\left( \mathrm{km\ s^{-1}} \right)$", args.galname, args.bin_method, vmin=-200, vmax=200, cmap='seismic')

if __name__ == "__main__":
    args = get_args()
    main(args)