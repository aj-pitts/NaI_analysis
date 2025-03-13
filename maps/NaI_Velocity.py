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
import mcmc_results


def make_vmap(cube_fil, maps_fil, mcmc_table, verbose=False):
    cube = fits.open(cube_fil)
    maps = fits.open(maps_fil)
    if cube['primary'].header['dapqual'] == 30:
        raise ValueError(f"LOGCUBE flagged as CRITICAL in Primary header.")

    lamrest = 5897.558
    c = 2.998e5

    binid = cube['BINID'].data[0]

    stellarvel = maps['STELLAR_VEL'].data
    stellarvel_mask = maps['STELLAR_VEL_MASK'].data #DAPPIXMASK

    vel_map = np.zeros(binid.shape) - 999.

    vel_map_error = np.zeros((2, binid.shape[0], binid.shape[1])) - 999.
    vmap_mask = np.zeros_like(vel_map)

    frac_map = np.zeros_like(vel_map)
    
    ### TODO ###
    stellarvel_ivar = maps['STELLAR_VEL_IVAR'].data
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


def apply_velocity_mask(galname, bin_method, spatial_bins, velocity_map_mask, ewmap, ewmap_mask, snrmap, verbose = False):

    threshold_dict = file_handler.threshold_parser(galname, bin_method)

    ## function to return the ew threshold
    def get_ew_cut(snr, threshold_dict):
        snr_lims = threshold_dict['snr']
        ew_lims = threshold_dict['ew']

        for i, sn_lim in enumerate(snr_lims):
            if snr <= sn_lim:
                return np.inf if ew_lims[i] == 'inf' else ew_lims[i]
            
        return ew_lims[-1]


    ## iterate the bins
    items = np.unique(spatial_bins)[1:]
    iterator = tqdm(items, desc="Masking velocities by EW and S/N threshold") if verbose else items

    for ID in iterator:
        w = ID == spatial_bins
        ny, nx = np.where(w)
        y, x = ny[0], nx[0]

        # if the ew was masked, mask the velocity
        if np.median(ewmap_mask[w]).astype(bool):
            velocity_map_mask[w] = 7
            continue
        
        # apply the limit
        sn = snrmap[y, x]
        ew_cut = get_ew_cut(sn, threshold_dict)

        if ewmap[y, x] < ew_cut:
            velocity_map_mask[w] = 7


def make_terminal_vmap(vmap_dict, mcmc_dict, cube_fil, verbose = False):
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

    local_fp = [f for f in os.listdir(gal_local_dir) if 'local_maps.fits' in f]
    if local_fp:
        local_file = os.path.join(gal_local_dir, local_fp[0])
        util.verbose_print(verbose, f"Using Local Maps File: {local_file}")
    else:
        raise FileNotFoundError(f"No file containing 'local_maps.fits' found in {gal_local_dir} \ncontents: {os.listdir(gal_local_dir)}")



    # combine the mcmc fits files into an astropy table
    mcmc_table = mcmc_results.combine_mcmc_results(mcmc_paths=mcmcfils, verbose=verbose)

    # write mcmc results into datacubes
    mcmc_dict, mcmc_header_dict = mcmc_results.make_mcmc_results_cube(args.galname, cubefil, mcmc_table, verbose=verbose)

    # measure Doppler V of line center
    vmap_dict = make_vmap(cubefil, mapsfil, local_file, mcmc_table, redshift, verbose)
    term_vmap_dict = make_terminal_vmap(vmap_dict=vmap_dict, mcmc_dict=mcmc_dict, cube_fil=cubefil, 
                                        verbose=verbose)

    # structure the data for writing
    velocity_hduname = "V_NaI"
    additional_data = ["V_NaI_FRAC"]
    additional_units = ['']
    additinoal_description = ['Fractional confidence of NaI_VELOCITY']
    units = "km / s"

    velocity_mapdict = file_handler.standard_map_dict(args.galname, vmap_dict, HDU_keyword=velocity_hduname, IMAGE_units=units,
                                                      additional_keywords=additional_data, additional_units=additional_units, 
                                                      additional_descriptions=additinoal_description, asymmetric_error=True)

    vout_hdu_name = "V_MAX_OUT"
    additional_masks = [(9, 'Bin Sodium is redshifted'), (10,'Bin velocity confidence is < 95%')]
    velocity_outflow_max_dict = file_handler.standard_map_dict(args.galname, term_vmap_dict, HDU_keyword=vout_hdu_name, IMAGE_units=units, 
                                                               additional_mask_bits=additional_masks, asymmetric_error=True)

    # write the data
    gal_dir = f"{args.galname}-{args.bin_method}"

    file_handler.map_file_handler(gal_dir, [velocity_mapdict, velocity_outflow_max_dict], 
                                  gal_local_dir, verbose=args.verbose)
    
    # make the plots
    plotter.map_plotter(vmap_dict['Vel Map'], vmap_dict['Vel Map Mask'], gal_figures_dir, velocity_hduname, r"$v_{\mathrm{Na\ D}}$",
                        r"$\left( \mathrm{km\ s^{-1}} \right)$", args.galname, args.bin_method, vmin=-200, vmax=200, cmap='seismic')

if __name__ == "__main__":
    args = get_args()
    main(args)