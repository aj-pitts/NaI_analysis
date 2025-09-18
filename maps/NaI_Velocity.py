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
from modules import defaults, file_handler, util, plotter, inspect
import mcmc_results


def make_vmap(galname, bin_method, manual=False, verbose=False, write_data=True):
    datapath_dict = file_handler.init_datapaths(galname, bin_method)
    mcmc_table = mcmc_results.combine_mcmc_results(datapath_dict['MCMC'])

    cube = fits.open(datapath_dict['LOGCUBE'])
    maps = fits.open(datapath_dict['MAPS'])
    local = fits.open(datapath_dict['LOCAL'])

    if cube['primary'].header['dapqual'] == 30:
        raise ValueError(f"LOGCUBE flagged as CRITICAL in Primary header.")

    lamrest = 5897.558
    c = 2.998e5

    binid = cube['BINID'].data[0]

    stellarvel = maps['STELLAR_VEL'].data
    stellarvel_mask = maps['STELLAR_VEL_MASK'].data #DAPPIXMASK
    
    ewmap = local['ew_noem'].data
    snrmap = local['nai_snr'].data

    vel_map = np.zeros(binid.shape) - 999.

    vel_map_error_upper = np.zeros(binid.shape) - 999.
    vel_map_error_lower = np.zeros(binid.shape) - 999.
    vmap_mask = np.zeros_like(vel_map)

    frac_map = np.zeros_like(vel_map)
    
    ### TODO ###
    #stellarvel_ivar = maps['STELLAR_VEL_IVAR'].data
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

        lambda_median, lambda_err_upper, lambda_err_lower = percentiles[0]

        velocity_err_upper = c * lambda_err_upper / lamrest
        velocity_err_lower = c * lambda_err_lower / lamrest

        if not np.isfinite(velocity):
            vmap_mask[w] = 4
            velocity = -999

        if max(abs(velocity_err_upper), abs(velocity_err_lower)) >= 30:
            vmap_mask[w] = 8

        if not np.isfinite(velocity_err_upper) or not np.isfinite(velocity_err_lower):
            vmap_mask[w] = 5
            velocity = -999

        if velocity == 0 or velocity == -999:
            frac = 0
            velocity_err_upper = -999
            velocity_err_lower = -999

        else:
            frac = np.sum(lambda_samples > lamrest)/lambda_samples.size if velocity > 0 else np.sum(lambda_samples < lamrest)/lambda_samples.size

        frac_map[w] = frac
        vel_map[w] = velocity
        vel_map_error_upper[w] = velocity_err_upper
        vel_map_error_lower[w] = velocity_err_lower


    vel_map_error = np.stack([vel_map_error_lower, vel_map_error_upper], axis=0)

    threshold_mask = apply_velocity_mask(galname, bin_method, vel_map, vel_map_error, manual=manual, verbose=verbose)
    w = np.logical_and(threshold_mask.astype(bool), ~vmap_mask.astype(bool))
    vmap_mask[w] = 7

    vmap_name = "Vel Map"
    vmap_dict = {f"{vmap_name}":vel_map, f"{vmap_name} Confidence":frac_map, f"{vmap_name} Mask":vmap_mask, f"{vmap_name} Uncertainty":vel_map_error}

    if write_data:
        velocity_hduname = "V_NaI"
        additional_data = ["V_NaI_FRAC"]
        additional_units = ['']
        additinoal_description = ['Fractional confidence of NaI_VELOCITY']
        units = "km / s"
        velocity_mapdict = file_handler.standard_map_dict(galname, vmap_dict, HDU_keyword=velocity_hduname, IMAGE_units=units,
                                                        additional_keywords=additional_data, additional_units=additional_units, 
                                                        additional_descriptions=additinoal_description, asymmetric_error=True)
        file_handler.write_maps_file(galname, bin_method, [velocity_mapdict], verbose=verbose)
    else:
        return vmap_dict

        

def compute_ew_thresholds(galname, bin_method, vmap, vmap_error, vmap_mask = None, scatter_lim = 30, error = True, equal_N = False, verbose = False):
    util.verbose_print(verbose, "Computing EW lims...")

    datapath_dict = file_handler.init_datapaths(galname, bin_method, verbose = False)
    local_file = datapath_dict['LOCAL']
    hdul = fits.open(local_file)

    spatial_bins = hdul['spatial_bins'].data.flatten()
    unique_bins, bin_inds = np.unique(spatial_bins, return_index=True)

    snr = hdul['nai_snr'].data.flatten()[bin_inds]
    ew = hdul['ew_noem'].data.flatten()[bin_inds]
    ew_mask = hdul['ew_noem_mask'].data.flatten().astype(bool)[bin_inds]
    velocity = vmap.flatten()[bin_inds]
    if vmap_error.ndim == 3:
        vmap_error = np.mean(abs(vmap_error), axis=0)
    velocity_error = vmap_error.flatten()[bin_inds]
    

    if vmap_mask is not None:
        velocity_mask = vmap_mask.flatten()[bin_inds]
        velocity_mask[velocity_mask==7] = 0
        velocity_mask = velocity_mask.astype(bool)
        datamask = np.logical_or(velocity_mask, ew_mask)
    else:
        datamask = ew_mask

    datamask = np.logical_or(datamask, velocity_error == 0)
    threshold_dict = file_handler.threshold_parser(galname, bin_method, require_ew=False)

    data = {}
    ew_lims = []

    for (sn_low, sn_high) in threshold_dict['sn_lims']:
        sn_low = int(sn_low) if np.isfinite(sn_low) else sn_low
        sn_high = int(sn_high) if np.isfinite(sn_high) else sn_high

        key = f"{sn_low}-{sn_high}"
        data[key] = {}

        w = (snr > sn_low) & (snr <= sn_high) & (ew > 0)

        mask = np.logical_and(w, ~datamask)
        masked_ew = ew[mask]
        masked_velocities = velocity[mask]
        masked_errors = velocity_error[mask]
        
        if not equal_N:
            ew_bins = np.arange(masked_ew.min(), masked_ew.max(), 0.2)
            med_ew = []
            velocity_stat = []

            for i in range(len(ew_bins) - 1):
                ew_low = ew_bins[i]
                ew_high = ew_bins[i+1]
                ew_binmask = (masked_ew > ew_low) & (masked_ew <= ew_high)

                vels = masked_velocities[ew_binmask]
                errs = masked_errors[ew_binmask]
                if len(vels) < 10:
                    continue
                
                stat = np.median(errs) if error else np.std(vels)

                velocity_stat.append(stat)
                med_ew.append((ew_low + ew_high)/2)
        else:
            sorted_inds = np.argsort(masked_ew)
            EWs = masked_ew[sorted_inds]
            Velocities = masked_velocities[sorted_inds]
            Errors = masked_errors[sorted_inds]

            n_bins = 20
            n_points = len(EWs)
            points_per_bin = n_points//n_bins
            remainder = n_points % n_bins


            med_ew = []
            velocity_stat = []
            start_idx = 0

            for i in range(n_bins):
                bin_size = points_per_bin + (1 if i < remainder else 0)
                end_idx = start_idx + bin_size

                bin_ew = EWs[start_idx:end_idx]
                bin_vel = Velocities[start_idx:end_idx]
                bin_err = Errors[start_idx:end_idx]

                stat = np.median(bin_err) if error else np.std(bin_vel)
                velocity_stat.append(stat)
                med_ew.append(np.mean(bin_ew))
                start_idx = end_idx

        velocity_stat = np.array(velocity_stat)
        med_ew = np.array(med_ew)
        cut = velocity_stat < scatter_lim
        if np.sum(cut) == 0:
            ew_lim = np.inf
        else:
            ew_lim = np.min(med_ew[cut])

        data[key]['std'] = velocity_stat
        data[key]['medew'] = med_ew
        data[key]['ew_lim'] = ew_lim
        ew_lims.append(ew_lim)
    
    file_handler.write_thresholds(galname, ew_lims=ew_lims, overwrite=True)
    util.verbose_print(verbose, "Done.")
    inspect.inspect_vstd_ew(galname, bin_method, data, vmap, vmap_error, ewnoem=True, scatter_lim=scatter_lim, verbose=verbose)



def apply_velocity_mask(galname, bin_method, vmap, vmap_error, vmap_mask = None, manual = False, verbose = False):
    datapath_dict = file_handler.init_datapaths(galname, bin_method)
    local = fits.open(datapath_dict['LOCAL'])

    spatial_bins = local['spatial_bins'].data
    ewmap = local['ew_noem'].data
    snrmap = local['nai_snr'].data
    
    threshold_mask = np.zeros_like(spatial_bins)

    if not manual:
        compute_ew_thresholds(galname, bin_method, vmap, vmap_error, vmap_mask, verbose=verbose)
    
    else: 
        threshold_dict = file_handler.threshold_parser(galname, bin_method)
        if threshold_dict['ew_lims'] is None:
            print(f"Cannot apply velocity mask if using --manual and no thresholds written to thresholds.yaml")
            print(f"Skipping Velocity Mask")
            inspect.inspect_vel_ew(galname, bin_method, contour=True, verbose=verbose)
            return
            

    threshold_dict = file_handler.threshold_parser(galname, bin_method)
    
    ## function to return the ew threshold given the snr
    def get_ew_cut(snr: float, thresholds: dict):
        for (sn_min, sn_max), ew_lim in zip(thresholds['sn_lims'], thresholds['ew_lims']):
            if sn_min < snr <= sn_max:
                return ew_lim
        return np.inf

    ## iterate the bins and update the mask
    items = np.unique(spatial_bins)[1:]
    iterator = tqdm(items, desc="Masking velocities by EW and S/N threshold") if verbose else items

    for ID in iterator:
        w = ID == spatial_bins
        ny, nx = np.where(w)
        y, x = ny[0], nx[0]
        
        # apply the limit
        sn = snrmap[y, x]
        ew_cut = get_ew_cut(sn, threshold_dict)

        if ewmap[y, x] < ew_cut:
            threshold_mask[w] = 7

    return threshold_mask


def make_terminal_vmap(galname, bin_method, verbose = False, write_data = True):
    datapath_dict = file_handler.init_datapaths(galname, bin_method)

    cube = fits.open(datapath_dict['LOGCUBE'])
    local = fits.open(datapath_dict['LOCAL'])

    binid = cube['BINID'].data[0]
    
    vmap = local['v_nai'].data
    vmap_mask = local['v_nai_mask'].data
    vmap_error = local['v_nai_error'].data

    doppler_param = local['mcmc_results'].data[2]
    doppler_param_16 = local['mcmc_16th_perc'].data[2]
    doppler_param_84 = local['mcmc_84th_perc'].data[2]

    term_vmap = np.zeros_like(vmap)
    term_vmap_mask = np.zeros_like(vmap)
    term_vmap_error = np.zeros_like(vmap_error)

    items = np.unique(binid[1:])
    iterator = tqdm(items, desc="Constructing terminal velocity outflow map") if verbose else items
    for ID in iterator:
        w = ID == binid
        Y, X = np.where(w)
        y, x = Y[0], X[0]

        bin_v_mask = vmap_mask[y, x]
        if bool(bin_v_mask):
            term_vmap_mask[w] = bin_v_mask
        
        #bin_frac = frac[y, x]
        bin_vel = vmap[y, x]
        bin_vel_mask = vmap_mask[y, x]
        bin_vel_error_16 = vmap_error[0 ,y, x]
        bin_vel_error_84 = vmap_error[1, y, x]


        term_vmap_mask[w] = bin_vel_mask
        
        bin_bD = doppler_param[y, x]
        bD_upper = doppler_param_84[y, x]
        bD_lower = doppler_param_16[y, x]

        terminal_velocity = bin_vel - (np.sqrt(abs(np.log(0.1))) * bin_bD)

        if terminal_velocity >= 0:
            term_vmap_mask[w] = 9
            continue

        term_vmap[w] = abs(terminal_velocity)

        term_vmap_error[0][w] = np.sqrt( bin_vel_error_16**2 + (np.sqrt(-np.log(0.1)) * bD_lower)**2 )
        term_vmap_error[1][w] = np.sqrt( bin_vel_error_84**2 + (np.sqrt(-np.log(0.1)) * bD_upper)**2 )

    name = "Vout"
    term_vdict = {f"{name}":term_vmap, f"{name} Mask":term_vmap_mask, f"{name} Uncertainty":term_vmap_error}
    if write_data:
        vout_hdu_name = "V_MAX_OUT"
        units = "km / s"
        additional_masks = [(9, 'Bin Sodium is redshifted'), (10,'Bin velocity confidence is < 95%')]
        terminal_velocity_mapdict = file_handler.standard_map_dict(galname, term_vdict, HDU_keyword=vout_hdu_name, IMAGE_units=units, 
                                                                additional_mask_bits=additional_masks, asymmetric_error=True)
        file_handler.write_maps_file(galname,bin_method,[terminal_velocity_mapdict], verbose=verbose)
    else:
        return term_vdict
        

    
def get_args():
    parser = argparse.ArgumentParser(description="A script to create an Na I velocity map from the MCMC results of a beta-corrected galaxy.")

    parser.add_argument('galname', type=str, help="Input galaxy name.")
    parser.add_argument('bin_method', type=str, help="Input DAP spatial binning method.")
    parser.add_argument('-v','--verbose', help = "Print verbose outputs (default: False)", action='store_true', default = False)

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