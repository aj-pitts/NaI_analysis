import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table, join
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

def NaD_snr(spatial_bins, stellarvel, wave, fluxcube, ivarcube, z_guess):
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
    items = mcmc_paths
    iterator = enumerate(tqdm(mcmc_paths,desc="Combining MCMC results")) if verbose else enumerate(items)

    table = None
    for i,mcmc_fil in iterator:
        data = fits.open(mcmc_fil)
        data_table = Table(data[1].data)
        data_table.remove_columns(['samples','percentiles'])
        data_table['id'] = np.arange(len(data_table))

        if i == 0:
            table = data_table
            continue

        table = join(table, data_table, join_type='outer')

    return table

## TODO: add all mcmc parameters to output
def make_vmap(cube_fil, maps_fil, EW_file, mcmc_paths, redshift, verbose=False):
    cube = fits.open(cube_fil)
    maps = fits.open(maps_fil)
    if cube['primary'].header['dapqual'] == 30:
        raise ValueError(f"LOGCUBE flagged as CRITICAL in Primary header.")
    

    binid = cube['BINID'].data[0]
    flux = cube['FLUX'].data
    ivar = cube['IVAR'].data
    wave = cube['WAVE'].data

    stellarvel = maps['STELLAR_VEL'].data
    stellarvel_mask = maps['STELLAR_VEL_MASK'].data #DAPPIXMASK
    stellarvel_ivar = maps['STELLAR_VEL_IVAR'].data

    lamrest = 5897.558

    ew_hdul = fits.open(EW_file)
    ewmap = ew_hdul['EQ_WIDTH_NAI'].data
    ewmap_mask = ew_hdul['EQ_WIDTH_NAI_MASK'].data
    ewmap_error = ew_hdul['EQ_WIDTH_NAI_ERR'].data



    vel_map = np.zeros(binid.shape) - 999.
    logN_map = np.copy(vel_map)
    cf_map = np.copy(vel_map)
    bd_map = np.copy(vel_map)

    mcmc_map_mask = np.zeros_like(vel_map)

    vel_map_error = np.zeros_like(vel_map) - 999.
    logN_map_error = np.copy(vel_map_error)
    cf_map_error = np.copy(vel_map_error)
    bd_map_error = np.copy(vel_map_error)

    ## mask unused spaxels
    w = binid == -1
    mcmc_map_mask[w] = 6

    snrmap = NaD_snr(binid, stellarvel, wave, flux, ivar, redshift)

    mcmc_table = combine_mcmc_results(mcmc_paths=mcmc_paths, verbose=verbose)
    
    bins, inds = np.unique(mcmc_table['bin'],return_index=True)

    zipped_items = zip(bins,inds)
    iterator = tqdm(zipped_items, desc='Constructing Velocity Map') if verbose else zipped_items
    for ID,ind in iterator:
        if ID == -1:
            continue
        w = binid == ID
        indxs = np.where(binid == ID)

        bin_check = util.check_bin_ID(ID, binid, DAPPIXMASK_list=[stellarvel_mask], stellar_velocity_map=stellarvel)
        mcmc_map_mask[w] = bin_check

        if args.mask:
            if np.median(ewmap_mask[w]).astype(bool):
                pass
            else:
                sn = snrmap[w][0]
                ew_cut = get_ew_cut(sn)
                ew = ewmap[w][0]

                if ew<ew_cut:
                    mcmc_map_mask[w] = 7

        vel_map[w] = mcmc_table[ind]['velocities']
        #logN_map[w] = mcmc_table[ind]['']
        #cf_map[w] = mcmc_table[ind]['']

    
    return {"Vel Map":vel_map, "Vel Map Mask":mcmc_map_mask, "Vel Map Uncertainty":vel_map_error}
    
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


    vmap_dict = make_vmap(cubefil, mapsfil, ew_file, mcmcfils, redshift, verbose)

    # write the data
    hdu_name = "NaI_VELOCITY"
    units = "km / s"

    mapdict = file_handler.standard_map_dict(args.galname, hdu_name, units, vmap_dict)

    # file_handler.simple_file_handler(f"{args.galname}-{args.bin_method}", mapdict, 'velocity-map', gal_local_dir,
    #                                  overwrite=True, verbose=True)
    file_handler.map_file_handler(f"{args.galname}-{args.bin_method}", mapdict, gal_local_dir,
                                  verbose=args.verbose)
    
    plotter.map_plotter(vmap_dict['Vel Map'], vmap_dict['Vel Map Mask'], gal_figures_dir, hdu_name, r"$v_{\mathrm{Na\ D}}$",
                        r"$\left( \mathrm{km\ s^{-1}} \right)$", args.galname, args.bin_method, vmin=-200, vmax=200, cmap='seismic')

if __name__ == "__main__":
    args = get_args()
    main(args)