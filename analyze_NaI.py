from glob import glob
import argparse
import os
import warnings
from astropy.io import fits

import numpy as np

import maps.Hii
import maps.NaI_EW
import maps.NaI_SNR
import maps.NaI_Velocity
import maps.SFR
import maps.redshift
from modules import util, defaults, file_handler, plot_results, inspect
import maps
import mcmc_results

### TODO: Add argument to put lines on inspect plot

def get_args():    
    parser = argparse.ArgumentParser(description="A script to create an equivalent width map of ISM Na I for beta-corrected DAP outputs.")
    
    parser.add_argument('galname',type=str,help='Input galaxy name.')
    parser.add_argument('bin_method',type=str,help='Input DAP patial binning method.')
    parser.add_argument('-v', '--verbose', help='Print verbose outputs. (Default: False)', action='store_true', default=False)
    
    return parser.parse_args()


def main(args):
    galname = args.galname
    bin_method = args.bin_method
    verbose = args.verbose
    analysis_plan = defaults.analysis_plans()
    corr_key = 'BETA-CORR'

    local_data = defaults.get_data_path(subdir='local')
    datapath_dict = file_handler.init_datapaths(galname, bin_method, verbose, redshift=True)

    z_config = datapath_dict['Z']
    mapsfile = datapath_dict['MAPS']
    cubefile = datapath_dict['LOGCUBE']
    mcmcfiles = datapath_dict['MCMC']

    cube_hdu = fits.open(cubefile)
    maps_hdu = fits.open(mapsfile)

    radius = maps_hdu['bin_lwellcoo'].data[1]
    radius_mapdict = {"RADIUS":radius}
    radius_header = file_handler.use_sdss_header(galname, bin_method, "RADIUS", maps_hdu['bin_lwellcoo'].header, desc="radius map", 
                                                 remove_keywords=['C1', 'U1', 'C3', 'U3', 'C4', 'U4'])
    radius_dict = file_handler.standard_map_dict(galname, radius_mapdict, custom_header_dict=radius_header)

    binids = cube_hdu['BINID'].data
    spatial_bins = binids[0]

    spatial_bins_mapdict = {"SPATIAL_BINS":spatial_bins}
    spatial_bins_header = file_handler.use_sdss_header(galname, bin_method, "SPATIAL_BINS", cube_hdu['binid'].header, "spatial bin IDs",
                                                       remove_keywords=['C2', 'C3', 'C4', 'C5'])
    spatial_bins_dict = file_handler.standard_map_dict(galname, spatial_bins_mapdict, custom_header_dict=spatial_bins_header)

    ## output directories
    output_dir = os.path.join(local_data, 'local_outputs', f"{galname}-{bin_method}", corr_key, analysis_plan)
    gal_figures_dir = os.path.join(output_dir, 'figures')
    inspect_figures_dir = os.path.join(gal_figures_dir, 'inspection')
    results_figures_dir = os.path.join(gal_figures_dir, 'results')

    ####### REDSHIFT #######
    zmap_dict = maps.redshift.redshift_map_dict(z_config, mapsfile, verbose=verbose)
    redshift_mapdict = file_handler.standard_map_dict(galname, zmap_dict, HDU_keyword="REDSHIFT", IMAGE_units="")

    ####### S/N NaD #######
    snr_dict, snr_header = maps.NaI_SNR.NaD_snr_map(galname, cubefile, mapsfile, zmap_dict['Redshift Map'], verbose=verbose)
    snr_mapdict = file_handler.standard_map_dict(galname, snr_dict, custom_header_dict=snr_header)

    ####### EQ W NaD #######
    ewmap_dict = maps.NaI_EW.measure_EW(cubefile, mapsfile, z_config, verbose=verbose)
    hdu_name = "EW_NAI"
    units = "Angstrom"
    ew_mapdict = file_handler.standard_map_dict(galname, ewmap_dict, HDU_keyword=hdu_name, IMAGE_units=units)    

    ####### Sigma_SFR #######
    sfr_dict = maps.SFR.SFR_map(mapsfile, z_config, verbose=verbose)
    hdu_keyword = "SFRSD"
    unit = r"$\left( \mathrm{M_{\odot}\ kpc^{-2}\ yr^{-1}\ spaxel^{-1}} \right)$"
    sfr_mapdict = file_handler.standard_map_dict(galname, sfr_dict, HDU_keyword=hdu_keyword, IMAGE_units=unit)
 
    ####### MCMC RESULTS #######
    mcmc_table = mcmc_results.combine_mcmc_results(mcmcfiles, verbose=verbose)
    mcmc_dict, mcmc_header = mcmc_results.make_mcmc_results_cube(galname, cubefile, mcmc_table, verbose=verbose)
    mcmc_cubedict = file_handler.standard_map_dict(galname, mcmc_dict, custom_header_dict=mcmc_header)

    ####### NaD VELOCITY #######
    vmap_dict = maps.NaI_Velocity.make_vmap(galname, bin_method, cubefile, mapsfile, mcmc_table, ewmap_dict['EW Map'], ewmap_dict['EW Map Mask'],
                                            snr_dict['NaI_SNR'], verbose=verbose)
        
    
    velocity_hduname = "V_NaI"
    additional_data = ["V_NaI_FRAC"]
    additional_units = ['']
    additinoal_description = ['Fractional confidence of NaI_VELOCITY']
    units = "km / s"
    velocity_mapdict = file_handler.standard_map_dict(galname, vmap_dict, HDU_keyword=velocity_hduname, IMAGE_units=units,
                                                      additional_keywords=additional_data, additional_units=additional_units, 
                                                      additional_descriptions=additinoal_description, asymmetric_error=True)
    
    ####### Terminal VELOCITY #######
    terminal_vmap_dict = maps.NaI_Velocity.make_terminal_vmap(vmap_dict, mcmc_dict, cubefile, verbose=verbose)
    vout_hdu_name = "V_MAX_OUT"
    additional_masks = [(9, 'Bin Sodium is redshifted'), (10,'Bin velocity confidence is < 95%')]
    terminal_velocity_mapdict = file_handler.standard_map_dict(galname, terminal_vmap_dict, HDU_keyword=vout_hdu_name, IMAGE_units=units, 
                                                               additional_mask_bits=additional_masks, asymmetric_error=True)
    
    ####### HII #######
    maps.Hii.write_fluxes(galname, bin_method, exists_ok=True, verbose=verbose)
    hii_dict, hii_header = maps.Hii.get_hii_mapdict(galname, bin_method, placeholder=True, verbose=verbose)
    hii_mapdict = file_handler.standard_map_dict(galname, hii_dict, custom_header_dict=hii_header)

    util.verbose_print(verbose, f"Analysis Complete!\nPreparing to write data to {output_dir}.")

    mapdict_list = [spatial_bins_dict, radius_dict, redshift_mapdict, snr_mapdict, ew_mapdict, sfr_mapdict, mcmc_cubedict, velocity_mapdict, terminal_velocity_mapdict, hii_mapdict]
    file_handler.map_file_handler(f"{galname}-{bin_method}", mapdict_list, 
                                  output_dir, verbose=verbose, preserve_standard_order= True, overwrite=True)
    
    ###### RESULT PLOTTTING #######
    plot_results.main(args=args)

    ## TODO
    ###### INSPECT PLOTTING #######


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    args = get_args()
    main(args)