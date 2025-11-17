import os
import numpy as np
from astropy.io import fits, ascii

import argparse

from scipy.optimize import curve_fit
from scipy.stats import pearsonr

from modules import util, file_handler, defaults



def get_args():
    parser = argparse.ArgumentParser(description="A script to create/overwrite plots without rerunning analyze_NaI")

    parser.add_argument('galname', type=str, help="Input galaxy name.")
    #parser.add_argument('-v','--verbose', help = "Print verbose outputs (default: False)", action='store_true', default = False)

    return parser.parse_args()

def main():
    args = get_args()

    galname = args.galname

    bin_methods = ['SQUARE0.6', 'SQUARE2.0']
    for bin_method in bin_methods:
        velocity_results(galname, bin_method)


def velocity_results(galname, bin_method):
    util.sys_message('--------------------------------', color='yellow')
    util.sys_message(f'----{galname} {bin_method}----', color='yellow')
    util.sys_message('###### v_cen results ######', color='yellow')
    datapath_dict = file_handler.init_datapaths(galname, bin_method)

    with fits.open(datapath_dict['LOCAL']) as local_hdu:
        spatial_bins = local_hdu['spatial_bins'].data

        vmap = local_hdu['v_nai'].data
        vmap_mask = local_hdu['v_nai_mask'].data
        vfrac = local_hdu['v_nai_frac'].data

        eqw = local_hdu['ew_noem'].data
        eqw_mask = local_hdu['ew_noem_mask'].data

        mcmc_cube = local_hdu['mcmc_results'].data

        logN = mcmc_cube[1]
        bd = mcmc_cube[2]
        Cf = mcmc_cube[3]

        sfrmap = local_hdu['sfrsd'].data
        sfr_mask = local_hdu['sfrsd_mask'].data
        ebv = local_hdu['e(b-v)'].data

    n_bins_total = np.unique(spatial_bins).size

    datamask = (vmap_mask + eqw_mask + sfr_mask).astype(bool)
    mcmc_mask = (logN == 0) | (bd == 0) | (Cf == 0)
    v_frac_mask = (vfrac < 0.95) & (vfrac > -0.95)

    mask = (datamask + mcmc_mask + v_frac_mask).astype(bool)
    n_bins_velocity = np.unique(spatial_bins[~mask]).size

    util.sys_message(f"    {n_bins_velocity} velocity bins out of {n_bins_total} total bins, fraction = {n_bins_velocity/n_bins_total:.2f}", status = 'RESU', color='green')

    v_con = vmap > 0
    n_bins_outflow = np.unique(spatial_bins[~(mask+v_con).astype(bool)]).size
    util.sys_message(f"    {n_bins_outflow} OUTFLOW bins out of {n_bins_total} total bins, tot_fraction = {n_bins_outflow/n_bins_total:.2f}, detec_fraction = {n_bins_outflow/n_bins_velocity:.2f}", status = 'RESU', color='green')
    v_con = vmap < 0
    n_bins_inflow = np.unique(spatial_bins[~(mask+v_con).astype(bool)]).size
    util.sys_message(f"    {n_bins_inflow} INFLOW bins out of {n_bins_total} total bins, tot_fraction = {n_bins_inflow/n_bins_total:.2f}, detec_fraction = {n_bins_inflow/n_bins_velocity:.2f}", status = 'RESU', color='green')

    mapdict = {'EW':eqw, 'vcen':vmap, 'extinction':ebv, 'sfr':sfrmap}

    v_class = {
        'Outflow':vmap > 0,
        'Inflow':vmap < 0
    }
    for key, v_cond in v_class.items():
        fullmask = v_cond + mask
        extracted_maps = util.extract_unique_binned_values(mapdict, spatial_bins, mask=fullmask)

        util.sys_message(f"        {key} sample Pearson Correlation:", color='yellow')
        for xkey in ['sfr', 'extinction']:
            for ykey in ['vcen', 'EW']:
                pearson = pearsonr(extracted_maps[xkey], extracted_maps[ykey])
                util.sys_message(
                    f"            {ykey:<4} vs {xkey:<12} :  {'r':>2} = {pearson[0]:>2.3f}, {'p':>2} = {pearson[1]:>9.3g}", 
                    status='RESU', color='green')
    


    

if __name__ == "__main__":
    main()