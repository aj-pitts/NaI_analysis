from glob import glob
import argparse
import os
import warnings
import shutil
from astropy.io import fits

import numpy as np

import collaboration.barolo
import collaboration.Hii
import maps.NaI_EW
import maps.NaI_SNR
import maps.NaI_Velocity
import maps.SFR
import maps.redshift
import maps.BPT
import maps.metallicity
from modules import util, defaults, file_handler, inspect
import maps
import mcmc_results
import plot_results


def get_args():    
    parser = argparse.ArgumentParser(description="A script to create an equivalent width map of ISM Na I for beta-corrected DAP outputs.")
    
    parser.add_argument('galname',type=str,help='Input galaxy name.')
    parser.add_argument('bin_method',type=str,help='Input DAP patial binning method.')
    parser.add_argument('--verbose', help='Print verbose outputs. (Default: False)', action='store_true', default=False)
    parser.add_argument('--newfile', help='Send the current local_maps fits file to /backups/ instead of attempting to overwrite. (Default: False)', action='store_true', default=False)
    parser.add_argument('--thresh', help='Compute EW thresholds for masking in thresholds.yaml. (Default: False)', action='store_true', default=False)
    parser.add_argument('--paper', help = "Make plots exclusively for the manuscript (default: False)", action='store_true', default = False)
    
    return parser.parse_args()



def record_galaxy_properties():
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from astropy.io import ascii
    supplementary_dir = defaults.get_data_path('supplemental')
    sgafil = os.path.join(supplementary_dir, 'sga2020.fits')
    s4gfil = os.path.join(supplementary_dir, 'asu.fit')
    madtabfil = os.path.join(supplementary_dir, 'MAD_sample.dat')

    with fits.open(sgafil) as hdu:
        sgadata = hdu[1].data
    with fits.open(s4gfil) as hdu:
        s4gdata = hdu[1].data

    madtable = ascii.read(madtabfil)

    pipelinedir = defaults.get_data_path('pipeline')
    muse_cubes = os.path.join(pipelinedir, 'muse_cubes')
    galnames = [dirname for dirname in os.listdir(muse_cubes) if 'NGC' in dirname or 'PGC' in dirname or 'IC' in dirname]
    
    dtype = []
    dtype.append(('name', 'U16'))
    dtype.append(('ra', 'U16'))
    dtype.append(('dec', 'U16'))
    dtype.append(('morph', 'U8'))
    dtype.append(('redshift', float))
    dtype.append(('reff', float))
    dtype.append(('i', float))
    dtype.append(('logM', float))
    dtype.append(('SFR', float))
    dtype.append(('ebv', float))

    data = np.zeros(len(galnames), dtype=dtype)
    
    t_type_mapping = {
    -6: 'cE', -5: 'E', -4: 'E+', -3: 'S0−', -2: 'S0', -1: 'S0+',
     0: 'S0/a', 1: 'Sa', 2: 'Sab', 3: 'Sb', 4: 'Sbc', 5: 'Sc',
     6: 'Scd', 7: 'Sd', 8: 'Sdm', 9: 'Sm', 10: 'Im', 11: 'Pec'
    }#https://ui.adsabs.harvard.edu/abs/1991rc3..book.....D/abstract
    
    for i, galname in enumerate(galnames):
        galdir = os.path.join(muse_cubes, galname)

        sgarow = sgadata[sgadata['GALAXY'] == galname]
        s4grow = s4gdata[s4gdata['Name'] == galname]
        madrow = madtable[madtable['col1'] == galname]

        if len(sgarow) != 0:
            morph = sgarow['MORPHTYPE']
        elif len(s4grow) != 0:
            morph = t_type_mapping[s4grow['TT']]
        else:
            morph = madtabfil['col2']

        inifil = glob(os.path.join(galdir, "*.ini"))[0]
        config = file_handler.parse_config(inifil)
        coords = SkyCoord(float(config['objra']), float(config['objdec']), unit='deg')
        coords_string = coords.to_string('hmsdms').split(' ')

        ell = float(config['ell'])
        inclination = (np.arccos(1 - ell) * u.radian).to(u.deg).value

        data[i]['name'] = galname
        data[i]['ra'] = coords_string[0]
        data[i]['dec'] = coords_string[1]
        data[i]['morph'] = morph
        data[i]['redshift'] = float(config['redshift'])
        data[i]['reff'] = float(config['reff'])
        data[i]['i'] = inclination

        data[i]['ebv'] = float(config['ebvgal'])

        

def main(args):
    newfile = args.newfile
    galname = args.galname
    bin_method = args.bin_method
    verbose = args.verbose
    thresh = args.thresh
    paper = args.paper
    analysis_plan = defaults.analysis_plans()
    corr_key = 'BETA-CORR'

    ## output directories
    local_data = defaults.get_data_path(subdir='local')
    output_dir = os.path.join(local_data, 'local_outputs', f"{galname}-{bin_method}", corr_key, analysis_plan)
    gal_figures_dir = os.path.join(output_dir, 'figures')
    inspect_figures_dir = os.path.join(gal_figures_dir, 'inspection')
    results_figures_dir = os.path.join(gal_figures_dir, 'results')
    datapath_dict = file_handler.init_datapaths(galname, bin_method, verbose, redshift=True)

    if newfile:
        local_filepath = datapath_dict['LOCAL']
        if local_filepath is not None:
            fpath, fname = os.path.split(local_filepath)
            name, ext = os.path.splitext(fname)
            newfname = f"{name}_old{ext}"

            backupdir = os.path.join(output_dir, 'backup')
            os.makedirs(backupdir, exist_ok=True)

            shutil.move(local_filepath, os.path.join(backupdir, newfname))
            print(f"{fname} moved to {backupdir}")


    mapsfile = datapath_dict['MAPS']
    cubefile = datapath_dict['LOGCUBE']

    with fits.open(cubefile) as cube_hdu:
        binids = cube_hdu['BINID'].data
        spatial_bins = binids[0]
        cube_header = cube_hdu['binid'].header

    with fits.open(mapsfile) as maps_hdu:
        radius = maps_hdu['bin_lwellcoo'].data[1]
        maps_header = maps_hdu['bin_lwellcoo'].header

    ## write radius map and spatial bin id map to local file
    radius_mapdict = {"RADIUS":radius}
    radius_header = file_handler.use_sdss_header(galname, bin_method, "RADIUS", maps_header, desc="radius map", 
                                                 remove_keywords=['C1', 'U1', 'C3', 'U3', 'C4', 'U4'])
    radius_dict = file_handler.standard_map_dict(galname, radius_mapdict, custom_header_dict=radius_header)

    

    spatial_bins_mapdict = {"SPATIAL_BINS":spatial_bins}
    spatial_bins_header = file_handler.use_sdss_header(galname, bin_method, "SPATIAL_BINS", cube_header, "spatial bin IDs",
                                                       remove_keywords=['C2', 'C3', 'C4', 'C5'])
    spatial_bins_dict = file_handler.standard_map_dict(galname, spatial_bins_mapdict, custom_header_dict=spatial_bins_header)

    file_handler.write_maps_file(galname, bin_method, [radius_dict, spatial_bins_dict], verbose=verbose, preserve_standard_order=True)

    ####### REDSHIFT #######
    maps.redshift.redshift_map_dict(galname, bin_method, verbose=verbose)

    ####### S/N NaD #######
    maps.NaI_SNR.NaD_snr_map(galname, bin_method, verbose=verbose)

    ####### EQ W NaD #######
    maps.NaI_EW.measure_EW(galname, bin_method, verbose=verbose)

    ####### Sigma_SFR #######
    maps.SFR.SFR_map(galname, bin_method, verbose=verbose)
 
    ####### MCMC RESULTS #######
    mcmc_results.make_mcmc_results_cube(galname, bin_method, verbose=verbose)

    ####### NaD VELOCITY #######
    maps.NaI_Velocity.make_vmap(galname, bin_method, thresh=thresh, verbose=verbose)
        
    ####### Terminal VELOCITY #######
    maps.NaI_Velocity.make_terminal_vmap(galname, bin_method, verbose=verbose)

    ####### HII #######
    # collaboration.Hii.write_fluxes(galname, bin_method, exists_ok=True, verbose=verbose)
    # hii_dict, hii_header = collaboration.Hii.get_hii_mapdict(galname, bin_method, placeholder=True, verbose=verbose)
    # hii_mapdict = file_handler.standard_map_dict(galname, hii_dict, custom_header_dict=hii_header)

    ####### BPT #######
    maps.BPT.classify_galaxy_bpt(galname, bin_method, verbose=verbose)

    
    # maps.metallicity.make_metallicity_map(galname, bin_method, verbose=verbose)

    ####### BAROLO #######
    collaboration.barolo.analysis_run(galname, bin_method, verbose=verbose)
    
    util.sys_message(f"Analysis Complete!", color='green', verbose=verbose)

    ###### RESULT PLOTTTING #######
    plot_results.make_plots(galname, bin_method, paper=paper, verbose=verbose)

    ## TODO
    ###### INSPECT PLOTTING #######


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    args = get_args()
    main(args)