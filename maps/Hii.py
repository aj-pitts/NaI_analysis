import os
from astropy.io import fits
import argparse
import numpy as np
from modules import util, file_handler, defaults

def write_fluxes(galname, bin_method, exists_ok = True, verbose=False):
    util.verbose_print(verbose, f"Checking for Hii EMLINE fluxes...")
    analysis_plan = defaults.analysis_plans()
    local_data_dir = defaults.get_data_path('local')

    hii_dir = os.path.join(local_data_dir, 'hii')
    util.check_filepath(hii_dir, mkdir=True, verbose=verbose)

    galaxy_dir = os.path.join(hii_dir, f"{galname}-{bin_method}-{analysis_plan}")
    util.check_filepath(galaxy_dir, mkdir=True, verbose=verbose)

    output_file = os.path.join(galaxy_dir, f"{galname}-{bin_method}-{analysis_plan}-EMLINE_FLUX.fits")
    if exists_ok:
        if os.path.isfile(output_file):
            util.verbose_print(verbose, f"File exists: {output_file}\nSkipping step")
            return

    filepath_dict = file_handler.init_datapaths(galname, bin_method, verbose=verbose)
    mapsfile = filepath_dict['MAPS']
    
    maps = fits.open(mapsfile)

    emline_key = 'EMLINE_GFLUX'

    emlines = maps[emline_key]
    emlines_ivar = maps[f"{emline_key}_IVAR"]
    emlines_mask = maps[f"{emline_key}_MASK"]

    ha = emlines.data[23]
    ha_ivar = emlines_ivar.data[23]
    ha_mask = emlines_mask.data[23]

    hb = emlines.data[14]
    hb_ivar = emlines_ivar.data[14]
    hb_mask = emlines_mask.data[14]

    primary = fits.PrimaryHDU()
    flux_image = fits.ImageHDU(data=np.stack((ha, hb), axis=0), header=emlines.header, name='EMLINE_FLUX')
    ivar_image = fits.ImageHDU(data=np.stack((ha_ivar, hb_ivar), axis=0), header=emlines_ivar.header, name='EMLINE_FLUX_IVAR')
    mask_image = fits.ImageHDU(data=np.stack((ha_mask, hb_mask), axis=0), header=emlines_mask.header, name='EMLINE_MASK')

    new_hdul = fits.HDUList([primary, flux_image, ivar_image, mask_image])

    util.verbose_print(args.verbose, f"Writing EMLINE data to {output_file}")
    
    new_hdul.writeto(output_file, overwrite=True)

    util.verbose_print(args.verbose, "Done.")


def get_hii_map(galname, bin_method, placeholder=False, verbose=False):
    ## init necessary paths
    analysis_plan = defaults.analysis_plans()
    corr_key = "BETA-CORR"

    local_data_dir = defaults.get_data_path('local')
    hii_dir = os.path.join(local_data_dir, 'hii')
    hii_subdir = os.path.join(hii_dir, 'hii_region_masks')
    util.check_filepath(hii_subdir, verbose=verbose)

    fname = f"{galname.lower()}_{bin_method.lower()}_hii_mask.fits"

    hii_mapfile = os.path.join(hii_subdir, fname)
    if os.path.isfile(hii_mapfile):
        hiimap = fits.getdata(hii_mapfile)
        return hiimap
    else:
        if not placeholder:
            raise ValueError(f"File does not exist: {hii_mapfile}")
        else:
            util.verbose_print(verbose, f"No Hii data found for {galname}, returning empty array")
            filedict = file_handler.init_datapaths(galname, bin_method, verbose=False, redshift=False)
            cubefil = filedict['LOGCUBE']
            hdul = fits.open(cubefil)
            spatial_bins = hdul['BINID'].data[0]
            return np.empty_like(spatial_bins)
            




def get_hii_mapdict(galname, bin_method, verbose=False):
    hdu_name = "HII"
    hiimap = get_hii_map(galname, bin_method, verbose=verbose)
    hii_header_dict = {
        hdu_name:{
            "DESC":"Unique IDs of spatially identified Hii regions",
            "EXTNAME":(hdu_name, "Extension name"),
            "AUTHOR":("Ryan Rickards Vaught","")
        }
    }  
    hii_mapdict = {hdu_name:hiimap}

    return hii_mapdict, hii_header_dict



def get_args():
    parser = argparse.ArgumentParser(description="A script to handle the HII region collaboration.")

    parser.add_argument('galname', type=str, help="Input galaxy name.")
    parser.add_argument('bin_method', type=str, help="Input DAP spatial binning method.")
    parser.add_argument('flag', type=int, help="Flag to specify whether to write fluxes to file or read in H II mask.\n0: write; 1: read.")
    parser.add_argument('-v','--verbose', help = "Print verbose outputs (default: False)", action='store_true', default = False)

    return parser.parse_args()


def main(args):
    if args.flag==0:
        write_fluxes(args.galname, args.bin_method, args.verbose)
    elif args.flag==1:
        get_hii_map(args.galanem, args.bin_method, args.verbose)
        print('Hii region map exists')

    else:
        print('Invalid Flag input')


if __name__ == "__main__":
    args = get_args()
    main(args)