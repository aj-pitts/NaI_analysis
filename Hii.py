import os
from astropy.io import fits
import argparse
import numpy as np
from modules import util, file_handler, defaults

def write_fluxes(mapsfile, local_maps_file, output_file, verbose=False):
    if not os.path.exists(local_maps_file):
        util.verbose_warning(verbose, f'Local maps file does not exist: {local_maps_file}\nIgnoring...')
        local_maps = None
    else:
        local_maps = fits.open(local_maps_file)
    
    maps = fits.open(mapsfile)

    emline_key = 'EMLINE_GFLUX'

    util.verbose_print(args.verbose, f"Aquiring {emline_key} values from {mapsfile}")

    emlines = maps[emline_key]
    emlines_ivar = maps[f"{emline_key}_IVAR"]
    emlines_mask = maps[f"{emline_key}_MASK"]

    ha = emlines.data[23]
    ha_ivar = emlines_ivar.data[23]
    ha_mask = emlines_mask.data[23]

    hb = emlines.data[14]
    hb_ivar = emlines_ivar.data[14]
    hb_mask = emlines_mask.data[14]

    util.verbose_print(args.verbose, f"Creating HDU list...")

    primary = fits.PrimaryHDU()
    flux_image = fits.ImageHDU(data=np.stack((ha, hb), axis=0), header=emlines.header, name='EMLINE_FLUX')
    ivar_image = fits.ImageHDU(data=np.stack((ha_ivar, hb_ivar), axis=0), header=emlines_ivar.header, name='EMLINE_FLUX_IVAR')
    mask_image = fits.ImageHDU(data=np.stack((ha_mask, hb_mask), axis=0), header=emlines_mask.header, name='EMLINE_MASK')

    new_hdul = fits.HDUList([primary, flux_image, ivar_image, mask_image])

    util.verbose_print(args.verbose, f"Writing data to {output_file}")
    
    new_hdul.writeto(output_file, overwrite=True)

    util.verbose_print(args.verbose, "Done.")

## TODO: write the hii region mask into the "main" galaxy fits file
def read_in_map(hiimapfile, local_datafile, verbose=False):
    return None


def get_args():
    parser = argparse.ArgumentParser(description="A script to handle the HII region collaboration.")

    parser.add_argument('galname', type=str, help="Input galaxy name.")
    parser.add_argument('bin_method', type=str, help="Input DAP spatial binning method.")
    parser.add_argument('flag', type=int, help="Flag to specify whether to write fluxes to file or read in H II mask.\n0: write; 1: read.")
    parser.add_argument('-v','--verbose', help = "Print verbose outputs (default: False)", action='store_true', default = False)

    return parser.parse_args()


def main(args):
    ## init necessary paths
    analysis_plan = defaults.analysis_plans()

    local_data_dir = defaults.get_data_path('local')
    hii_dir = os.path.join(local_data_dir, 'hii')
    util.check_filepath(hii_dir, mkdir=True, verbose=args.verbose)

    galaxy_dir = os.path.join(hii_dir, f"{args.galname}-{args.bin_method}-{analysis_plan}")
    util.check_filepath(galaxy_dir, mkdir=True, verbose=args.verbose)
    
    

    if args.flag == 0:
        outfil = os.path.join(galaxy_dir, f"{args.galname}-{args.bin_method}-{analysis_plan}-EMLINE_FLUX.fits")
        filepath_dict = file_handler.init_datapaths(args.galname, args.bin_method, verbose=args.verbose)
        mapfile = filepath_dict['MAPS']

        write_fluxes(mapfile, outfil, verbose=args.verbose)

    else:
        corr_key = "BETA-CORR"
        local_outputs = os.path.join(local_data_dir, 'local_outputs')
        galaxy_local_dir = os.path.join(local_outputs, f"{args.galname}-{args.bin_method}", corr_key, analysis_plan)
        local_maps_file = f"{args.galname}-{args.bin_method}-local_maps.fits"
        local_maps_filepath = os.path.join(galaxy_local_dir, local_maps_file)
        if not os.path.isfile(local_maps_filepath):
            raise ValueError(f"File does not exist: {local_maps_filepath}")
        
        hii_file = f"{args.galname}-{args.bin_method}-Hii.fits"
        hii_filepath = os.path.join(hii_dir, hii_file)
        if not os.path.isfile(hii_filepath):
            raise ValueError(f"File does not exist: {hii_filepath}")

        read_in_map(hii_filepath, local_maps_filepath, verbose=args.verbose)


if __name__ == "__main__":
    args = get_args()
    main(args)