import os
from astropy.io import fits
import argparse
import numpy as np
from modules import util, file_handler

def write_fluxes(mapsfile, output_file, verbose=False):
    maps = fits.open(mapsfile)

    emline_key = 'EMLINE_GFLUX'

    emlines = maps[emline_key]
    emlines_ivar = maps[f"{emline_key}_IVAR"]
    emlines_mask = maps[f"{emline_key}_MASK"]

    ha = emlines[23]
    ha_ivar = emlines_ivar[23]
    ha_mask = emlines_mask[23]

    hb = emlines[14]
    hb_ivar = emlines_ivar[14]
    hb_mask = emlines_mask[14]

    primary = fits.PrimaryHDU()
    flux_image = fits.ImageHDU(data=np.stack((ha, hb), axis=0), header=emlines.header, name='EMLINE_FLUX')
    ivar_image = fits.ImageHDU(data=np.stack((ha_ivar, hb_ivar), axis=0), header=emlines_ivar.header, name='EMLINE_FLUX_IVAR')
    mask_image = fits.ImageHDU(data=np.stack((ha_mask, hb_mask), axis=0), header=emlines_mask.header, name='EMLINE_MASK')

    new_hdul = fits.HDUList([primary, flux_image, ivar_image, mask_image])

    new_hdul.writeto(output_file, overwrite=True)


def read_in_map(hiimapfile, local_datafile):
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
    repodir = os.path.dirname(os.path.abspath(__file__))
    repo_parentdir = os.path.dirname(repodir)
    hii_dir = os.path.join(repo_parentdir, 'data', 'local', 'hii')

    if args.flag == 0:
        outfil = os.path.join(hii_dir, f"{args.galname}-{args.bin_method}-EMLINE_FLUX.fits")
        filepath_dict = file_handler.init_datapaths(args.galname, args.bin_method, verbose=args.verbose)
        mapfile = filepath_dict['MAPS']

        write_fluxes(mapfile, outfil, verbose=args.verbose)

    else:
        read_in_map()





if __name__ == "__main__":
    args = get_args()
    main(args)