import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table, join
from glob import glob
import os
import argparse
from tqdm import tqdm
import sys
import astropy.units as u
import warnings
import configparser

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../modules/")))
from util import check_filepath, clean_ini_file


def BPT(map_fil, fig_output):
    logging.info("Obtaining Maps.")
    Map = fits.open(map_fil)
    emlines = Map['EMLINE_SFLUX'].data
    ivars = Map['EMLINE_SFLUX_IVAR'].data
    binid = Map['BINID'].data[0]


    ha = emlines[23]
    hb = emlines[14]

    oiii = emlines[15] #oiii 4690
    #oiii = emlines[16] #oiii 5007 

    sii = emlines[25] # sii 6718
    #sii = emlines[26] # sii 6732

    oi = emlines[20] # oi 6302
    #oi = emlines[21] # oi 6365

    


def get_args():
    parser = argparse.ArgumentParser(description="A script to create a SFR map from the DAP emline results of a beta-corrected galaxy.")

    parser.add_argument('galname', type=str, help="Input galaxy name.")
    parser.add_argument('bin_method', type=str, help="Input DAP spatial binning method.")

    return parser.parse_args()



def main(args):
    logging.info("Intitalizing directories and paths.")
    # intialize directories and paths
    repodir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    plt.style.use(os.path.join(repodir,"figures.mplstyle"))
    datapath = os.path.join(repodir,"data",f"{args.galname}")

    fig_output_path = os.path.join(repodir,"figures","BPT")
    check_filepath(fig_output_path)

    map_output_path = os.path.join(datapath,"maps")
    check_filepath(map_output_path)

    cube_path = os.path.join(datapath,"cube",f"{args.galname}-{args.bin_method}","BETA-CORR")
    check_filepath(cube_path,mkdir=False)

    cube_fils = glob(os.path.join(cube_path,"**","*.fits"),recursive=True)
    map_fil = None
    for fil in cube_fils:
        if 'MAPS' in fil:
            map_fil = fil
            break

    if map_fil is None:
        print("Glob found:\n")
        print(cube_fils)
        raise ValueError(f"Maps file not found in {cube_path}")    
    
    logging.info("Done.")
    
    

    map_output_fil = os.path.join(map_output_path, f"{args.galname}_SFR-Map.fits")
    logging.info(f"Writing SFR map to {map_output_fil}")

    hdu = fits.PrimaryHDU(sfr)
    hdu.header['DESC'] = f"SFR Map for galaxy {args.galname}"
    hdu.header['UNITS'] = f"M_sun / yr / spaxel"
    hdul = fits.HDUList([hdu])
    hdul.writeto(map_output_fil,overwrite=True)
    logging.info("Done.")



if __name__ == "__main__":
    args = get_args()
    main(args)