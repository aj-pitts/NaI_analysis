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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.coordinates import Angle

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../modules/")))
from util import check_filepath, clean_ini_file


def correct_dust(F_Ha, F_Hb, HaHb_ratio = 2.87, Rv = 3.1, k_Ha = 2.45, k_Hb = 3.65):
    """
    Returns the dust corrected H alpha flux.
    """
    E_BV = np.zeros(F_Ha.shape)
    F_corr = np.zeros(F_Ha.shape)

    w = (F_Ha > 0) & (np.isfinite(F_Ha)) & (F_Hb > 0) & (np.isfinite(F_Hb))
    E_BV[w] = (2.5 / (k_Hb - k_Ha)) * np.log10((F_Ha[w]/F_Hb[w])/HaHb_ratio)
    A_gas = E_BV * k_Ha

    power = 0.4 * A_gas
    F_corr[w] = F_Ha[w] * (10 ** power[w])
    F_corr[~w] = np.nan

    return F_corr

def SFR_map(map_fil,redshift,figpath):
    if isinstance(redshift, str):
        redshift = float(redshift)

    logging.info("Obtaining Maps.")
    Map = fits.open(map_fil)
    stellar_vel = Map['STELLAR_VEL'].data
    #stellar_vel_ivar = Map['STELLAR_VEL_IVAR'].data
    emlines = Map['EMLINE_SFLUX'].data
    ivars = Map['EMLINE_SFLUX_IVAR'].data
    binid = Map['BINID'].data[0]

    F_ha = emlines[23]
    err_ha = ivars[23]
    F_hb = emlines[14]
    err_hb = ivars[14]
    
    F_ha_corr = correct_dust(F_ha, F_hb)


    H0 = 70 * u.km / u.s / u.Mpc
    c = 2.998e5 * u.km / u.s

    sfrmap = np.zeros(binid.shape)
    sfrdensitymap = np.zeros(binid.shape)

    uniqids = np.unique(binid)
    for ID in tqdm(uniqids[1:], desc="Constructing SFR map"):
        w = (binid == ID)
        ha_flux = np.median(F_ha_corr[w])
        if ha_flux <= 0:
            continue

        sv = np.median(stellar_vel[w])
        z = ((sv * (1+redshift))/c.value + redshift)
        D = (c * z / H0)
        theta = Angle(0.2, unit='arcsec')
        s = D * theta.radian
        
        luminosity = 4 * np.pi * (D.to(u.cm).value)**2 * ha_flux * 1e-17
        SFR = np.log10(luminosity) - 41.27

        if s.to(u.pc).value > 300:
            continue
        sigma_SFR = np.log10( (10**SFR) / ((s.to(u.kpc).value)**2) )

        sfrmap[w] = SFR
        sfrdensitymap[w] = sigma_SFR


    ### SFR Map
    logging.info("Creating plots.")
    w = np.isfinite(sfrmap) & (sfrmap!=0)
    vmin = int(np.median(sfrmap[w]) - np.std(sfrmap[w]))
    vmax = int(np.median(sfrmap[w]) + np.std(sfrmap[w]))
    plotmap = np.copy(sfrmap)

    #w = (plotmap<vmin) | (plotmap>vmax)
    #plotmap[w] = np.nan
    plotmap[plotmap==0] = np.nan

    im = plt.imshow(plotmap,cmap='rainbow',vmin=vmin,vmax=vmax,origin='lower',
               extent=[32.4, -32.6,-32.4, 32.6])
    plt.xlabel(r'$\alpha$ (arcsec)')
    plt.ylabel(r'$\delta$ (arcsec)')
    ax = plt.gca()
    ax.set_facecolor('lightgray')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    cbar = plt.colorbar(im,cax=cax,label=r"$\mathrm{log\ SFR_{H\alpha}\ (M_{\odot}\ yr^{-1}\ spaxel^{-1})}$")
    #plt.colorbar(label=r"$\mathrm{SFR\ (M_{\odot}\ yr^{-1}\ spaxel^{-1})}$",fraction=0.0465, pad=0.01)
    
    imname = os.path.join(figpath,f"{args.galname}_SFR-map.{args.imgftype}")
    plt.savefig(imname,bbox_inches='tight',dpi=200)
    logging.info(f"SFR map plot saved to {imname}")
    plt.close()


    ### SFR Density
    w = np.isfinite(sfrdensitymap) & (sfrdensitymap != 0)
    vmin = int(np.median(sfrdensitymap[w]) - 2 * np.std(sfrdensitymap[w]))
    vmax = int(np.median(sfrdensitymap[w]) + 2 * np.std(sfrdensitymap[w]))
    plotmap = np.copy(sfrdensitymap)

    #w = (plotmap<vmin) | (plotmap>vmax)
    #plotmap[w] = np.nan
    plotmap[plotmap==0] = np.nan

    plt.imshow(plotmap,cmap='rainbow',origin='lower',vmin=vmin,vmax=vmax,
               extent=[32.4, -32.6,-32.4, 32.6])
    plt.gca().set_facecolor('lightgray')
    plt.xlabel(r'$\alpha$ (arcsec)')
    plt.ylabel(r'$\delta$ (arcsec)')
    plt.colorbar(label=r"$\mathrm{\Sigma_{SFR}\ (M_{\odot}\ yr^{-1}\ kpc^{-2}\ spaxel^{-1})}$",fraction=0.0465, pad=0.01)
    
    imname = os.path.join(figpath,f"{args.galname}_SFR-density-map.{args.imgftype}")
    plt.savefig(imname,bbox_inches='tight',dpi=200)
    logging.info(f"SFR surface density map plot saved to {imname}")
    plt.close()


    ### SFR Hist
    w = np.isfinite(sfrmap)
    finite_map = sfrmap[w]
    flatmap = finite_map.flatten()
    flatmap = flatmap[flatmap!=0]
    bin_width = 3.5 * np.std(flatmap) / (flatmap.size ** (1/3))
    nbins = (max(flatmap) - min(flatmap)) / bin_width
    plt.hist(flatmap,bins=int(nbins),color='k')
    plt.xlabel(r"$\mathrm{SFR\ (M_{\odot}\ yr^{-1}\ spaxel^{-1})}$")
    plt.ylabel(r"$N_{\mathrm{bins}}$")
    plt.xlim(np.median(flatmap) - 7* np.std(flatmap), np.median(flatmap) + 7 * np.std(flatmap))
    imname = os.path.join(figpath,f"{args.galname}_SFR-distribution.{args.imgftype}")
    plt.savefig(imname,bbox_inches='tight',dpi=200)
    logging.info(f"SFR distribution plot saved to {imname}")
    plt.close()

    return sfrmap

def get_args():
    parser = argparse.ArgumentParser(description="A script to create a SFR map from the DAP emline results of a beta-corrected galaxy.")

    parser.add_argument('galname', type=str, help="Input galaxy name.")
    parser.add_argument('bin_method', type=str, help="Input DAP spatial binning method.")
    parser.add_argument('--imgftype', type=str, help="Input filetype for output map plots. [pdf/png]", default = "pdf")
    parser.add_argument('--redshift',type=str,help='Input galaxy redshift guess.',default=None)

    return parser.parse_args()

def main(args):
    logging.info("Intitalizing directories and paths.")
    # intialize directories and paths
    repodir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    plt.style.use(os.path.join(repodir,"figures.mplstyle"))
    datapath = os.path.join(repodir,"data",f"{args.galname}")

    fig_output_path = os.path.join(repodir,"figures","SFR-map")
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

    redshift = args.redshift
    
    if redshift is None:
        ## Get the redshift guess from the .ini file
        config_dir = os.path.join(datapath,"config")
        ini_fil = glob(f"{config_dir}/*.ini")
        config_fil = ini_fil[0]
        if len(ini_fil)>1:
            warnings.warn(f"Multiple configuration files found in {config_dir}.",UserWarning)
            for fil in ini_fil:
                if 'cleaned' in fil:
                    config_fil = fil

            print(f"Defauling configuration file to {config_fil}.")

        config = configparser.ConfigParser()
        parsing = True
        while parsing:
            try:
                config.read(config_fil)
                parsing = False
            except configparser.Error as e:
                print(f"Error parsing file: {e}")
                print(f"Cleaning {config_fil}")
                clean_ini_file(config_fil, overwrite=True)


        redshift = config['default']['z']
        print(f"Redshift z={redshift} found in {config_fil}.")
    

    logging.info("Constructing SFR map.")
    sfr = SFR_map(map_fil, redshift, fig_output_path)

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
    if args.imgftype == "pdf" or args.imgftype == "png":
        pass
    else:
        raise ValueError(f"{args.imgftype} not a valid value for the output image filetype.\nAccepted formats [pdf/png]\nDefault: pdf")
    main(args)