import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table, join
from glob import glob
import os
import argparse
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def get_EW(datadir):
    ewmap_path = glob(os.path.join(datadir, "**", "*EW-Map.fits"), recursive=True)
    if len(ewmap_path) == 0:
        raise ValueError(f"No EW map data found in {datadir}")
    if len(ewmap_path) > 1:
        print(ewmap_path)
        raise ValueError(f"Multiple EW map .fits files found")
    
    data = fits.getdata(ewmap_path[0])
    return data

def get_sn_cut(ew):
    if ew <= 0.25:
        sn_cut = 40
    elif ew <= 0.5:
        sn_cut = 20
    elif ew <= 0.75:
        sn_cut = 10
    else:
        sn_cut = 5

    return sn_cut

def NaD_snr(wave, flux, ivar):
    windows = [(5865, 5875), (5915, 5925)]
    inds1 = np.where((wave>=windows[0][0]) & (wave<=windows[0][1]))[0]
    inds2 = np.where((wave>=windows[1][0]) & (wave<=windows[1][1]))[0]
    inds = np.concatenate((inds1,inds2))
    
    flux_window = flux[inds]
    ivar_window = ivar[inds]

    sig = 1/np.sqrt(ivar_window)
    w = np.logical_and(np.isfinite(sig), np.isfinite(flux_window))

    return np.median(flux_window[w]/sig[w])


def make_vmap(mapspath, mcmc_paths,cube_fil,figpath):
    cube = fits.open(cube_fil)
    binid = cube['BINID'].data[0]
    flux = cube['FLUX'].data
    ivar = cube['IVAR'].data
    wave = cube['WAVE'].data
    
    lamrest = 5897.558

    ewmap = get_EW(mapspath)

    vel_map = np.zeros(binid.shape)
    table = None

    for i,mcmc_fil in enumerate(tqdm(mcmc_paths,desc="Combining MCMC results")):
        data = fits.open(mcmc_fil)
        data_table = Table(data[1].data)
        data_table.remove_columns(['samples','percentiles'])
        data_table['id'] = np.arange(len(data_table))

        if i == 0:
            table = data_table
            continue
        table = join(table, data_table, join_type='outer')
    
    badsnr = 0
    badew = 0
    bins, inds = np.unique(table['bin'],return_index=True)
    for ID,ind in zip(bins,inds):
        w = binid == ID
        indxs = np.where(binid == ID)

        ew = np.median(ewmap[w])
        if not np.isfinite(ew) or ew<=0:
            badew+=1
            continue

        sn_cut = get_sn_cut(ew)
        flux_bin = flux[:, indxs[0], indxs[1]]
        ivar_bin = ivar[:, indxs[0], indxs[1]]
        sn = NaD_snr(wave, flux_bin[:,0], ivar_bin[:,0])
        if sn<sn_cut:
            warnings.warn(f"NaD S/N in Bin {ID} is {sn}. Threshold is {sn_cut}")
            badsnr+=1
            continue

        vel_map[w] = table[ind]['velocities']
    
    logging.info('Creating plots.')
    minmax = round(2 * np.std(vel_map),ndigits=-1)
    plotmap = np.copy(vel_map)
    #w = (plotmap<-minmax) | (plotmap>minmax)
    plotmap[(plotmap==0) | (plotmap==-999)] = np.nan

    im = plt.imshow(plotmap,cmap='bwr',vmin=-minmax,vmax=minmax,origin='lower',
               extent=[32.4, -32.6,-32.4, 32.6])
    #plt.gca().set_facecolor('lightgray')
    plt.xlabel(r'$\Delta \alpha$ (arcsec)')
    plt.ylabel(r'$\Delta \delta$ (arcsec)')
    
    ax = plt.gca()
    ax.set_facecolor('lightgray')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    cbar = plt.colorbar(im,cax=cax,label=r"$v_{\mathrm{Na I}}\ (\mathrm{km\ s^{-1}})$")
    #plt.colorbar(label=r"$v_{\mathrm{Na I}}\ (\mathrm{km\ s^{-1}})$",fraction=0.0465, pad=0.01)
    
    imname = os.path.join(figpath,f"{args.galname}_velocity-map.{args.imgftype}")
    plt.savefig(imname,bbox_inches='tight',dpi=200)
    logging.info(f"Velocity map plot saved to {imname}")
    plt.close()

    flat_vels = table['velocities']
    badfitcount = np.sum(flat_vels==-999)
    flat_vels = flat_vels[flat_vels!=-999]
    bin_width = 3.5 * np.std(flat_vels) / (flat_vels.size ** (1/3))
    nbins = (max(flat_vels) - min(flat_vels)) / bin_width
    plt.hist(flat_vels,bins=int(nbins),color='k')
    plt.xlabel(r"$v_{\mathrm{Na I}}\ (\mathrm{km\ s^{-1}})$")
    plt.ylabel(r"$N_{\mathrm{bins}}$")
    plt.text(0.05,0.9,f"Removed: {badfitcount}", transform=plt.gca().transAxes)

    imname = os.path.join(figpath,f"{args.galname}_velocity-distribution.{args.imgftype}")
    plt.savefig(imname,bbox_inches='tight',dpi=200)
    logging.info(f"Velocity distribution plot saved to {imname}")
    plt.close()

    cleanvel = vel_map[np.isfinite(vel_map) & (vel_map!=0) & (vel_map!=-999)]
    logging.info(f"Velocity Map Info for {args.galname}")
    print(f"V_max = {np.max(cleanvel)}")
    print(f"V_med = {np.median(cleanvel)}")
    print(f"V_min = {np.min(cleanvel)}")
    print(f"sig_V = {np.std(cleanvel)}")
    print("\n")
    print(f"Bad EWs: {badew}")
    print(f"Bad S/N {badsnr}")
    print(f"Bad MCMC Fits {badfitcount}")
    logging.info("Done.")

    return vel_map

    
def get_args():
    parser = argparse.ArgumentParser(description="A script to create an Na I velocity map from the MCMC results of a beta-corrected galaxy.")

    parser.add_argument('galname', type=str, help="Input galaxy name.")
    parser.add_argument('bin_method', type=str, help="Input DAP spatial binning method.")
    parser.add_argument('--imgftype', type=str, help="Input filetype for output map plots. [pdf/png]", default = "pdf")

    return parser.parse_args()


def main(args):
    # initialize directories and paths
    repodir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    plt.style.use(os.path.join(repodir,"figures.mplstyle"))
    datapath = os.path.abspath(os.path.join(repodir,"data",f"{args.galname}"))

    fig_output_path = os.path.abspath(os.path.join(repodir,"figures","Velocity-map"))
    if not os.path.exists(fig_output_path):
        print(f"Creating filepath: {fig_output_path}")
        os.mkdir(fig_output_path)
    
    map_output_path = os.path.abspath(os.path.join(datapath,"maps"))
    if not os.path.exists(map_output_path):
        print(f"Creating filepath: {map_output_path}")
        os.mkdir(map_output_path)

    cube_path = os.path.join(datapath,f"cube/{args.galname}-{args.bin_method}/BETA-CORR")

    mcmc_path = os.path.join(datapath,"mcmc")
    if not os.path.exists(mcmc_path):
        raise ValueError(f"{mcmc_path} does not exist.")
    
    # obtain mcmc files and cube file
    mcmc_fils = glob(os.path.join(mcmc_path,"**","*.fits"),recursive=True)
    if len(mcmc_fils)==0:
        raise ValueError(f"MCMC output files not found in {mcmc_path}")
    
    cube_fils = glob(os.path.join(cube_path,"**","*.fits"),recursive=True)
    cube_fil = None
    for fil in cube_fils:
        if 'LOGCUBE' in fil:
            cube_fil = fil
            break

    if cube_fil is None:
        print(cube_fils)
        raise ValueError(f"Cube file not found in {cube_path}")
    
    
    # call the fn
    vmap = make_vmap(map_output_path,mcmc_fils,cube_fil,fig_output_path)

    # write the data
    map_output_fil = os.path.join(map_output_path,f"{args.galname}_Velocity-map.fits")
    logging.info(f"Writing velocity map to {map_output_fil}")

    hdu = fits.PrimaryHDU(vmap)
    hdu.header['DESC'] = f"Na I absorption velocity map for {args.galname}"
    hdu.header['UNITS'] = "km/s"

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