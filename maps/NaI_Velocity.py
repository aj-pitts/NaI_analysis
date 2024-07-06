import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table, join
from glob import glob
import os
import argparse
from tqdm import tqdm

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def make_vmap(mcmc_paths,cube_fil,figpath):
    cube = fits.open(cube_fil)
    binid = cube['BINID'].data[0]

    lamrest = 5897.558

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
    
    bins, inds = np.unique(table['bin'],return_index=True)
    for ID,ind in zip(bins,inds):
        w = binid == ID
        vel_map[w] = table[ind]['velocities']
    
    logging.info('Creating plots.')
    minmax = round(np.std(vel_map),ndigits=-1)
    plotmap = np.copy(vel_map)
    w = (plotmap<-minmax) | (plotmap>minmax)
    plotmap[w] = np.nan

    plt.imshow(plotmap,cmap='coolwarm',vmin=-minmax,vmax=minmax,origin='lower')
    #plt.gca().set_facecolor('lightgray')
    plt.xlabel("Spaxel")
    plt.ylabel("Spaxel")
    plt.colorbar(label=r"$v_{\mathrm{Na I}}\ (\mathrm{km\ s^{-1}})$",fraction=0.0465, pad=0.01)
    
    imname = os.path.join(figpath,f"{args.galname}_velocity-map.png")
    plt.savefig(imname,bbox_inches='tight',dpi=200)
    logging.info(f"Velocity map plot saved to {imname}")
    plt.close()

    flat_vels = table['velocities']
    bin_width = 3.5 * np.std(flat_vels) / (flat_vels.size ** (1/3))
    nbins = (max(flat_vels) - min(flat_vels)) / bin_width
    plt.hist(flat_vels,bins=int(nbins),color='k')
    plt.xlabel(r"$v_{\mathrm{Na I}}\ (\mathrm{km\ s^{-1}})$")
    plt.ylabel(r"$N_{\mathrm{bins}}$")
    imname = os.path.join(figpath,f"{args.galname}_velocity-distribution.png")
    plt.savefig(imname,bbox_inches='tight',dpi=200)
    logging.info(f"Velocity distribution plot saved to {imname}")
    plt.close()

    return vel_map

    
def get_args():
    parser = argparse.ArgumentParser(description="A script to create an Na I velocity map from the MCMC results of a beta-corrected galaxy.")

    parser.add_argument('galname', type=str, help="Input galaxy name.")
    parser.add_argument('bin_method', type=str, help="Input DAP spatial binning method.")

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
    vmap = make_vmap(mcmc_fils,cube_fil,fig_output_path)

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
    main(args)