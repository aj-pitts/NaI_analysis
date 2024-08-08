import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import argparse
import configparser
import os
from glob import glob
from tqdm import tqdm
import sys
import logging
import warnings
logger = logging.getLogger()
logger.setLevel(logging.INFO)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../modules/")))
from util import clean_ini_file
from interactive_plot import make_bokeh_map


def make_EW_map(cubefil,mapfil,z_guess,savepath,vmin=-0.2,vmax=4,bad_bins=False,show_warnings=True):
    c = 2.998e5
    
    if isinstance(z_guess, str):
        z_guess = float(z_guess)
        
    if bad_bins:
        bbins=[]
        
    cube = fits.open(cubefil)
    Map = fits.open(mapfil)
    
    flux = cube['FLUX'].data
    wave = cube['WAVE'].data
    ivar = cube['IVAR'].data
    model = cube['MODEL'].data
    
    stellarvel = Map['STELLAR_VEL'].data
    binid = Map['BINID'].data[0]
    uniqids = np.unique(binid)
    
    region = 5880, 5910
    
    
    l, ny, nx = flux.shape
    ewmap = np.zeros((ny,nx))
    wavecube = np.zeros(flux.shape)

    logging.info('Constructing equivalent width map.')
    for ID in uniqids[1:]:
        inds = np.where(binid == ID)
        w = binid == ID

        ## get the stellar velocity of the bin
        sv = stellarvel[w][0]
            
        ## Calculate redshift
        z = (sv * (1+z_guess))/c + z_guess

        # shift wavelengths to restframe
        restwave = wave / (1+z)

        for y,x in zip(inds[0],inds[1]):
            wavecube[np.arange(len(restwave)),y,x] = restwave

        # define wavelength boundaries and slice flux, model, and wavelength arrays
        inbounds = np.where((restwave>region[0]) & (restwave<region[1]))[0]
        Lam = restwave[inbounds]
        fluxbound = flux[inbounds,:,:]
        modelbound = model[inbounds,:,:]       

        ## check the flux and model in the bin
        
        # slice the flux/model to just those in the current bin
        fluxbin = fluxbound[:,inds[0],inds[1]]
        modelbin = modelbound[:,inds[0],inds[1]]

        if abs(sv) > 4 * np.std(stellarvel):
            if show_warnings:
                warnings.warn(f"Stellar velocity in Bin ID {ID} beyond 4 standard deviations. Bin {ID} EW set to Nan",UserWarning,
                             stacklevel=2)
            ewmap[w] = np.nan
            if bad_bins:
                bbins.append(ID)
            continue

        # make sure flux is identical throughout the bin
        if not np.all(fluxbin == fluxbin[:,0][:,np.newaxis]):
            if show_warnings:
                warnings.warn(f"Fluxes in Bin {ID} are not identical. Bin {ID} EW set to NaN",UserWarning,
                             stacklevel=2)
            ewmap[w] = np.nan
            if bad_bins:
                bbins.append(ID)
            continue
            
        # repeat comparison for the model
        if not np.all(modelbin == modelbin[:,0][:,np.newaxis]):
            if show_warnings:
                warnings.warn(f"Stellar models in Bin {ID} not identical. Bin {ID} EW set to NaN",UserWarning,
                             stacklevel=2)
            ewmap[w] = np.nan
            if bad_bins:
                bbins.append(ID)
            continue
         
        F = fluxbin[:,0]
        M = modelbin[:,0]
        
        if not all(F>=0) or not all(M>=0):
            if show_warnings:
                warnings.warn(f"Flux or model arrays in Bin {ID} contain values < 0. Logging Bin ID.", UserWarning,
                             stacklevel=2)
            #ewmap[w] = np.nan
            if bad_bins:
                bbins.append(ID)
            #continue
            
            
        # create dlambda array
        dLam = np.diff(Lam)
        dLam = np.insert(dLam, 0, dLam[0])
        
        # exclude models equal to zero to avoid nan in calculation
        nonzero = (M != 0) & (F != 0)
        cont = np.ones(np.sum(nonzero))
        W = np.sum( (cont - (F[nonzero])/M[nonzero]) * dLam[nonzero] )
        ewmap[w] = W
    
    logging.info('Creating plots.')
    
    flatew = ewmap.flatten()
    w = (flatew != 0) & (np.isfinite(flatew))
    flatewcleaned = flatew[w]
    
    bin_width = 3.5 * np.std(flatewcleaned) / (flatewcleaned.size ** (1/3))
    nbins = (max(flatewcleaned) - min(flatewcleaned)) / bin_width
    
    plt.hist(flatewcleaned,bins=int(nbins),color='k')
    plt.xlim(-5,5)
    plt.xlabel(r'$\mathrm{EW_{Na\ I}\ (\AA)}$')
    plt.ylabel(r'$N_{\mathrm{spax}}$')
    
    im2name = f"{args.galname}-EW_distribution.png"
    output = os.path.join(savepath,im2name)
    plt.savefig(output,bbox_inches='tight',dpi=150)
    logging.info(f"EW distriubtion plot saved to {output}")
    plt.close()
    
    
    plotmap = np.copy(ewmap)
    plotmap[(plotmap==0) | (plotmap>vmax) | (plotmap<vmin)] = np.nan
    
    nvmax = np.median(plotmap[np.isfinite(plotmap)]) + np.std(plotmap[np.isfinite(plotmap)])
    if vmax < nvmax:
        vmax = np.round(nvmax)
 
    plt.imshow(plotmap,origin='lower',cmap='rainbow',vmin=vmin,vmax=vmax,
           extent=[32.4, -32.6,-32.4, 32.6])
    plt.colorbar(label=r'$\mathrm{EW_{Na\ I}\ (\AA)}$',fraction=0.0465, pad=0.01)
    plt.gca().set_facecolor('lightgray')
    plt.xlabel(r'$\Delta \alpha$ (arcsec)')
    plt.ylabel(r'$\Delta \delta$ (arcsec)')
    
    im1name = f"{args.galname}-EW_map.png"
    output = os.path.join(savepath,im1name)
    plt.savefig(output,bbox_inches='tight',dpi=200)
    logging.info(f"EW map plot saved to {output}")
    plt.close()
    
    if args.bokeh:
        logging.info("Creating BOKEH plot.")
        keyword = f"{args.galname}-EW-bokeh"
        make_bokeh_map(flux, model, ivar, wavecube, ewmap, binid, savepath, keyword)


    if bad_bins:
        return ewmap, bbins
    
    return ewmap


def get_args():
    
    parser = argparse.ArgumentParser(description="A script to create an equivalent width map of ISM Na I for beta-corrected DAP outputs.")
    
    parser.add_argument('galname',type=str,help='Input galaxy name.')
    parser.add_argument('bin_method',type=str,help='Input DAP patial binning method.')
    parser.add_argument('--redshift',type=str,help='Input galaxy redshift guess.',default=None)
    parser.add_argument('--bokeh', type=bool, help="Input [True/False] for creating a Bokeh interactive plot.", default = False)
    
    return parser.parse_args()



def main(args):
    ## Full path of the script directory
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    plt.style.use(os.path.join(script_dir,"figures.mplstyle"))
    ## Full path of the DAP outputs
    data_dir = os.path.join(script_dir, "data")

    ## Path to the specific galaxy and binning method input
    cube_dir = f"{args.galname}"
    cubepath_bc = os.path.join(data_dir,cube_dir,"cube",f"{args.galname}-{args.bin_method}")
    
    ## Path to save the plots
    savepath = os.path.join(script_dir,"figures",'EW-Map')
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    
    ## All fits files within cube path
    fils = glob(os.path.join(cubepath_bc,"BETA-CORR",'**','*.fits'),recursive=True)
    
    ## Finds the logcube and map files from the DAP output
    cubefil = None
    mapfil = None
    for fil in fils:
        if 'LOGCUBE' in fil:
            cubefil = fil
        if 'MAPS' in fil:
            mapfil = fil
    
    if cubefil is None:
        raise ValueError(f'Cube File not found in {cubepath_bc}')
    if mapfil is None:
        raise ValueError(f'Map File not found in {cubepath_bc}')
    
    
    redshift = args.redshift

    
    if redshift is None:
        ## Get the redshift guess from the .ini file
        config_dir = os.path.join(data_dir,cube_dir,"config")
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
    

    W_equiv,bins = make_EW_map(cubefil,mapfil,redshift,savepath,bad_bins=True)
    
    mapspath = os.path.join(data_dir,cube_dir,"maps")
    if not os.path.exists(mapspath):
        os.mkdir(mapspath)

    savefits = os.path.join(mapspath,f"{cube_dir}_EW-Map.fits")

    logging.info(f"Writing EW Map data to {savefits}")
    
    hdu = fits.PrimaryHDU(W_equiv)
    hdu.header['DESC'] = "ISM Na I equivalent width map"
    hdu.header['UNITS'] = "Angstrom"
    
    hdu2 = fits.ImageHDU(bins)
    hdu2.header['DESC'] = "DAP spatial bin IDs for bad bins"
    
    hdul = fits.HDUList([hdu,hdu2])
    hdul.writeto(savefits,overwrite=True)
    logging.info('Done.')

    
    
if __name__ == "__main__":
    args = get_args()
    main(args)